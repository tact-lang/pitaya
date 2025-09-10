"""Top-level runner that wires config, orchestrator creation, and mode dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import argparse

from rich.console import Console

from ...config import (
    get_default_config,
    load_dotenv_config,
    load_env_config,
    load_global_config,
    merge_config,
)
from ...exceptions import DockerError, OrchestratorError, ValidationError
from ...orchestration import Orchestrator
from ...shared import AuthConfig, ContainerLimits, RetryConfig
from .auth import get_auth_config
from .config_loader import build_cli_config, load_project_config
from .preflight import perform_preflight_checks
from .validation import validate_full_config
from . import headless as headless_run
from . import tui_runner

__all__ = ["run"]


def _make_run_id(resume: Optional[str]) -> str:
    if resume:
        return resume
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    import uuid as _uuid

    return f"run_{ts}_{_uuid.uuid4().hex[:8]}"


def _merge_full_config(args) -> Dict[str, Any]:
    cli = build_cli_config(args)
    env = load_env_config()
    dotenv = load_dotenv_config()
    project = load_project_config(getattr(args, "config", None))
    defaults = get_default_config()
    global_cfg = load_global_config()
    return merge_config(
        cli, env, dotenv, project, merge_config({}, {}, {}, global_cfg or {}, defaults)
    )


def _container_limits(cfg: Dict[str, Any]) -> ContainerLimits:
    r = cfg.get("runner", {})
    mem = r.get("memory_limit", 4)
    mem_gb = (
        int(mem[:-1])
        if isinstance(mem, str) and str(mem).lower().endswith("g")
        else int(mem)
    )
    return ContainerLimits(
        cpu_count=int(r.get("cpu_limit", 2)), memory_gb=mem_gb, memory_swap_gb=mem_gb
    )


def _parallel(cfg: Dict[str, Any]) -> tuple[int, int]:
    import os as _os

    def _cpu_default() -> int:
        try:
            return max(2, int(_os.cpu_count() or 2))
        except (TypeError, ValueError):
            return 2

    orch = cfg.get("orchestration", {})
    total = orch.get("max_parallel_instances", "auto")
    if isinstance(total, str) and total.lower() == "auto":
        total_val = _cpu_default()
    else:
        total_val = max(1, int(total))
    start = orch.get("max_parallel_startup", "auto")
    if isinstance(start, str) and start.lower() == "auto":
        start_val = min(10, total_val)
    else:
        start_val = max(1, int(start))
    return total_val, min(start_val, total_val)


def _build_orchestrator(
    cfg: Dict[str, Any], auth: AuthConfig, args=None
) -> Orchestrator:
    total, start = _parallel(cfg)
    orch = cfg.get("orchestration", {})
    runr = cfg.get("runner", {})

    # Collect agent CLI passthrough args (parity with pre-refactor CLI)
    agent_args = None
    try:
        import shlex as _shlex

        collected: list[str] = []
        if args is not None and getattr(args, "agent_cli_arg", None):
            collected.extend([str(a) for a in args.agent_cli_arg if a is not None])
        if args is not None and getattr(args, "agent_cli_args_str", None):
            collected.extend(_shlex.split(str(args.agent_cli_args_str)))
        agent_args = collected or None
    except (AttributeError, ValueError, ImportError):
        agent_args = None

    return Orchestrator(
        max_parallel_instances=total,
        max_parallel_startup=start,
        state_dir=Path(orch.get("state_dir", Path("./pitaya_state"))),
        logs_dir=Path(orch.get("logs_dir", Path("./logs"))),
        container_limits=_container_limits(cfg),
        retry_config=RetryConfig(max_attempts=3),
        auth_config=auth,
        snapshot_interval=int(orch.get("snapshot_interval", 30)),
        event_buffer_size=int(orch.get("event_buffer_size", 10000)),
        runner_timeout_seconds=int(runr.get("timeout", 3600)),
        default_network_egress=str(runr.get("network_egress", "online")),
        branch_namespace=str(orch.get("branch_namespace", "hierarchical")),
        allow_overwrite_protected_refs=bool(
            orch.get("allow_overwrite_protected_refs", False)
        ),
        default_plugin_name=str(cfg.get("plugin_name", "claude-code")),
        default_model_alias=str(cfg.get("model", "sonnet")),
        default_docker_image=runr.get("docker_image"),
        default_agent_cli_args=agent_args,
        force_commit=bool(runr.get("force_commit", False)),
        randomize_queue_order=bool(orch.get("randomize_queue_order", False)),
    )


async def run(console: Console, args: argparse.Namespace) -> int:
    run_id = _make_run_id(getattr(args, "resume", None))
    # Restore --json convenience: implies headless JSON output
    if getattr(args, "json", False):
        args.no_tui = True
        args.output = "json"
    # Default to headless in non-TTY/CI unless explicitly overridden
    try:
        if not sys.stdout.isatty() and not getattr(args, "json", False):
            args.no_tui = True
    except (AttributeError, OSError):
        pass

    # Validate Docker connectivity early (matches previous behavior)
    try:
        from ...utils.platform_utils import validate_docker_setup

        ok, _err = validate_docker_setup()
        if not ok:
            console.print("[red]cannot connect to docker daemon[/red]")
            console.print("Try:")
            console.print("  • start Docker Desktop / system service")
            console.print("  • check $DOCKER_HOST")
            console.print("  • run: docker info")
            return 1
    except ImportError:
        # If platform utils not available, continue; orchestrator will surface errors
        pass
    if not getattr(args, "resume", None):
        if not perform_preflight_checks(console, args):
            return 1

    full_config = _merge_full_config(args)
    if not validate_full_config(console, full_config, args):
        return 1

    auth_cfg = get_auth_config(args, full_config)
    orch = _build_orchestrator(full_config, auth_cfg, args)
    await orch.initialize()

    try:
        if getattr(args, "no_tui", False):
            return await headless_run.run_headless(
                console, orch, args, full_config, run_id
            )
        return await tui_runner.run_tui(console, orch, args, full_config, run_id)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — shutting down gracefully[/yellow]")
        return 2
    except ValueError as e:
        console.print(f"[red]Invalid arguments: {e}[/red]")
        return 2
    except (OrchestratorError, DockerError, ValidationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print_exception()
        return 1
    finally:
        try:
            await orch.shutdown()
        except (OrchestratorError, RuntimeError, OSError):
            pass
