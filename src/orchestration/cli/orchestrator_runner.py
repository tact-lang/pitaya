"""Top-level runner that wires config, orchestrator creation, and mode dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
        except Exception:
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


def _build_orchestrator(cfg: Dict[str, Any], auth: AuthConfig) -> Orchestrator:
    total, start = _parallel(cfg)
    orch = cfg.get("orchestration", {})
    runr = cfg.get("runner", {})
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
        default_agent_cli_args=None,
        force_commit=bool(runr.get("force_commit", False)),
        randomize_queue_order=bool(orch.get("randomize_queue_order", False)),
    )


async def run(console: Console, args) -> int:
    run_id = _make_run_id(getattr(args, "resume", None))
    if not getattr(args, "resume", None):
        if not perform_preflight_checks(console, args):
            return 1

    full_config = _merge_full_config(args)
    if not validate_full_config(console, full_config, args):
        return 1

    auth_cfg = get_auth_config(args, full_config)
    orch = _build_orchestrator(full_config, auth_cfg)
    await orch.initialize()

    try:
        if getattr(args, "no_tui", False):
            return await headless_run.run_headless(
                console, orch, args, full_config, run_id
            )
        return await tui_runner.run_tui(console, orch, args, full_config, run_id)
    except (OrchestratorError, DockerError, ValidationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print_exception()
        return 1
    finally:
        try:
            await orch.shutdown()
        except Exception:
            pass
