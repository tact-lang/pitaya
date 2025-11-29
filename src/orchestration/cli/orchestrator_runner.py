"""Top-level runner that wires config, orchestrator creation, and mode dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console

from ...exceptions import DockerError, OrchestratorError, ValidationError
from ...orchestration import Orchestrator
from ...shared import AuthConfig, ContainerLimits, RetryConfig
from . import headless as headless_run
from . import tui_runner
from .auth import get_auth_config
from .preflight import perform_preflight_checks
from .runner_setup import (
    apply_resume_overrides,
    load_effective_config,
    persist_config_snapshots,
    propagate_resume_paths,
    setup_logging,
)
from .validation import validate_full_config

__all__ = ["run"]


def _make_run_id(resume: Optional[str]) -> str:
    if resume:
        return resume
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    import uuid as _uuid

    return f"run_{ts}_{_uuid.uuid4().hex[:8]}"


def _apply_headless_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        args.no_tui = True
        args.output = "json"
    try:
        if not sys.stdout.isatty() and not getattr(args, "json", False):
            args.no_tui = True
    except (AttributeError, OSError):
        pass


def _handle_interrupt(console: Console, args: argparse.Namespace, run_id: str) -> int:
    console.print("\n[yellow]Interrupted — shutting down gracefully[/yellow]")
    try:
        is_json = bool(
            getattr(args, "json", False) or getattr(args, "output", "") == "json"
        )
        if not is_json and run_id:
            console.print(f"[blue]Resume:[/blue] pitaya --resume {run_id}")
    except Exception:
        pass
    return 2


def _validate_docker(
    console: Console, args: argparse.Namespace, run_id: str
) -> Optional[int]:
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
    except (KeyboardInterrupt, asyncio.CancelledError):
        return _handle_interrupt(console, args, run_id)
    except ImportError:
        pass
    return None


def _preflight(
    console: Console, args: argparse.Namespace, run_id: str
) -> Optional[int]:
    if getattr(args, "resume", None):
        return None
    try:
        if not perform_preflight_checks(console, args):
            return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        return _handle_interrupt(console, args, run_id)
    return None


def _container_limits(cfg: Dict[str, Any]) -> ContainerLimits:
    runner_cfg = cfg.get("runner", {})
    mem = runner_cfg.get("memory_limit", 4)
    mem_gb = (
        int(mem[:-1])
        if isinstance(mem, str) and str(mem).lower().endswith("g")
        else int(mem)
    )
    return ContainerLimits(
        cpu_count=int(runner_cfg.get("cpu_limit", 2)),
        memory_gb=mem_gb,
        memory_swap_gb=mem_gb,
    )


def _parallel(cfg: Dict[str, Any]) -> Tuple[int, int]:
    import os as _os

    def _cpu_default() -> int:
        try:
            return max(2, int(_os.cpu_count() or 2))
        except (TypeError, ValueError):
            return 2

    orch_cfg = cfg.get("orchestration", {})
    total = orch_cfg.get("max_parallel_instances", "auto")
    auto_total = isinstance(total, str) and total.lower() == "auto"
    total_val = _cpu_default() if auto_total else max(1, int(total))
    start = orch_cfg.get("max_parallel_startup", "auto")
    auto_start = isinstance(start, str) and start.lower() == "auto"
    start_val = min(total_val, 10) if auto_start else max(1, int(start))
    return total_val, min(start_val, total_val)


def _collect_agent_args(args: argparse.Namespace | None) -> list[str] | None:
    if args is None:
        return None
    try:
        import shlex as _shlex

        collected: list[str] = []
        if getattr(args, "agent_cli_arg", None):
            collected.extend([str(a) for a in args.agent_cli_arg if a is not None])
        if getattr(args, "agent_cli_args_str", None):
            collected.extend(_shlex.split(str(args.agent_cli_args_str)))
        return collected or None
    except (AttributeError, ValueError, ImportError):
        return None


def _build_orchestrator(
    cfg: Dict[str, Any], auth: AuthConfig, args=None
) -> Orchestrator:
    total, start = _parallel(cfg)
    orch_cfg = cfg.get("orchestration", {})
    runner_cfg = cfg.get("runner", {})

    return Orchestrator(
        max_parallel_instances=total,
        max_parallel_startup=start,
        state_dir=Path(orch_cfg.get("state_dir", Path("./pitaya_state"))),
        logs_dir=Path(orch_cfg.get("logs_dir", Path("./logs"))),
        container_limits=_container_limits(cfg),
        retry_config=RetryConfig(max_attempts=3),
        auth_config=auth,
        snapshot_interval=int(orch_cfg.get("snapshot_interval", 30)),
        event_buffer_size=int(orch_cfg.get("event_buffer_size", 10000)),
        runner_timeout_seconds=int(runner_cfg.get("timeout", 3600)),
        default_network_egress=str(runner_cfg.get("network_egress", "online")),
        branch_namespace=str(orch_cfg.get("branch_namespace", "hierarchical")),
        allow_overwrite_protected_refs=bool(
            orch_cfg.get("allow_overwrite_protected_refs", False)
        ),
        default_plugin_name=str(cfg.get("plugin_name", "claude-code")),
        default_model_alias=str(cfg.get("model", "sonnet")),
        default_docker_image=runner_cfg.get("docker_image"),
        default_agent_cli_args=_collect_agent_args(args),
        force_commit=bool(runner_cfg.get("force_commit", False)),
        randomize_queue_order=bool(orch_cfg.get("randomize_queue_order", False)),
        default_workspace_include_branches=(
            list(runner_cfg.get("include_branches"))
            if isinstance(runner_cfg.get("include_branches"), list)
            else None
        ),
    )


def _apply_resume_suffix(orch: Orchestrator, full_config: Dict[str, Any]) -> None:
    try:
        suffix = full_config.get("orchestration", {}).get("resume_key_suffix")
        if suffix:
            setattr(orch, "resume_key_suffix", str(suffix))
    except Exception:
        pass


def _apply_redaction_patterns(orch: Orchestrator, full_config: Dict[str, Any]) -> None:
    try:
        redaction = (full_config.get("logging", {}) or {}).get("redaction", {}) or {}
        patterns = redaction.get("custom_patterns") or []
        if patterns and hasattr(orch, "set_pending_redaction_patterns"):
            orch.set_pending_redaction_patterns([str(p) for p in patterns])
    except Exception:
        pass


async def _dispatch(
    console: Console,
    orch: Orchestrator,
    args: argparse.Namespace,
    full_config: Dict[str, Any],
    run_id: str,
) -> int:
    if getattr(args, "no_tui", False):
        return await headless_run.run_headless(console, orch, args, full_config, run_id)
    return await tui_runner.run_tui(console, orch, args, full_config, run_id)


async def _run_once(
    console: Console, args: argparse.Namespace, run_id: str
) -> Tuple[int, Orchestrator | None]:
    _apply_headless_defaults(args)

    rc = _validate_docker(console, args, run_id)
    if rc is not None:
        return rc, None

    rc = _preflight(console, args, run_id)
    if rc is not None:
        return rc, None

    full_config = load_effective_config(args, run_id)
    setup_logging(full_config, args, run_id)

    if not validate_full_config(console, full_config, args):
        return 1, None

    if not getattr(args, "resume", None):
        persist_config_snapshots(full_config, args, run_id)
    else:
        full_config = apply_resume_overrides(full_config, console, args)
        if not validate_full_config(console, full_config, args):
            return 1, None
        propagate_resume_paths(args, full_config)

    auth_cfg = get_auth_config(args, full_config)
    orch = _build_orchestrator(full_config, auth_cfg, args)
    _apply_resume_suffix(orch, full_config)
    _apply_redaction_patterns(orch, full_config)

    try:
        await orch.initialize()
        rc = await _dispatch(console, orch, args, full_config, run_id)
        return rc, orch
    except BaseException:
        # Ensure we tear down even if initialize or dispatch fail so containers are cleaned up.
        try:
            await orch.shutdown()
        except Exception:
            pass
        raise


async def run(console: Console, args: argparse.Namespace) -> int:
    run_id = _make_run_id(getattr(args, "resume", None))
    orch: Orchestrator | None = None
    try:
        rc, orch = await _run_once(console, args, run_id)
        return rc
    except (KeyboardInterrupt, asyncio.CancelledError):
        return _handle_interrupt(console, args, run_id)
    except ValueError as exc:
        console.print(f"[red]Invalid arguments: {exc}[/red]")
        return 2
    except (OrchestratorError, DockerError, ValidationError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        console.print_exception()
        return 1
    finally:
        if orch is not None:
            try:
                await orch.shutdown()
            except (OrchestratorError, RuntimeError, OSError):
                pass
