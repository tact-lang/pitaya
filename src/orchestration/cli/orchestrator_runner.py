"""Top-level runner that wires config, orchestrator creation, and mode dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
import asyncio
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
        default_workspace_include_branches=(
            list(runr.get("include_branches"))
            if isinstance(runr.get("include_branches"), list)
            else None
        ),
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
    except (KeyboardInterrupt, asyncio.CancelledError):
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
    except ImportError:
        # If platform utils not available, continue; orchestrator will surface errors
        pass
    if not getattr(args, "resume", None):
        try:
            if not perform_preflight_checks(console, args):
                return 1
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Interrupted — shutting down gracefully[/yellow]")
            try:
                is_json = bool(
                    getattr(args, "json", False)
                    or getattr(args, "output", "") == "json"
                )
                if not is_json and run_id:
                    console.print(f"[blue]Resume:[/blue] pitaya --resume {run_id}")
            except Exception:
                pass
            return 2
    orch: Orchestrator | None = None

    try:
        # Merge full configuration (may raise ValueError for bad --config)
        # On resume, prefer saved config from state_dir/<run_id>/config.json
        full_config = None
        if getattr(args, "resume", None):
            try:
                cfg_path = (
                    Path(getattr(args, "state_dir", Path("./pitaya_state")))
                    / run_id
                    / "config.json"
                )
                import json as _json

                with open(cfg_path, "r", encoding="utf-8") as f:
                    full_config = _json.load(f)
            except Exception:
                full_config = None
        if not full_config:
            full_config = _merge_full_config(args)

        # Structured logging and rotation setup (restored from pre-refactor behavior)
        try:
            from ...utils.structured_logging import setup_structured_logging
            from ...utils.log_rotation import cleanup_old_logs, setup_log_rotation_task

            logs_dir_cfg = full_config.get("orchestration", {}).get(
                "logs_dir"
            ) or getattr(args, "logs_dir", Path("./logs"))
            logs_dir = Path(logs_dir_cfg)
            quiet = bool(
                getattr(args, "no_tui", False)
                and getattr(args, "output", "") == "quiet"
            )
            setup_structured_logging(
                logs_dir=logs_dir,
                run_id=run_id,
                debug=True,
                quiet=quiet,
                no_tui=bool(getattr(args, "no_tui", False)),
            )
            try:
                cleanup_old_logs(logs_dir)
                # Determine max size from config if present (bytes → MB)
                max_bytes = (full_config.get("logging", {}) or {}).get(
                    "max_file_size", 10485760
                )
                try:
                    max_mb = (
                        int(max_bytes / (1024 * 1024))
                        if isinstance(max_bytes, (int, float))
                        else 100
                    )
                except (TypeError, ValueError):
                    max_mb = 100
                import asyncio as _asyncio

                _asyncio.create_task(
                    setup_log_rotation_task(logs_dir, max_size_mb=max_mb)
                )
            except (OSError, PermissionError):
                # Non-fatal; continue without rotation
                pass
        except ImportError:
            # Logging helpers unavailable; skip structured logging setup
            pass

        if not validate_full_config(console, full_config, args):
            return 1

        # Persist effective config for fresh runs to support resume fidelity
        if not getattr(args, "resume", None):
            try:
                import json as _json

                sdir = Path(
                    full_config.get("orchestration", {}).get(
                        "state_dir", getattr(args, "state_dir", Path("./pitaya_state"))
                    )
                )
                ldir = Path(
                    full_config.get("orchestration", {}).get(
                        "logs_dir", getattr(args, "logs_dir", Path("./logs"))
                    )
                )
                (sdir / run_id).mkdir(parents=True, exist_ok=True)
                (ldir / run_id).mkdir(parents=True, exist_ok=True)

                def _redact_cfg(d: Dict[str, Any]) -> Dict[str, Any]:
                    import copy as _copy

                    def scrub(obj):
                        if isinstance(obj, dict):
                            out = {}
                            for k, v in obj.items():
                                kl = str(k).lower()
                                if any(
                                    s in kl
                                    for s in (
                                        "token",
                                        "api_key",
                                        "apikey",
                                        "secret",
                                        "password",
                                        "authorization",
                                        "cookie",
                                    )
                                ):
                                    out[k] = "[REDACTED]"
                                else:
                                    out[k] = scrub(v)
                            return out
                        if isinstance(obj, list):
                            return [scrub(v) for v in obj]
                        return obj

                    return scrub(_copy.deepcopy(d))

                with open(sdir / run_id / "config.json", "w", encoding="utf-8") as f:
                    _json.dump(full_config, f, indent=2, default=str)
                with open(ldir / run_id / "config.json", "w", encoding="utf-8") as f:
                    _json.dump(_redact_cfg(full_config), f, indent=2, default=str)
            except Exception:
                pass

        # Apply resume overrides: safe by default; unsafe require --override-config
        if getattr(args, "resume", None):
            # Only apply CLI flags that the user explicitly provided, not parser defaults.
            # This prevents silently overwriting recovered config values (e.g., timeout)
            # with default CLI values during resume.
            def _explicit_cli_overrides(a: argparse.Namespace) -> Dict[str, Any]:
                try:
                    from ..cli_parser import create_parser  # type: ignore
                except Exception:
                    # Fallback: treat all as explicit (conservative)
                    return build_cli_config(a)
                parser = create_parser()
                defaults = parser.parse_args([])

                cfg: Dict[str, Any] = {}

                def changed(name: str) -> bool:
                    return getattr(a, name, None) != getattr(defaults, name, None)

                if (
                    changed("max_parallel")
                    and getattr(a, "max_parallel", None) is not None
                ):
                    cfg.setdefault("orchestration", {})[
                        "max_parallel_instances"
                    ] = a.max_parallel
                if (
                    changed("max_startup_parallel")
                    and getattr(a, "max_startup_parallel", None) is not None
                ):
                    cfg.setdefault("orchestration", {})[
                        "max_parallel_startup"
                    ] = a.max_startup_parallel
                if changed("randomize_queue") and getattr(a, "randomize_queue", False):
                    cfg.setdefault("orchestration", {})["randomize_queue_order"] = True
                if changed("timeout") and getattr(a, "timeout", None) is not None:
                    cfg.setdefault("runner", {})["timeout"] = a.timeout
                if changed("force_commit") and getattr(a, "force_commit", False):
                    cfg.setdefault("runner", {})["force_commit"] = True
                if changed("docker_image") and getattr(a, "docker_image", None):
                    cfg.setdefault("runner", {})["docker_image"] = a.docker_image
                if changed("model") and getattr(a, "model", None):
                    cfg["model"] = a.model
                if changed("plugin") and getattr(a, "plugin", None):
                    cfg["plugin_name"] = a.plugin
                # Auth are explicit if present (defaults are None)
                if getattr(a, "oauth_token", None) is not None:
                    cfg.setdefault("runner", {})["oauth_token"] = a.oauth_token
                if getattr(a, "api_key", None) is not None:
                    cfg.setdefault("runner", {})["api_key"] = a.api_key
                if getattr(a, "base_url", None) is not None:
                    cfg.setdefault("runner", {})["base_url"] = a.base_url
                return cfg

            overrides = _explicit_cli_overrides(args)
            safe_paths = {
                ("runner", "timeout"),
                ("runner", "docker_image"),
                ("runner", "force_commit"),
                ("orchestration", "max_parallel_instances"),
                ("orchestration", "max_parallel_startup"),
                ("orchestration", "randomize_queue_order"),
                # Auth overrides
                ("runner", "oauth_token"),
                ("runner", "api_key"),
                ("runner", "base_url"),
            }
            unsafe_top = {"model", "plugin_name"}
            unsafe_runner = {"network_egress"}
            for sect, key in safe_paths:
                val = (overrides.get(sect, {}) or {}).get(key)
                if val is not None:
                    full_config.setdefault(sect, {})[key] = val
            unsafe_requested: list[str] = []
            for k in unsafe_top:
                if (
                    k in overrides
                    and overrides[k] is not None
                    and overrides[k] != full_config.get(k)
                ):
                    unsafe_requested.append(k)
            for k in unsafe_runner:
                if (
                    k in (overrides.get("runner", {}) or {})
                    and overrides["runner"][k] is not None
                    and overrides["runner"][k]
                    != (full_config.get("runner", {}) or {}).get(k)
                ):
                    unsafe_requested.append(f"runner.{k}")
            if unsafe_requested and not getattr(args, "override_config", False):
                console.print(
                    "[yellow]Ignoring overrides on resume for: "
                    + ", ".join(unsafe_requested)
                    + ". Use --override-config to force (may change durable keys).[/yellow]"
                )
            elif unsafe_requested and getattr(args, "override_config", False):
                if "model" in overrides:
                    full_config["model"] = overrides["model"]
                if "plugin_name" in overrides:
                    full_config["plugin_name"] = overrides["plugin_name"]
                if "network_egress" in (overrides.get("runner", {}) or {}):
                    full_config.setdefault("runner", {})["network_egress"] = overrides[
                        "runner"
                    ]["network_egress"]
                if getattr(args, "resume_key_policy", "strict") == "suffix":
                    full_config.setdefault("orchestration", {})["resume_key_suffix"] = (
                        full_config.get("orchestration", {}).get("resume_key_suffix")
                        or ("r" + run_id[-4:])
                    )

            # Important: re-validate after applying overrides so malformed values are caught early
            if not validate_full_config(console, full_config, args):
                return 1

        # If resuming, propagate recovered directories from config back to args so
        # downstream components (e.g., TUI, diagnostics) read the correct paths.
        try:
            if getattr(args, "resume", None):
                orch_cfg = full_config.get("orchestration", {}) or {}
                ld = orch_cfg.get("logs_dir")
                sd = orch_cfg.get("state_dir")
                if ld:
                    args.logs_dir = Path(ld)
                if sd:
                    args.state_dir = Path(sd)
        except Exception:
            pass

        auth_cfg = get_auth_config(args, full_config)
        orch = _build_orchestrator(full_config, auth_cfg, args)
        try:
            suffix = full_config.get("orchestration", {}).get("resume_key_suffix")
            if suffix:
                setattr(orch, "resume_key_suffix", str(suffix))
        except Exception:
            pass
        # Apply custom redaction patterns from config to event bus (on creation during run)
        try:
            redaction = (full_config.get("logging", {}) or {}).get(
                "redaction", {}
            ) or {}
            patterns = redaction.get("custom_patterns") or []
            if patterns and hasattr(orch, "set_pending_redaction_patterns"):
                orch.set_pending_redaction_patterns([str(p) for p in patterns])
        except Exception:
            # Non-fatal: continue without custom patterns
            pass
        await orch.initialize()

        if getattr(args, "no_tui", False):
            return await headless_run.run_headless(
                console, orch, args, full_config, run_id
            )
        rc = await tui_runner.run_tui(console, orch, args, full_config, run_id)
        # TUI prints its own resume hint on interrupt; avoid duplicate here
        return rc
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Interrupted — shutting down gracefully[/yellow]")
        try:
            # Avoid polluting JSON mode; only print hint for streaming/TUI
            is_json = bool(
                getattr(args, "json", False) or getattr(args, "output", "") == "json"
            )
            if not is_json and run_id:
                console.print(f"[blue]Resume:[/blue] pitaya --resume {run_id}")
        except Exception:
            pass
        return 2
    except ValueError as e:
        console.print(f"[red]Invalid arguments: {e}[/red]")
        return 2
    except (OrchestratorError, DockerError, ValidationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print_exception()
        return 1
    finally:
        if orch is not None:
            try:
                await orch.shutdown()
            except (OrchestratorError, RuntimeError, OSError):
                pass
