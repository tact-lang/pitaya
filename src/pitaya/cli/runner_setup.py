"""Helpers for assembling config, logging, and resume handling for the runner."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

from pitaya.config import (
    get_default_config,
    load_dotenv_config,
    load_env_config,
    load_global_config,
    merge_config,
)
from pitaya.config.loader import build_cli_config, load_project_config

__all__ = [
    "apply_resume_overrides",
    "load_effective_config",
    "merge_full_config",
    "persist_config_snapshots",
    "propagate_resume_paths",
    "setup_logging",
]


SAFE_RESUME_PATHS = {
    ("runner", "timeout"),
    ("runner", "docker_image"),
    ("runner", "force_commit"),
    ("orchestration", "max_parallel_instances"),
    ("orchestration", "max_parallel_startup"),
    ("orchestration", "randomize_queue_order"),
    ("runner", "oauth_token"),
    ("runner", "api_key"),
    ("runner", "base_url"),
}
UNSAFE_TOP_LEVEL = {"model", "plugin_name"}
UNSAFE_RUNNER_KEYS = {"network_egress"}


def merge_full_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config from CLI, env, dotenv, project, global, and defaults."""

    cli = build_cli_config(args)
    env = load_env_config()
    dotenv = load_dotenv_config()
    project = load_project_config(getattr(args, "config", None))
    defaults = get_default_config()
    global_cfg = load_global_config()
    return merge_config(
        cli,
        env,
        dotenv,
        project,
        merge_config({}, {}, {}, global_cfg or {}, defaults),
    )


def load_effective_config(args: argparse.Namespace, run_id: str) -> Dict[str, Any]:
    """Load config for this run, preferring resume snapshot when available."""

    if getattr(args, "resume", None):
        try:
            cfg_path = Path(getattr(args, "state_dir", Path(".pitaya/state")))
            cfg_file = cfg_path / run_id / "config.json"
            with open(cfg_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            # Fall back to live merge if snapshot missing or corrupt
            pass
    return merge_full_config(args)


def setup_logging(
    full_config: Dict[str, Any], args: argparse.Namespace, run_id: str
) -> None:
    """Configure structured logging and background log rotation if available."""

    try:
        from pitaya.utils.log_rotation import cleanup_old_logs, setup_log_rotation_task
        from pitaya.utils.structured_logging import setup_structured_logging
    except ImportError:
        return

    logs_dir_cfg = full_config.get("orchestration", {}).get("logs_dir") or getattr(
        args, "logs_dir", Path(".pitaya/logs")
    )
    logs_dir = Path(logs_dir_cfg)
    logging_cfg = full_config.get("logging", {}) or {}

    if not getattr(args, "verbose", False) and logging_cfg.get(
        "console_verbose", False
    ):
        args.verbose = True

    quiet = bool(
        getattr(args, "no_tui", False) and getattr(args, "output", "") == "quiet"
    )
    setup_structured_logging(
        logs_dir=logs_dir,
        run_id=run_id,
        quiet=quiet,
        no_tui=bool(getattr(args, "no_tui", False)),
        enable_console=False,
    )

    try:
        cleanup_old_logs(logs_dir)
        max_bytes = logging_cfg.get("max_file_size", 10485760)
        try:
            max_mb = (
                int(max_bytes / (1024 * 1024))
                if isinstance(max_bytes, (int, float))
                else 100
            )
        except (TypeError, ValueError):
            max_mb = 100
        import asyncio as _asyncio

        _asyncio.create_task(setup_log_rotation_task(logs_dir, max_size_mb=max_mb))
    except (OSError, PermissionError):
        # Non-fatal: logging remains functional without rotation
        pass


def _redact_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    def scrub(obj):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                kl = str(key).lower()
                if any(
                    token in kl
                    for token in (
                        "token",
                        "api_key",
                        "apikey",
                        "secret",
                        "password",
                        "authorization",
                        "cookie",
                    )
                ):
                    out[key] = "[REDACTED]"
                else:
                    out[key] = scrub(value)
            return out
        if isinstance(obj, list):
            return [scrub(item) for item in obj]
        return obj

    return scrub(deepcopy(raw))


def persist_config_snapshots(
    full_config: Dict[str, Any], args: argparse.Namespace, run_id: str
) -> None:
    """Persist full and redacted configs to state/logs to enable resume."""

    try:
        sdir = Path(
            full_config.get("orchestration", {}).get(
                "state_dir", getattr(args, "state_dir", Path(".pitaya/state"))
            )
        )
        ldir = Path(
            full_config.get("orchestration", {}).get(
                "logs_dir", getattr(args, "logs_dir", Path(".pitaya/logs"))
            )
        )
        (sdir / run_id).mkdir(parents=True, exist_ok=True)
        (ldir / run_id).mkdir(parents=True, exist_ok=True)

        with open(sdir / run_id / "config.json", "w", encoding="utf-8") as fh:
            json.dump(full_config, fh, indent=2, default=str)
        with open(ldir / run_id / "config.json", "w", encoding="utf-8") as fh:
            json.dump(_redact_cfg(full_config), fh, indent=2, default=str)
    except Exception:
        # Do not block execution if persistence fails
        pass


def _apply_safe_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for sect, key in SAFE_RESUME_PATHS:
        val = (overrides.get(sect, {}) or {}).get(key)
        if val is not None:
            cfg.setdefault(sect, {})[key] = val


def _collect_unsafe_requests(
    cfg: Dict[str, Any], overrides: Dict[str, Any]
) -> list[str]:
    unsafe_requested: list[str] = []
    for key in UNSAFE_TOP_LEVEL:
        if (
            key in overrides
            and overrides[key] is not None
            and overrides[key] != cfg.get(key)
        ):
            unsafe_requested.append(key)
    for key in UNSAFE_RUNNER_KEYS:
        runner_val = (overrides.get("runner", {}) or {}).get(key)
        if runner_val is not None and runner_val != (cfg.get("runner", {}) or {}).get(
            key
        ):
            unsafe_requested.append(f"runner.{key}")
    return unsafe_requested


def _apply_unsafe_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    if "model" in overrides and overrides.get("model") is not None:
        cfg["model"] = overrides["model"]
    if "plugin_name" in overrides and overrides.get("plugin_name") is not None:
        cfg["plugin_name"] = overrides["plugin_name"]
    if "network_egress" in (overrides.get("runner", {}) or {}):
        cfg.setdefault("runner", {})["network_egress"] = overrides["runner"][
            "network_egress"
        ]


def _explicit_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from ..cli_parser import create_parser
    except Exception:
        return build_cli_config(args)

    parser = create_parser()
    defaults = parser.parse_args([])

    cfg: Dict[str, Any] = {}

    def changed(name: str) -> bool:
        return getattr(args, name, None) != getattr(defaults, name, None)

    if changed("max_parallel") and getattr(args, "max_parallel", None) is not None:
        cfg.setdefault("orchestration", {})[
            "max_parallel_instances"
        ] = args.max_parallel
    if (
        changed("max_startup_parallel")
        and getattr(args, "max_startup_parallel", None) is not None
    ):
        cfg.setdefault("orchestration", {})[
            "max_parallel_startup"
        ] = args.max_startup_parallel
    if changed("randomize_queue") and getattr(args, "randomize_queue", False):
        cfg.setdefault("orchestration", {})["randomize_queue_order"] = True
    if changed("timeout") and getattr(args, "timeout", None) is not None:
        cfg.setdefault("runner", {})["timeout"] = args.timeout
    if changed("force_commit") and getattr(args, "force_commit", False):
        cfg.setdefault("runner", {})["force_commit"] = True
    if changed("docker_image") and getattr(args, "docker_image", None):
        cfg.setdefault("runner", {})["docker_image"] = args.docker_image
    if changed("model") and getattr(args, "model", None):
        cfg["model"] = args.model
    if changed("plugin") and getattr(args, "plugin", None):
        cfg["plugin_name"] = args.plugin

    # Auth overrides are explicit if present (parser defaults are None)
    if getattr(args, "oauth_token", None) is not None:
        cfg.setdefault("runner", {})["oauth_token"] = args.oauth_token
    if getattr(args, "api_key", None) is not None:
        cfg.setdefault("runner", {})["api_key"] = args.api_key
    if getattr(args, "base_url", None) is not None:
        cfg.setdefault("runner", {})["base_url"] = args.base_url
    return cfg


def apply_resume_overrides(
    full_config: Dict[str, Any], console: Console, args: argparse.Namespace
) -> Dict[str, Any]:
    """Apply explicit CLI overrides when resuming a run.

    Safe fields are applied automatically. Potentially unsafe overrides require
    ``--override-config``; otherwise they are ignored with a warning.
    """

    overrides = _explicit_cli_overrides(args)
    cfg = deepcopy(full_config)

    _apply_safe_overrides(cfg, overrides)
    unsafe_requested = _collect_unsafe_requests(cfg, overrides)

    if unsafe_requested and not getattr(args, "override_config", False):
        console.print(
            "[yellow]Ignoring overrides on resume for: "
            + ", ".join(unsafe_requested)
            + ". Use --override-config to force (may change durable keys).[/yellow]"
        )
        return cfg

    _apply_unsafe_overrides(cfg, overrides)

    if getattr(args, "resume_key_policy", "strict") == "suffix":
        cfg.setdefault("orchestration", {})["resume_key_suffix"] = cfg.get(
            "orchestration", {}
        ).get("resume_key_suffix") or ("r" + run_id_suffix(args))

    return cfg


def run_id_suffix(args: argparse.Namespace) -> str:
    rid = getattr(args, "resume", "") or ""
    return str(rid)[-4:] if rid else "temp"


def propagate_resume_paths(
    args: argparse.Namespace, full_config: Dict[str, Any]
) -> None:
    """Update args with state/log directories recovered from config for resume."""

    orch_cfg = full_config.get("orchestration", {}) or {}
    if orch_cfg.get("logs_dir"):
        args.logs_dir = Path(orch_cfg["logs_dir"])
    if orch_cfg.get("state_dir"):
        args.state_dir = Path(orch_cfg["state_dir"])
