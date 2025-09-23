"""Project config loading and CLI config assembly.

Behavior: if the user explicitly passes `--config <path>` and the file is
missing or unreadable, raise `ValueError` so the CLI can fail fast with a
clear message. If no path is provided and the default `pitaya.yaml` is not
present, return an empty dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["load_project_config", "build_cli_config"]


def load_project_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML project configuration.

    Rules:
    - If `config_path` is provided and does not exist or is unreadable ⇒ raise ValueError.
    - If `config_path` is None and `pitaya.yaml` does not exist ⇒ return {}.
    - On YAML parse errors for an explicit file ⇒ raise ValueError.
    """
    explicit = config_path is not None
    path = config_path
    if path is None:
        default_path = Path("pitaya.yaml")
        path = default_path if default_path.exists() else None
    if path is None:
        return {}
    if not path.exists():
        if explicit:
            raise ValueError(f"config file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            if explicit:
                raise ValueError(f"invalid config format (expected mapping): {path}")
            return {}
        return data
    except (OSError, yaml.YAMLError) as e:
        if explicit:
            raise ValueError(f"failed to read config {path}: {e}")
        return {}


def build_cli_config(args) -> Dict[str, Any]:
    """Translate argparse args into a hierarchical config dict."""
    cfg: Dict[str, Any] = {}
    if getattr(args, "max_parallel", None):
        cfg.setdefault("orchestration", {})[
            "max_parallel_instances"
        ] = args.max_parallel
    if getattr(args, "max_startup_parallel", None):
        cfg.setdefault("orchestration", {})[
            "max_parallel_startup"
        ] = args.max_startup_parallel
    if getattr(args, "randomize_queue", False):
        cfg.setdefault("orchestration", {})["randomize_queue_order"] = True
    if getattr(args, "timeout", None):
        cfg.setdefault("runner", {})["timeout"] = args.timeout
    if getattr(args, "force_commit", False):
        cfg.setdefault("runner", {})["force_commit"] = True
    if getattr(args, "model", None):
        cfg["model"] = args.model
    if getattr(args, "plugin", None):
        cfg["plugin_name"] = args.plugin
    if getattr(args, "docker_image", None):
        cfg.setdefault("runner", {})["docker_image"] = args.docker_image
    # Parse include-branches (CSV or JSON list)
    if getattr(args, "include_branches", None):
        import json as _json

        raw = str(args.include_branches)
        parsed: list[str]
        try:
            if raw.strip().startswith("["):
                val = _json.loads(raw)
                parsed = [str(x).strip() for x in (val or []) if str(x).strip()]
            else:
                parsed = [s.strip() for s in raw.split(",") if s.strip()]
        except Exception:
            parsed = [s.strip() for s in raw.split(",") if s.strip()]
        if parsed:
            cfg.setdefault("runner", {})["include_branches"] = parsed
    if getattr(args, "strategy", None):
        cfg["strategy"] = args.strategy
    if getattr(args, "state_dir", None):
        cfg.setdefault("orchestration", {})["state_dir"] = args.state_dir
    if getattr(args, "logs_dir", None):
        cfg.setdefault("orchestration", {})["logs_dir"] = args.logs_dir
    if getattr(args, "output", None):
        cfg["output"] = args.output
    if getattr(args, "oauth_token", None):
        cfg.setdefault("runner", {})["oauth_token"] = args.oauth_token
    if getattr(args, "api_key", None):
        cfg.setdefault("runner", {})["api_key"] = args.api_key
    if getattr(args, "base_url", None):
        cfg.setdefault("runner", {})["base_url"] = args.base_url
    return cfg
