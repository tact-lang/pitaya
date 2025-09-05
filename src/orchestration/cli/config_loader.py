"""Project config loading and CLI config assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["load_project_config", "build_cli_config"]


def load_project_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML project configuration, returning an empty dict on absence."""
    path = config_path
    if not path:
        default_path = Path("pitaya.yaml")
        path = default_path if default_path.exists() else None
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


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
