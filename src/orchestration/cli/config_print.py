"""Effective configuration printing for Pitaya CLI.

Merges sources and prints values with redaction and provenance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
import argparse

from rich.console import Console

from ...config import (
    get_default_config,
    load_dotenv_config,
    load_env_config,
    load_global_config,
    merge_config,
)

__all__ = ["run_config_print"]


def _flat(d: Dict[str, Any] | None, p: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        kk = f"{p}.{k}" if p else str(k)
        if isinstance(v, dict):
            out.update(_flat(v, kk))
        else:
            out[kk] = v
    return out


def _redact(allow_unred: bool, k: str, v: Any) -> Any:
    if allow_unred:
        return v
    kl = k.lower()
    if any(
        s in kl
        for s in ("token", "key", "secret", "password", "authorization", "cookie")
    ):
        return "[REDACTED]"
    return v


def _load_project_config(path: Path | None) -> Dict[str, Any]:
    if not path:
        default = Path("pitaya.yaml")
        if not default.exists():
            return {}
        path = default
    if not path.exists():
        # Explicit path not found: fail fast
        raise ValueError(f"config file not found: {path}")
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"invalid config format (expected mapping): {path}")
        return data
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"failed to read config {path}: {e}")


async def run_config_print(console: Console, args: argparse.Namespace) -> int:
    try:
        env = load_env_config()
        dotenv = load_dotenv_config()
        defaults = get_default_config()
        global_cfg = load_global_config()
        project_cfg = _load_project_config(getattr(args, "config", None))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    merged = merge_config(
        {},
        env,
        dotenv,
        project_cfg,
        merge_config({}, {}, {}, global_cfg or {}, defaults),
    )
    allow_unred = (
        os.environ.get("PITAYA_ALLOW_UNREDACTED") == "1"
        and str(getattr(args, "redact", "true")).lower() == "false"
    )

    sources = {
        "env": _flat(env),
        "dotenv": _flat(dotenv),
        "project": _flat(project_cfg),
        "global": _flat(global_cfg or {}),
        "defaults": _flat(defaults),
    }
    flat = _flat(merged)

    def _src_for(k: str) -> str:
        for name in ("env", "dotenv", "project", "global", "defaults"):
            if k in sources[name]:
                return name
        return "defaults"

    if getattr(args, "json", False):
        out = {
            k: {"value": _redact(allow_unred, k, v), "source": _src_for(k)}
            for k, v in flat.items()
        }
        print(json.dumps(out, indent=2, default=str))
        return 0
    for k in sorted(flat.keys()):
        print(f"{k}: {_redact(allow_unred, k, flat[k])}  ({_src_for(k)})")
    return 0
