"""Strategy configuration assembly from args and config file."""

from __future__ import annotations

import argparse
from typing import Any, Dict


__all__ = ["get_strategy_config"]


def _parse_value(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            lowered = value.lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
            return value


def get_strategy_config(
    args: argparse.Namespace, full_config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Build a dict to configure the selected strategy.

    Raises:
        ValueError: if a strategy parameter is not formatted as KEY=VALUE.
    """
    cfg: Dict[str, Any] = {}
    strategy_name = getattr(args, "strategy", None) or (full_config or {}).get(
        "strategy", "simple"
    )
    strategies = (full_config or {}).get("strategies", {})
    # aliases hyphen/underscore
    for name in (
        strategy_name,
        strategy_name.replace("-", "_"),
        strategy_name.replace("_", "-"),
    ):
        if name in strategies:
            cfg.update(strategies[name])
            break
    cfg["model"] = getattr(args, "model", None) or (full_config or {}).get(
        "model", "sonnet"
    )
    for param in getattr(args, "strategy_params", []) or []:
        if "=" not in param:
            raise ValueError(f"invalid strategy param: {param!r}; expected KEY=VALUE")
        key, value = param.split("=", 1)
        cfg[key] = _parse_value(value)
    return cfg
