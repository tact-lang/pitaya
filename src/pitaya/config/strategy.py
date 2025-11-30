"""Strategy configuration assembly from args and config file."""

from __future__ import annotations

import argparse
from typing import Any, Dict


__all__ = ["get_strategy_config"]


def _parse_value(value: str) -> Any:
    """Best-effort typed parsing for -S KEY=VALUE.

    Order:
    - int/float/bool
    - JSON list/object when VALUE looks like JSON (starts with [ or {)
    - raw string otherwise
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            lowered = value.lower().strip()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
            # Lightweight JSON hint: list or object literals
            s = value.strip()
            if (s.startswith("[") and s.endswith("]")) or (
                s.startswith("{") and s.endswith("}")
            ):
                try:
                    import json as _json

                    return _json.loads(s)
                except Exception:
                    pass
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
        parsed = _parse_value(value)
        # Convenience: allow CSV or JSON list for include_branches via -S
        if key == "include_branches":
            import json as _json

            if isinstance(parsed, str):
                s = parsed.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = _json.loads(s)
                    except Exception:
                        parsed = [
                            p.strip() for p in s.strip("[]").split(",") if p.strip()
                        ]
                else:
                    parsed = [p.strip() for p in s.split(",") if p.strip()]
            cfg[key] = parsed
        else:
            cfg[key] = parsed
    return cfg
