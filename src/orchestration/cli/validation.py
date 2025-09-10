"""Validation utilities for CLI configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...orchestration.strategies import AVAILABLE_STRATEGIES
from rich.console import Console

__all__ = ["validate_full_config"]


def _add(
    errors: list[tuple[str, str, str]], field: str, reason: str, example: str = ""
) -> None:
    errors.append((field, reason, example))


def _validate_runner(cfg: Dict[str, Any], errors: list[tuple[str, str, str]]) -> None:
    rt = cfg.get("runner", {}).get("timeout")
    if not isinstance(rt, (int, float)) or int(rt) <= 0:
        _add(errors, "runner.timeout", "must be a positive integer", "3600")
    try:
        cpu = int(cfg.get("runner", {}).get("cpu_limit", 2))
        if cpu <= 0:
            _add(errors, "runner.cpu_limit", "must be > 0", "2")
    except (TypeError, ValueError):
        _add(errors, "runner.cpu_limit", "must be an integer", "2")
    try:
        mem = cfg.get("runner", {}).get("memory_limit", "4g")
        val = int(mem[:-1]) if isinstance(mem, str) and mem.lower().endswith("g") else int(mem)  # type: ignore[arg-type]
        if val <= 0:
            _add(errors, "runner.memory_limit", "must be > 0", "4g")
    except (TypeError, ValueError):
        _add(errors, "runner.memory_limit", "must be number or '<n>g'", "4g")
    egress = str(cfg.get("runner", {}).get("network_egress", "online")).lower()
    if egress not in {"online", "offline", "proxy"}:
        _add(errors, "runner.network_egress", "must be online|offline|proxy", "online")


def _validate_orch(cfg: Dict[str, Any], errors: list[tuple[str, str, str]]) -> None:
    mpi = cfg.get("orchestration", {}).get("max_parallel_instances", "auto")
    if not (isinstance(mpi, str) and mpi.lower() == "auto"):
        try:
            v = int(mpi)  # type: ignore[arg-type]
            if v <= 0:
                _add(
                    errors,
                    "orchestration.max_parallel_instances",
                    "must be > 0 or 'auto'",
                    "auto",
                )
        except (TypeError, ValueError):
            _add(
                errors,
                "orchestration.max_parallel_instances",
                "must be integer or 'auto'",
                "auto",
            )
    mps = cfg.get("orchestration", {}).get("max_parallel_startup", "auto")
    if not (isinstance(mps, str) and mps.lower() == "auto"):
        try:
            v2 = int(mps)  # type: ignore[arg-type]
            if v2 <= 0:
                _add(
                    errors,
                    "orchestration.max_parallel_startup",
                    "must be > 0 or 'auto'",
                    "auto",
                )
        except (TypeError, ValueError):
            _add(
                errors,
                "orchestration.max_parallel_startup",
                "must be integer or 'auto'",
                "auto",
            )


def _validate_strategy(
    cfg: Dict[str, Any], requested: str, errors: list[tuple[str, str, str]]
) -> None:
    strategy = cfg.get("strategy", requested)
    ok = False
    if isinstance(strategy, str):
        if strategy in AVAILABLE_STRATEGIES:
            ok = True
        else:
            spec = str(strategy)
            spec_path = spec.split(":", 1)[0] if ":" in spec else spec
            if spec_path.endswith(".py") and Path(spec_path).exists():
                ok = True
            else:
                import re as _re

                mod_re = _re.compile(
                    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
                )
                if mod_re.match(spec_path):
                    ok = True
    if not ok:
        _add(
            errors,
            "strategy",
            "unknown strategy (use built-in, file.py[:Class], or module.path[:Class])",
            ",".join(AVAILABLE_STRATEGIES.keys()),
        )


def validate_full_config(console: Console, full_config: Dict[str, Any], args) -> bool:
    errors: list[tuple[str, str, str]] = []
    _validate_runner(full_config, errors)
    _validate_orch(full_config, errors)
    _validate_strategy(full_config, getattr(args, "strategy", ""), errors)
    if errors:
        console.print("[red]Invalid configuration:[/red]")
        console.print("field | reason | example")
        for f, r, ex in errors:
            console.print(f"{f} | {r} | {ex}")
        return False
    return True
