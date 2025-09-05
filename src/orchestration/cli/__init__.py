"""CLI subpackage for orchestrator-facing commands and helpers.

Each module is kept small and focused to comply with the handbook budgets.
Public surface is explicit via __all__.
"""

from __future__ import annotations

__all__ = [
    "doctor",
    "config_print",
    "runs",
    "results_display",
    "config_loader",
    "auth",
    "strategy_config",
    "validation",
    "preflight",
    "orchestrator_runner",
    "headless",
    "tui_runner",
]
