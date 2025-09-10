"""CLI parser builder for the Pitaya orchestrator.

Lean `create_parser()` that assembles argument groups via small helpers.
This keeps functions under the handbook budget while preserving UX.
"""

from __future__ import annotations

import argparse
from . import __version__
from .strategies import AVAILABLE_STRATEGIES
from .cli.parser_sections import (
    add_global_and_positional,
    add_strategy_args,
    add_model_plugin_args,
    add_repo_args,
    add_display_args,
    add_auth_args,
    add_limits_args,
    add_state_args,
    add_maintenance_args,
    add_diag_args,
)

__all__ = ["create_parser"]


def _epilog() -> str:
    return (
        "Quick examples:\n"
        "  # One-shot\n"
        '  pitaya "implement auth" --strategy simple\n\n'
        "  # Best-of-N with scoring\n"
        '  pitaya "fix bug in user.py" --strategy best-of-n -S n=5 -S scorer_model=opus\n\n'
        "  # Iterative refine (3 rounds)\n"
        '  pitaya "refactor module" --strategy iterative -S iterations=3\n\n'
        "  # Bug finding (target area)\n"
        '  pitaya "find a bug" --strategy bug-finding -S target_area=src/parser\n\n'
        "  # Custom strategy (module or file)\n"
        '  pitaya "task" --strategy examples.fanout_two:FanOutTwoStrategy\n'
        '  pitaya "task" --strategy ./examples/propose_refine.py\n\n'
        "  # Headless modes\n"
        '  pitaya "task" --no-tui --output streaming --verbose\n'
        '  pitaya "task" --json  # emits NDJSON only (no human summary)\n\n'
        "  # Run management\n"
        "  pitaya --list-runs\n"
        "  pitaya --show-run run_20250114_123456\n"
        "  pitaya --resume run_20250114_123456\n\n"
        "Tips:\n"
        "  • Built-ins: " + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys()))) + "\n"
        "  • Strategies accept -S key=value (numbers/bools auto-parsed).\n"
        "  • --model picks the agent model; --plugin chooses the runner.\n"
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitaya",
        description=(
            "Orchestrate AI coding agents with pluggable strategies and a clean TUI.\n"
            "Quote your prompt; pass strategy params with -S key=value."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_global_and_positional(parser)
    add_strategy_args(parser)
    add_model_plugin_args(parser)
    add_repo_args(parser)
    add_display_args(parser)
    add_auth_args(parser)
    add_limits_args(parser)
    add_state_args(parser)
    add_maintenance_args(parser)
    add_diag_args(parser)
    return parser
