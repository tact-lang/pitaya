"""CLI parser builder for the Pitaya orchestrator.

Provides a single `create_parser()` function so the top-level CLI module
can remain thin and focused on orchestration. Kept argparse-based to avoid
new runtime dependencies and preserve existing UX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .strategies import AVAILABLE_STRATEGIES

__all__ = ["create_parser"]


def _plugin_choices() -> list[str]:
    try:
        from ..instance_runner.plugins import AVAILABLE_PLUGINS as _APLUG

        return sorted(list(_APLUG.keys()))
    except Exception:
        return ["claude-code"]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Pitaya with clear grouping and examples."""
    parser = argparse.ArgumentParser(
        prog="pitaya",
        description=(
            "Orchestrate AI coding agents with pluggable strategies and a clean TUI.\n"
            "Quote your prompt; pass strategy params with -S key=value."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
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
            '  pitaya "task" --json\n\n'
            "  # Run management\n"
            "  pitaya --list-runs\n"
            "  pitaya --show-run run_20250114_123456\n"
            "  pitaya --resume run_20250114_123456\n\n"
            "Tips:\n"
            "  • Built-ins: "
            + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys())))
            + "\n"
            "  • Strategies accept -S key=value (numbers/bools auto-parsed).\n"
            "  • --model picks the agent model; --plugin chooses the runner.\n"
        ),
    )

    # Global flags
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Main argument - the task prompt or subcommand (doctor | config)
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task prompt, or 'doctor'/'config' for utility modes",
    )
    # Optional secondary token for 'config print'
    parser.add_argument(
        "subcommand",
        nargs="?",
        help="Subcommand for special modes (e.g., 'print' for 'config print')",
    )

    # Strategy group
    g_strategy = parser.add_argument_group("Strategy")
    g_strategy.add_argument(
        "--strategy",
        default="simple",
        help=(
            "Execution strategy. Built-ins: "
            + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys())))
            + ". Or pass a file/module: path/to/strategy.py[:Class] or package.module[:Class]"
        ),
    )
    g_strategy.add_argument(
        "-S",
        action="append",
        dest="strategy_params",
        metavar="KEY=VALUE",
        help="Strategy parameter (repeatable)",
    )
    g_strategy.add_argument(
        "--set",
        action="append",
        dest="strategy_params",
        metavar="KEY=VALUE",
        help="Alias for -S",
    )
    g_strategy.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Parallel strategy executions (default: 1)",
    )

    # Model & plugin group
    g_model = parser.add_argument_group("Model & Plugin")
    g_model.add_argument(
        "--model", type=str, default="sonnet", help="Model name (plugin-agnostic)"
    )
    g_model.add_argument(
        "--plugin",
        choices=_plugin_choices(),
        default="claude-code",
        help="Runner plugin (e.g., claude-code, codex)",
    )
    g_model.add_argument(
        "--docker-image",
        type=str,
        help="Override Docker image (e.g., myrepo/custom:tag)",
    )
    # Agent CLI passthrough
    g_model.add_argument(
        "--cli-arg",
        action="append",
        dest="agent_cli_arg",
        metavar="ARG",
        help="Pass-through single argument to the agent CLI (repeatable)",
    )
    g_model.add_argument(
        "--cli-args",
        dest="agent_cli_args_str",
        metavar="'ARGS'",
        help="Quoted list of arguments to pass to the agent CLI (e.g., --cli-args '-c key=\"v\" --flag')",
    )

    # Repository group
    g_repo = parser.add_argument_group("Repository")
    g_repo.add_argument(
        "--repo", type=Path, default=Path.cwd(), help="Path to git repository"
    )
    g_repo.add_argument(
        "--base-branch", default="main", help="Base branch to work from"
    )
    g_repo.add_argument(
        "--require-clean-wt", action="store_true", help="Fail if working tree is dirty"
    )
    g_repo.add_argument(
        "--allow-overwrite-protected-refs",
        action="store_true",
        help="Allow overwriting protected refs (use with extreme caution)",
    )

    # Display & output group
    g_display = parser.add_argument_group("Display & Output")
    g_display.add_argument(
        "--no-tui", action="store_true", help="Disable TUI; stream output to console"
    )
    g_display.add_argument(
        "--display",
        choices=["auto", "detailed", "compact", "dense"],
        default="auto",
        help="TUI density (default: auto)",
    )
    g_display.add_argument(
        "--output",
        choices=["streaming", "json", "quiet"],
        default="streaming",
        help="Output format when --no-tui (default: streaming)",
    )
    g_display.add_argument(
        "--json",
        action="store_true",
        help="Convenience: JSON output (implies --no-tui --output json)",
    )
    g_display.add_argument(
        "--no-emoji", action="store_true", help="Disable emoji in console output"
    )
    g_display.add_argument(
        "--show-ids",
        choices=["short", "full"],
        default="short",
        help="Identifier verbosity in streaming output",
    )
    g_display.add_argument(
        "--verbose",
        action="store_true",
        help="More verbose streaming logs (container phases, imports)",
    )

    # Auth group
    g_auth = parser.add_argument_group("Auth")
    g_auth.add_argument(
        "--mode",
        choices=["subscription", "api"],
        help="Auth mode (default: auto-detect)",
    )
    g_auth.add_argument(
        "--oauth-token", help="OAuth token (or set CLAUDE_CODE_OAUTH_TOKEN)"
    )
    g_auth.add_argument(
        "--api-key", help="API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY)"
    )
    g_auth.add_argument(
        "--base-url", help="Custom API base URL (e.g., OpenAI/Anthropic proxy)"
    )

    # Execution & limits group
    g_limits = parser.add_argument_group("Execution & Limits")
    g_limits.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max parallel instances (auto if omitted)",
    )
    g_limits.add_argument(
        "--max-startup-parallel",
        type=int,
        default=None,
        help="Max parallel startup initializations (default: min(10,max-parallel))",
    )
    g_limits.add_argument(
        "--timeout", type=int, default=3600, help="Timeout per instance (seconds)"
    )
    g_limits.add_argument(
        "--force-commit",
        action="store_true",
        help="Force a git commit in the workspace after agent finishes (if there are changes)",
    )
    g_limits.add_argument(
        "--randomize-queue",
        dest="randomize_queue",
        action="store_true",
        help="Execute queued instances in random order instead of FIFO",
    )

    # Config, state & logs group
    g_state = parser.add_argument_group("Config, State & Logs")
    g_state.add_argument(
        "--config", type=Path, help="Config file (default: pitaya.yaml if present)"
    )
    g_state.add_argument(
        "--state-dir", type=Path, default=Path("./pitaya_state"), help="State directory"
    )
    g_state.add_argument(
        "--logs-dir", type=Path, default=Path("./logs"), help="Logs directory"
    )
    g_state.add_argument(
        "--redact",
        choices=["true", "false"],
        default="true",
        help="For 'config print': redact secrets (default: true)",
    )

    # Maintenance group
    g_maint = parser.add_argument_group("Maintenance")
    g_maint.add_argument("--resume", metavar="RUN_ID", help="Resume an interrupted run")
    g_maint.add_argument("--list-runs", action="store_true", help="List previous runs")
    g_maint.add_argument("--show-run", metavar="RUN_ID", help="Show run details")

    # Diagnostics group
    g_diag = parser.add_argument_group("Diagnostics")
    g_diag.add_argument("--yes", action="store_true", help="Assume 'yes' for prompts")

    return parser
