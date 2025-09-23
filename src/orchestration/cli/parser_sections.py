"""Small helpers to add argparse groups for the CLI parser.

Each function stays under the handbook function budget and keeps
`cli_parser.create_parser()` lean and readable.
"""

from __future__ import annotations

import argparse
from pathlib import Path


__all__ = [
    "add_global_and_positional",
    "add_strategy_args",
    "add_model_plugin_args",
    "add_repo_args",
    "add_display_args",
    "add_auth_args",
    "add_limits_args",
    "add_state_args",
    "add_maintenance_args",
    "add_diag_args",
]


def add_global_and_positional(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt", nargs="?", help="Task prompt, or 'doctor'/'config'")
    parser.add_argument(
        "subcommand", nargs="?", help="Subcommand for modes like 'config print'"
    )


def add_strategy_args(parser: argparse.ArgumentParser) -> None:
    from ..strategies import AVAILABLE_STRATEGIES

    g = parser.add_argument_group("Strategy")
    g.add_argument(
        "--strategy",
        default="simple",
        help=(
            "Execution strategy. Built-ins: "
            + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys())))
            + ". File/module: path.py[:Class] or package.module[:Class]"
        ),
    )
    g.add_argument("-S", action="append", dest="strategy_params", metavar="KEY=VALUE")
    g.add_argument(
        "--set", action="append", dest="strategy_params", metavar="KEY=VALUE"
    )
    g.add_argument(
        "--runs", type=int, default=1, help="Parallel executions (default: 1)"
    )


def _plugin_choices() -> list[str]:
    try:
        from ...instance_runner.plugins import AVAILABLE_PLUGINS as _APLUG

        return sorted(list(_APLUG.keys()))
    except (ImportError, Exception):
        return ["claude-code"]


def add_model_plugin_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Model & Plugin")
    g.add_argument("--model", type=str, default="sonnet", help="Model alias")
    g.add_argument(
        "--plugin",
        choices=_plugin_choices(),
        default="claude-code",
        help="Runner plugin (e.g., claude-code, codex)",
    )
    g.add_argument("--docker-image", type=str, help="Override Docker image")
    g.add_argument(
        "--cli-arg",
        action="append",
        dest="agent_cli_arg",
        metavar="ARG",
        help="Pass single arg to agent CLI (repeatable)",
    )
    g.add_argument(
        "--cli-args",
        dest="agent_cli_args_str",
        metavar="'ARGS'",
        help="Quoted args string for agent CLI",
    )


def add_repo_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Repository")
    g.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to git repo")
    g.add_argument("--base-branch", default="main", help="Base branch to work from")
    g.add_argument(
        "--include-branches",
        metavar="BRANCHES",
        help=(
            "Extra branches to include read-only in the workspace for all tasks. "
            'CSV (a,b,c) or JSON list (e.g., \'["a","b"]\').'
        ),
    )
    g.add_argument(
        "--require-clean-wt", action="store_true", help="Fail if working tree is dirty"
    )
    g.add_argument(
        "--allow-overwrite-protected-refs",
        action="store_true",
        help="Allow overwriting protected refs (dangerous)",
    )


def add_display_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Display & Output")
    g.add_argument(
        "--no-tui", action="store_true", help="Disable TUI; stream to console"
    )
    g.add_argument(
        "--display",
        choices=["auto", "detailed", "compact", "dense"],
        default="auto",
        help="TUI density",
    )
    g.add_argument(
        "--output",
        choices=["streaming", "json", "quiet"],
        default="streaming",
        help="Headless output format (json emits NDJSON only)",
    )
    g.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --no-tui --output json (NDJSON only)",
    )
    g.add_argument("--no-emoji", action="store_true", help="Disable emoji in output")
    g.add_argument(
        "--show-ids", choices=["short", "full"], default="short", help="ID verbosity"
    )
    g.add_argument("--verbose", action="store_true", help="Verbose streaming logs")


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Auth")
    g.add_argument("--mode", choices=["subscription", "api"], help="Auth mode")
    g.add_argument("--oauth-token", help="OAuth token (or CLAUDE_CODE_OAUTH_TOKEN)")
    g.add_argument("--api-key", help="API key (or ANTHROPIC/OPENAI API KEY env)")
    g.add_argument("--base-url", help="Custom API base URL")


def add_limits_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Execution & Limits")
    g.add_argument(
        "--max-parallel", type=int, default=None, help="Max instances (auto)"
    )
    g.add_argument(
        "--max-startup-parallel",
        type=int,
        default=None,
        help="Max parallel startups (default: min(10,max-parallel))",
    )
    g.add_argument("--timeout", type=int, default=3600, help="Timeout per instance (s)")
    g.add_argument(
        "--force-commit",
        action="store_true",
        help="Force a commit if workspace changed",
    )
    g.add_argument(
        "--randomize-queue",
        dest="randomize_queue",
        action="store_true",
        help="Randomize queued instance order",
    )


def add_state_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Config, State & Logs")
    g.add_argument("--config", type=Path, help="Config file (default pitaya.yaml)")
    g.add_argument("--state-dir", type=Path, default=Path("./pitaya_state"))
    g.add_argument("--logs-dir", type=Path, default=Path("./logs"))
    g.add_argument(
        "--redact",
        choices=["true", "false"],
        default="true",
        help="For 'config print': redact secrets",
    )


def add_maintenance_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Maintenance")
    g.add_argument("--resume", metavar="RUN_ID", help="Resume an interrupted run")
    g.add_argument("--list-runs", action="store_true", help="List previous runs")
    g.add_argument("--show-run", metavar="RUN_ID", help="Show run details")
    g.add_argument(
        "--override-config",
        action="store_true",
        help="Allow overrides to the saved run configuration on resume",
    )
    g.add_argument(
        "--resume-key-policy",
        choices=["strict", "suffix"],
        default="strict",
        help="When overriding config on resume: 'strict' enforces durable-key fidelity; 'suffix' appends a resume suffix to new task keys",
    )


def add_diag_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Diagnostics")
    g.add_argument("--yes", action="store_true", help="Assume 'yes' for prompts")
