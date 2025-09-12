#!/usr/bin/env python3
"""Pitaya CLI entrypoint (lean, testable, handbook-compliant).

Provides a thin shell that delegates to small, cohesive helpers under
`src/orchestration/cli/` and a separate parser builder.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Final

from rich.console import Console

from .orchestration.cli_parser import create_parser
from .orchestration.cli import config_print, doctor, orchestrator_runner, runs

__all__: Final = ["main"]


async def _dispatch(console: Console, args: argparse.Namespace) -> int:
    cmd = (args.prompt or "").strip().lower()
    sub = (args.subcommand or "").strip().lower()
    if cmd == "doctor":
        return await doctor.run_doctor(console, args)
    if cmd == "config" and sub == "print":
        return await config_print.run_config_print(console, args)
    if getattr(args, "list_runs", False):
        return await runs.run_list_runs(console, args)
    if getattr(args, "show_run", None):
        return await runs.run_show_run(console, args)
    return await orchestrator_runner.run(console, args)


def main() -> None:
    """CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()
    console = Console()

    # Let downstream code handle KeyboardInterrupt and print helpful hints.
    # If an interrupt somehow bubbles this far, allow it to terminate normally
    # so the shell shows the standard ^C without swallowing prior output.
    rc = asyncio.run(_dispatch(console, args))
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
