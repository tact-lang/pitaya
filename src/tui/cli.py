"""
CLI interface for the Pitaya TUI.

Handles argument parsing and coordinates output modes:
- TUI (interactive)
- streaming (text)
- json (NDJSON)
- quiet (summary)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from .cli_modes import json_events_file, quiet_mode_file, stream_events_file
from .cli_parser import create_tui_parser
from .display import TUIDisplay
from .models import TUIState
from .event_handler import EventProcessor


class OrchestratorTUI:
    """Main Pitaya TUI application."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.output_mode = "tui"
        self.display_mode = None  # auto by default

    async def run(self, args: argparse.Namespace) -> int:
        self.output_mode = args.output
        self.display_mode = args.display_mode if args.display_mode != "auto" else None

        if args.no_color:
            self.console = Console(force_terminal=False, no_color=True)

        try:
            if not args.events_file and args.run_id:
                args.events_file = Path("logs") / args.run_id / "events.jsonl"
            if not args.events_file and not args.run_id:
                self.console.print(
                    "[red]Missing input.[/red] Provide either --events-file <path> or --run-id <id>.\n"
                    "Examples:\n"
                    "  pitaya-tui --run-id run_20250114_123456\n"
                    "  pitaya-tui --events-file logs/run_20250114_123456/events.jsonl"
                )
                return 2
            return await self._run_offline(args)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            return 2
        except (OSError, IOError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            self.console.print_exception()
            return 1

    async def _run_offline(self, args: argparse.Namespace) -> int:
        events_file = args.events_file
        if events_file is None:
            self.console.print(
                "[red]No events file resolved. Use --events-file or --run-id.[/red]"
            )
            return 2

        if self.output_mode == "tui":
            display = TUIDisplay(
                console=self.console, refresh_rate=0.1, state_poll_interval=3.0
            )
            display.set_forced_display_mode(self.display_mode)
            await display.run(None, events_file, args.from_offset)
        elif self.output_mode == "streaming":
            await stream_events_file(
                self.console, self.event_processor, events_file, args
            )
        elif self.output_mode == "json":
            if not events_file.exists():
                self.console.print(f"[red]Events file not found: {events_file}[/red]")
                return 1
            await json_events_file(events_file, args)
        elif self.output_mode == "quiet":
            if not events_file.exists():
                self.console.print(f"[red]Events file not found: {events_file}[/red]")
                return 1
            await quiet_mode_file(self.console, self.event_processor, events_file, args)
        return 0


def main() -> None:
    parser = create_tui_parser()
    args = parser.parse_args()
    tui = OrchestratorTUI()
    exit_code = asyncio.run(tui.run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
