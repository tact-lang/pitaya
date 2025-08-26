"""
CLI interface for the Pitaya TUI.

Handles argument parsing and coordinates different output modes:
- TUI: Rich interactive display
- Streaming: Real-time text output
- JSON: Structured JSON events
- Quiet: Minimal output
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from rich.console import Console

from .display import TUIDisplay
from .models import TUIState
from .event_handler import EventProcessor, AsyncEventStream
from .. import __version__


class OrchestratorTUI:
    """Main Pitaya TUI application."""

    def __init__(self):
        """Initialize TUI application."""
        self.console = Console()
        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.output_mode = "tui"
        self.display_mode = None  # auto by default

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser for TUI with clear grouping and examples."""
        parser = argparse.ArgumentParser(
            prog="pitaya-tui",
            description="Monitor Pitaya runs from an events file (TUI or text)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Examples:\n"
                "  # Watch a run by ID\n"
                "  pitaya-tui --run-id run_20250114_123456\n\n"
                "  # Stream as text from a file\n"
                "  pitaya-tui --events-file logs/run_20250114_123456/events.jsonl --output streaming\n\n"
                "  # Emit JSON events (no TUI)\n"
                "  pitaya-tui --events-file events.jsonl --output json\n"
            ),
        )

        # Global
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )

        # Input
        g_input = parser.add_argument_group("Input")
        g_input.add_argument(
            "--events-file", type=Path, help="Read events from file (offline mode)"
        )
        g_input.add_argument(
            "--from-offset", type=int, default=0, help="Start reading from byte offset"
        )
        g_input.add_argument("--run-id", help="Infer events file by run ID")

        # Display
        g_display = parser.add_argument_group("Display")
        g_display.add_argument(
            "--output",
            choices=["tui", "streaming", "json", "quiet"],
            default="tui",
            help="Output mode (default: tui)",
        )
        g_display.add_argument(
            "--display-mode",
            choices=["auto", "detailed", "compact", "dense"],
            default="auto",
            help="Force display density for TUI",
        )
        g_display.add_argument(
            "--no-color", action="store_true", help="Disable colored output"
        )

        # Filters
        g_filters = parser.add_argument_group("Filters")
        g_filters.add_argument("--instance-id", help="Filter by instance ID")
        g_filters.add_argument(
            "--event-types", nargs="+", help="Filter by event type(s)"
        )

        # Diagnostics (debug mode removed; verbose by default)

        return parser

    async def run(self, args: argparse.Namespace) -> int:
        """
        Run TUI with parsed arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Exit code
        """
        self.output_mode = args.output
        self.display_mode = args.display_mode if args.display_mode != "auto" else None

        # Configure console
        if args.no_color:
            self.console = Console(force_terminal=False, no_color=True)

        try:
            # If events file not provided but run-id is, infer path
            if not args.events_file and args.run_id:
                inferred = Path("logs") / args.run_id / "events.jsonl"
                args.events_file = inferred
            # If neither events-file nor run-id provided, show helpful message
            if not args.events_file and not args.run_id:
                self.console.print(
                    "[red]Missing input.[/red] Provide either --events-file <path> or --run-id <id>.\n"
                    "Examples:\n"
                    "  pitaya-tui --run-id run_20250114_123456\n"
                    "  pitaya-tui --events-file logs/run_20250114_123456/events.jsonl"
                )
                return 2
            # Read from events file
            return await self._run_offline(args)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            # Standardized interrupted exit code per spec
            return 2
        except (OSError, IOError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            self.console.print_exception()
            return 1

    # Connected modes removed

    async def _run_offline(self, args: argparse.Namespace) -> int:
        """
        Run TUI reading from events file.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        events_file = args.events_file
        if events_file is None:
            self.console.print(
                "[red]No events file resolved. Use --events-file or --run-id.[/red]"
            )
            return 2

        # Run appropriate output mode
        if self.output_mode == "tui":
            # Create and run TUI display
            refresh_rate = 0.1
            display = TUIDisplay(
                console=self.console, refresh_rate=refresh_rate, state_poll_interval=3.0
            )
            # Apply CLI/env forced display mode via display helper
            display.set_forced_display_mode(self.display_mode)

            # Run TUI with file (waits for file creation internally)
            await display.run(None, events_file, args.from_offset)

        elif self.output_mode == "streaming":
            # Stream events as text (waits for file creation internally)
            await self._stream_events_file(events_file, args)

        elif self.output_mode == "json":
            # Output events as JSON
            # For JSON mode, require the file to already exist
            if not events_file.exists():
                self.console.print(f"[red]Events file not found: {events_file}[/red]")
                return 1
            await self._json_events_file(events_file, args)

        elif self.output_mode == "quiet":
            # Minimal output
            # For quiet mode, require the file to already exist
            if not events_file.exists():
                self.console.print(f"[red]Events file not found: {events_file}[/red]")
                return 1
            await self._quiet_mode_file(events_file, args)

        return 0

    async def _stream_events_file(
        self, events_file: Path, args: argparse.Namespace
    ) -> None:
        """Stream events as human-readable text."""
        # Create event stream
        event_stream = AsyncEventStream(self.event_processor)

        # Custom callback for streaming
        def stream_callback(event: Dict[str, Any]):
            # Apply filters
            if args.run_id and event.get("run_id") != args.run_id:
                return
            if args.instance_id and event.get("instance_id") != args.instance_id:
                return
            if args.event_types and event.get("type") not in args.event_types:
                return

            # Format event for display
            timestamp = event.get("ts") or event.get("timestamp", "")
            event_type = event.get("type", "unknown")
            instance_id = event.get("instance_id", "")

            # Color code by event type
            if "failed" in event_type:
                style = "red"
            elif "completed" in event_type:
                style = "green"
            elif "started" in event_type:
                style = "yellow"
            else:
                style = "white"

            # Build message
            msg = f"[{timestamp}]"
            if instance_id:
                msg += f" [{instance_id[:8]}]"
            msg += f" {event_type}"

            # Add key data points
            data = event.get("data", {})
            if "error" in data:
                msg += f" - Error: {data['error']}"
            elif "activity" in data:
                msg += f" - {data['activity']}"
            elif "tool" in data:
                msg += f" - Tool: {data['tool']}"
            elif "strategy" in data:
                msg += f" - Strategy: {data['strategy']}"

            self.console.print(msg, style=style)

        # Override processor callback
        original_process = self.event_processor.process_event
        self.event_processor.process_event = lambda e: (
            original_process(e),
            stream_callback(e),
        )

        # Start streaming
        await event_stream.start(events_file, args.from_offset)

        # Wait for interrupt
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await event_stream.stop()

    async def _json_events_file(
        self, events_file: Path, args: argparse.Namespace
    ) -> None:
        """Output events as JSON."""
        # Read events directly
        with open(events_file, "rb") as f:
            if args.from_offset > 0:
                # align to newline boundary
                off = args.from_offset
                if off > 0:
                    f.seek(off - 1)
                    if f.read(1) != b"\n":
                        while True:
                            b = f.read(1)
                            if not b or b == b"\n":
                                break
                f.seek(max(0, args.from_offset))

            while True:
                line = f.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="ignore").strip()
                if not s:
                    continue

                try:
                    event = json.loads(s)

                    # Apply filters
                    if args.run_id and event.get("run_id") != args.run_id:
                        continue
                    if (
                        args.instance_id
                        and event.get("instance_id") != args.instance_id
                    ):
                        continue
                    if args.event_types and event.get("type") not in args.event_types:
                        continue

                    # Output JSON
                    print(json.dumps(event))

                except json.JSONDecodeError:
                    continue

    async def _quiet_mode_file(
        self, events_file: Path, args: argparse.Namespace
    ) -> None:
        """Minimal output mode - just summary at end."""
        # Process all events
        event_stream = AsyncEventStream(self.event_processor)
        await event_stream._read_existing_events(events_file, args.from_offset)

        # Print summary
        if self.state.current_run:
            run = self.state.current_run

            self.console.print(f"Run: {run.run_id}")
            self.console.print(f"Total instances: {run.total_instances}")
            self.console.print(f"Completed: {run.completed_instances}")
            self.console.print(f"Failed: {run.failed_instances}")
            self.console.print(f"Success rate: {run.success_rate:.1f}%")
            self.console.print(f"Total cost: ${run.total_cost:.4f}")
            self.console.print(f"Total tokens: {run.total_tokens:,}")
            self.console.print(
                f"Duration: {self._format_duration(run.duration_seconds)}"
            )
        else:
            self.console.print("No run data found")

    # Connected helper methods removed

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def main():
    """Main entry point for TUI CLI."""
    # Create parser
    parser = OrchestratorTUI.create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Create and run TUI
    tui = OrchestratorTUI()

    # Run async
    exit_code = asyncio.run(tui.run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
