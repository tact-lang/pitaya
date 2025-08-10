"""
CLI interface for the orchestrator TUI.

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


class OrchestratorTUI:
    """Main orchestrator TUI application."""

    def __init__(self):
        """Initialize TUI application."""
        self.console = Console()
        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.output_mode = "tui"
        self.display_mode = None  # auto by default

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser for TUI."""
        parser = argparse.ArgumentParser(
            description="Orchestrator TUI - Monitor AI coding agent execution",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Connect to running orchestrator
  orchestrator-tui --connect localhost:8080
  
  # Watch events file
  orchestrator-tui --events-file logs/run_20250114_123456/events.jsonl
  
  # Stream events as text
  orchestrator-tui --events-file events.jsonl --output streaming
  
  # Output JSON events
  orchestrator-tui --events-file events.jsonl --output json
""",
        )

        # Connection options
        conn_group = parser.add_mutually_exclusive_group(required=False)
        conn_group.add_argument(
            "--connect", metavar="HOST:PORT", help="Connect to orchestrator HTTP server"
        )
        conn_group.add_argument(
            "--events-file", type=Path, help="Read events from file (offline mode)"
        )

        # Display options
        parser.add_argument(
            "--output",
            choices=["tui", "streaming", "json", "quiet"],
            default="tui",
            help="Output mode (default: tui)",
        )
        parser.add_argument(
            "--display-mode",
            choices=["auto", "detailed", "compact", "dense"],
            default="auto",
            help="Force display mode for TUI (default: auto)",
        )
        parser.add_argument(
            "--from-offset",
            type=int,
            default=0,
            help="Start reading events from offset (default: 0)",
        )

        # Filtering options
        parser.add_argument("--run-id", help="Filter events by run ID")
        parser.add_argument("--instance-id", help="Filter events by instance ID")
        parser.add_argument("--event-types", nargs="+", help="Filter by event types")

        # Other options
        parser.add_argument(
            "--no-color", action="store_true", help="Disable colored output"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")

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
            if args.connect:
                # Connect to orchestrator HTTP server
                return await self._run_connected(args)
            else:
                # If events file not provided but run-id is, infer path
                if not args.events_file and args.run_id:
                    # Respect ORCHESTRATOR_LOGS_DIR if set
                    import os
                    logs_base = os.environ.get("ORCHESTRATOR_LOGS_DIR", "logs")
                    inferred = Path(logs_base) / args.run_id / "events.jsonl"
                    args.events_file = inferred
                # Read from events file
                return await self._run_offline(args)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130
        except (OSError, IOError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            if args.debug:
                self.console.print_exception()
            return 1

    async def _run_connected(self, args: argparse.Namespace) -> int:
        """
        Run TUI connected to orchestrator.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        # Parse connection string
        host, port = args.connect.split(":")
        port = int(port)

        # Import HTTP client
        from src.orchestration.http_client import OrchestratorClient

        # Create client
        client = OrchestratorClient(f"http://{host}:{port}")

        # Test connection
        try:
            state = await client.get_state()
            if not state:
                self.console.print("[red]Failed to connect to orchestrator[/red]")
                return 1
        except (OSError, asyncio.TimeoutError) as e:
            self.console.print(f"[red]Connection failed: {e}[/red]")
            return 1

        # Run appropriate output mode
        if self.output_mode == "tui":
            # Create and run TUI display
            # Read refresh rate and display mode from env when not provided
            import os
            refresh_rate = 0.1
            try:
                rr = os.environ.get("ORCHESTRATOR_TUI__REFRESH_RATE")
                if rr is not None:
                    refresh_rate = float(rr) / 100.0 if float(rr) > 1 else float(rr)
            except Exception:
                pass

            display = TUIDisplay(
                console=self.console, refresh_rate=refresh_rate, state_poll_interval=3.0
            )

            # Override display mode if specified
            if self.display_mode:
                display.state.current_run.force_detail_level = self.display_mode
            else:
                # Try env override
                dm = os.environ.get("ORCHESTRATOR_TUI__FORCE_DISPLAY_MODE")
                if dm in {"detailed", "compact", "dense"} and display.state.current_run:
                    display.state.current_run.force_detail_level = dm

            # Run TUI with client
            await display.run_connected(client, args.from_offset)

        elif self.output_mode == "streaming":
            # Stream events as text
            await self._stream_events_connected(client, args)

        elif self.output_mode == "json":
            # Output events as JSON
            await self._json_events_connected(client, args)

        elif self.output_mode == "quiet":
            # Minimal output
            await self._quiet_mode_connected(client, args)

        return 0

    async def _run_offline(self, args: argparse.Namespace) -> int:
        """
        Run TUI reading from events file.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        events_file = args.events_file

        # Check file exists
        if not events_file.exists():
            self.console.print(f"[red]Events file not found: {events_file}[/red]")
            return 1

        # Run appropriate output mode
        if self.output_mode == "tui":
            # Create and run TUI display
            import os
            refresh_rate = 0.1
            try:
                rr = os.environ.get("ORCHESTRATOR_TUI__REFRESH_RATE")
                if rr is not None:
                    refresh_rate = float(rr) / 100.0 if float(rr) > 1 else float(rr)
            except Exception:
                pass
            display = TUIDisplay(
                console=self.console, refresh_rate=refresh_rate, state_poll_interval=3.0
            )

            # Override display mode if specified
            if self.display_mode and display.state.current_run:
                display.state.current_run.force_detail_level = self.display_mode
            else:
                dm = os.environ.get("ORCHESTRATOR_TUI__FORCE_DISPLAY_MODE")
                if dm in {"detailed", "compact", "dense"} and display.state.current_run:
                    display.state.current_run.force_detail_level = dm

            # Run TUI with file
            await display.run(None, events_file, args.from_offset)

        elif self.output_mode == "streaming":
            # Stream events as text
            await self._stream_events_file(events_file, args)

        elif self.output_mode == "json":
            # Output events as JSON
            await self._json_events_file(events_file, args)

        elif self.output_mode == "quiet":
            # Minimal output
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
            timestamp = event.get("timestamp", "")
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
        with open(events_file, "r") as f:
            if args.from_offset > 0:
                f.seek(args.from_offset)

            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)

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

    async def _stream_events_connected(self, client, args: argparse.Namespace) -> None:
        """Stream events from connected orchestrator."""
        # Get events endpoint
        offset = args.from_offset

        while True:
            events, new_offset = await client.get_events(
                offset=offset,
                limit=100,
                run_id=args.run_id,
                event_types=set(args.event_types) if args.event_types else None,
            )

            for event in events:
                # Format and display
                timestamp = event.get("timestamp", "")
                event_type = event.get("type", "unknown")
                instance_id = event.get("instance_id", "")

                msg = f"[{timestamp}]"
                if instance_id:
                    msg += f" [{instance_id[:8]}]"
                msg += f" {event_type}"

                # Add data
                data = event.get("data", {})
                if "error" in data:
                    msg += f" - Error: {data['error']}"
                elif "activity" in data:
                    msg += f" - {data['activity']}"

                self.console.print(msg)

            offset = new_offset
            await asyncio.sleep(0.5)  # Poll interval

    async def _json_events_connected(self, client, args: argparse.Namespace) -> None:
        """Output JSON events from connected orchestrator."""
        offset = args.from_offset

        while True:
            events, new_offset = await client.get_events(
                offset=offset,
                limit=100,
                run_id=args.run_id,
                event_types=set(args.event_types) if args.event_types else None,
            )

            for event in events:
                print(json.dumps(event))

            offset = new_offset
            await asyncio.sleep(0.5)

    async def _quiet_mode_connected(self, client, args: argparse.Namespace) -> None:
        """Quiet mode for connected orchestrator."""
        # Just get final state
        state = await client.get_state()

        if state:
            self.console.print(f"Run: {state.get('run_id', 'unknown')}")
            self.console.print(f"Total cost: ${state.get('total_cost', 0):.4f}")
            self.console.print(f"Total instances: {state.get('total_instances', 0)}")
            self.console.print(f"Completed: {state.get('completed_instances', 0)}")
            self.console.print(f"Failed: {state.get('failed_instances', 0)}")

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
