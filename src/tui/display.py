"""
Rich-based TUI display for the orchestrator.

Implements the three-zone layout (header, dashboard, footer) with
adaptive display based on instance count.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Any
from pathlib import Path
import os

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

from .models import TUIState, InstanceStatus
from .event_handler import EventProcessor, AsyncEventStream
from .adaptive import AdaptiveDisplay

logger = logging.getLogger(__name__)


class TUIDisplay:
    """Main TUI display using Rich."""

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_rate: float = 0.1,  # 10Hz as per spec
        state_poll_interval: float = 3.0,  # 3 seconds as per spec
    ):
        """
        Initialize TUI display.

        Args:
            console: Rich console (creates new if not provided)
            refresh_rate: Display refresh rate in seconds
            state_poll_interval: State reconciliation interval
        """
        self.console = console or Console()
        self.refresh_rate = refresh_rate
        logger.info(f"TUI refresh rate: {refresh_rate}s ({1/refresh_rate}Hz)")
        self.state_poll_interval = state_poll_interval

        # Core components
        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.event_stream = AsyncEventStream(self.event_processor)
        self.adaptive_display = AdaptiveDisplay()
        # Optional env override for display mode
        self._force_display_mode_env = os.environ.get("ORCHESTRATOR_TUI__FORCE_DISPLAY_MODE")

        # Display state
        self._live: Optional[Live] = None
        self._shutdown = False
        self._orchestrator = None  # Set when connected

        # Layout components
        self._layout = self._create_layout()

    def _create_layout(self) -> Layout:
        """Create the three-zone layout."""
        layout = Layout()

        # Split into header, body, footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),  # Footer needs space for content
        )

        # Initialize with placeholder content
        layout["header"].update(Panel("Initializing...", style="blue"))
        layout["body"].update(Panel("Loading...", style="dim"))
        layout["footer"].update(Panel("Starting...", style="blue"))

        return layout

    async def run(self, orchestrator, events_file: Path, from_offset: int = 0) -> None:
        """
        Run the TUI display.

        Args:
            orchestrator: Orchestrator instance for state queries
            events_file: Path to events.jsonl
            from_offset: Starting offset in events file
        """
        logger.info(f"TUI starting with events file: {events_file}")
        self._orchestrator = orchestrator

        # Start event stream - specification requires file watching for all components
        await self.event_stream.start(events_file, from_offset)

        # Start state polling
        asyncio.create_task(self._state_polling_loop())

        # Start display
        with Live(
            self._layout,
            console=self.console,
            refresh_per_second=10,  # Auto refresh at 10Hz
            screen=False,  # Don't use alternate screen
            vertical_overflow="crop",  # Crop content that goes beyond screen
            transient=False,  # Keep content on screen after exit
        ) as live:
            self._live = live

            # Force initial refresh
            live.refresh()

            # Run render loop
            await self._render_loop()

    async def run_connected(self, client, from_offset: int = 0) -> None:
        """
        Run the TUI display connected to orchestrator HTTP server.

        Args:
            client: OrchestratorClient instance
            from_offset: Starting event offset
        """
        self._orchestrator = client

        # Start event polling from HTTP
        asyncio.create_task(self._http_event_polling_loop(client, from_offset))

        # Start state polling
        asyncio.create_task(self._state_polling_loop())

        # Start display
        with Live(
            self._layout,
            console=self.console,
            refresh_per_second=10,  # Auto refresh at 10Hz
            screen=False,  # Don't use alternate screen
            vertical_overflow="crop",  # Crop content that goes beyond screen
            transient=False,  # Keep content on screen after exit
        ) as live:
            self._live = live

            # Force initial refresh
            live.refresh()

            # Run render loop
            await self._render_loop()

    async def stop(self) -> None:
        """Stop the TUI display."""
        logger.info("TUI stop() called")
        self._shutdown = True
        await self.event_stream.stop()
        # Force the Live display to stop
        if self._live:
            self._live.stop()

    async def _render_loop(self) -> None:
        """Main render loop at fixed rate."""
        logger.info("Render loop started")
        iterations = 0
        while not self._shutdown:
            try:
                iterations += 1
                if iterations % 10 == 0:  # Log every 10 iterations
                    logger.info(f"Render loop iteration {iterations}")

                # Update layout components
                self._update_header()
                self._update_dashboard()
                self._update_footer()

                # Force refresh the live display
                if self._live:
                    self._live.refresh()

                # Wait for next render cycle
                await asyncio.sleep(self.refresh_rate)

            except asyncio.CancelledError:
                # This is expected when shutting down
                logger.debug("Render loop cancelled")
                break
            except KeyboardInterrupt:
                # Propagate interrupt immediately
                logger.debug("Render loop interrupted")
                raise
            except OSError as e:
                self.state.add_error(f"Render error: {e}")
                logger.error(f"Render error: {e}", exc_info=True)

    async def _state_polling_loop(self) -> None:
        """Periodic state reconciliation loop."""
        while not self._shutdown:
            try:
                # Poll orchestrator state
                if self._orchestrator:
                    if hasattr(self._orchestrator, "get_current_state"):
                        # Direct orchestrator instance
                        state = self._orchestrator.get_current_state()
                    else:
                        # HTTP client
                        state = await self._orchestrator.get_state()

                    if state:
                        self._reconcile_state(state)
                        # Apply forced display mode from env if provided
                        if (
                            self._force_display_mode_env
                            and self.state.current_run
                            and not self.state.current_run.force_detail_level
                        ):
                            mode = self._force_display_mode_env.strip().lower()
                            if mode in ("detailed", "compact", "dense"):
                                self.state.current_run.force_detail_level = mode

                self.state.last_state_poll = datetime.now()

                # Wait for next poll
                await asyncio.sleep(self.state_poll_interval)

            except asyncio.CancelledError:
                # Expected during shutdown
                break
            except KeyboardInterrupt:
                # Propagate interrupt immediately
                raise
            except (OSError, AttributeError) as e:
                self.state.add_error(f"State poll error: {e}")

    async def _http_event_polling_loop(self, client, from_offset: int) -> None:
        """Poll for events from HTTP server."""
        offset = from_offset

        while not self._shutdown:
            try:
                # Get events from server
                events, new_offset = await client.get_events(offset=offset, limit=100)

                # Process each event
                for event in events:
                    self.event_processor.process_event(event)

                # Update offset
                offset = new_offset
                self.state.last_event_offset = offset

                # Short delay between polls
                await asyncio.sleep(0.5)

            except (OSError, asyncio.CancelledError, AttributeError) as e:
                self.state.add_error(f"Event poll error: {e}")
                await asyncio.sleep(2.0)  # Back off on error

    def _reconcile_state(self, orchestrator_state: Any) -> None:
        """Reconcile orchestrator state with TUI state."""
        # This ensures we catch any missed events
        # The orchestrator state is authoritative

        if not self.state.current_run:
            return

        run = self.state.current_run

        # Update aggregate metrics
        run.total_cost = orchestrator_state.total_cost
        run.total_tokens = orchestrator_state.total_tokens
        run.total_instances = orchestrator_state.total_instances
        run.completed_instances = orchestrator_state.completed_instances
        run.failed_instances = orchestrator_state.failed_instances

    def _update_header(self) -> None:
        """Update header zone."""
        try:
            if not self.state.current_run:
                # No active run
                header_content = Text(
                    "Orchestrator TUI - No Active Run", style="bold yellow"
                )
            else:
                run = self.state.current_run
                # Simple header text for now
                header_content = Text()
                header_content.append(f"Run: {run.run_id}", style="bold cyan")
                header_content.append(" | ")
                header_content.append(
                    f"Strategy: {self._get_strategy_summary()}", style="green"
                )
                # Model (from first strategy config if available)
                try:
                    model = None
                    if run.strategies:
                        first = next(iter(run.strategies.values()))
                        model = first.config.get("model") if hasattr(first, "config") else None
                    if model:
                        header_content.append(" | ")
                        header_content.append(f"Model: {model}", style="magenta")
                except Exception:
                    pass
                header_content.append(" | ")
                header_content.append(
                    f"Started: {self._format_time(run.started_at)}", style="dim"
                )
                # Total runtime inline
                try:
                    if run.started_at:
                        elapsed = datetime.now(timezone.utc) - run.started_at
                        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
                        minutes, seconds = divmod(rem, 60)
                        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        header_content.append(" | ")
                        header_content.append(f"Runtime: {duration}", style="dim")
                except Exception:
                    pass

                # Instance counts summary in header for quick glance
                try:
                    total = len(run.instances)
                    running = sum(1 for i in run.instances.values() if i.status == InstanceStatus.RUNNING)
                    completed = sum(1 for i in run.instances.values() if i.status == InstanceStatus.COMPLETED)
                    failed = sum(1 for i in run.instances.values() if i.status == InstanceStatus.FAILED)
                    header_content.append(" | ")
                    header_content.append(
                        f"Instances: {total} (R:{running} C:{completed} F:{failed})",
                        style="cyan",
                    )
                except Exception:
                    pass

            # Update layout with simple panel
            self._layout["header"].update(
                Panel(Align.center(header_content), style="blue", height=3)
            )
        except (AttributeError, TypeError, KeyError) as e:
            # Fallback on error
            self._layout["header"].update(Panel(f"Header Error: {e}", style="red"))

    def _update_dashboard(self) -> None:
        """Update dashboard zone."""
        try:
            if not self.state.current_run:
                # No active run
                self._layout["body"].update(
                    Panel(
                        Align.center(
                            Text("Waiting for orchestration run...", style="dim")
                        ),
                        style="dim",
                    )
                )
                return

            # Get display mode
            display_mode = self.state.current_run.get_display_mode()

            # Get dashboard content from adaptive display
            dashboard_content = self.adaptive_display.render_dashboard(
                self.state.current_run, display_mode
            )

            self._layout["body"].update(dashboard_content)
        except (AttributeError, TypeError, KeyError) as e:
            self._layout["body"].update(Panel(f"Dashboard Error: {e}", style="red"))

    def _update_footer(self) -> None:
        """Update footer zone with aggregate metrics."""
        try:
            # Build footer text as simple string
            footer_lines = []

            # Show stats as per specification
            if self.state.current_run:
                run = self.state.current_run

                # Count instances by status
                active_count = sum(
                    1
                    for i in run.instances.values()
                    if i.status == InstanceStatus.RUNNING
                )
                completed_count = sum(
                    1
                    for i in run.instances.values()
                    if i.status == InstanceStatus.COMPLETED
                )
                failed_count = sum(
                    1
                    for i in run.instances.values()
                    if i.status == InstanceStatus.FAILED
                )

                # Calculate duration
                duration = "--:--:--"
                if run.started_at:
                    elapsed = datetime.now(timezone.utc) - run.started_at
                    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                # Get total cost
                total_cost = sum(i.cost for i in run.instances.values())

                # First line: instance counts and duration
                line1 = f"Instances: {len(run.instances)} | Active: {active_count} | Completed: {completed_count} | Failed: {failed_count} | Duration: {duration}"
                footer_lines.append(line1)

                # Second line: events and cost
                line2 = (
                    f"Events: {self.state.events_processed} | Cost: ${total_cost:.4f}"
                )
                footer_lines.append(line2)
            else:
                footer_lines.append(
                    f"Not connected | Events: {self.state.events_processed}"
                )

            footer_content = "\n".join(footer_lines)
            self._layout["footer"].update(Panel(footer_content, style="blue"))
        except (AttributeError, TypeError, KeyError) as e:
            # Log the error for debugging
            logger.error(f"Footer rendering error: {e}", exc_info=True)
            self._layout["footer"].update(Panel(f"Footer Error: {e}", style="red"))

    def _get_strategy_summary(self) -> str:
        """Get strategy summary for header."""
        if not self.state.current_run or not self.state.current_run.strategies:
            return "Unknown"

        # Get unique strategy names
        strategy_names = set(
            s.strategy_name for s in self.state.current_run.strategies.values()
        )

        if len(strategy_names) == 1:
            name = list(strategy_names)[0]
            count = len(self.state.current_run.strategies)
            if count > 1:
                return f"{name} (x{count})"
            return name
        else:
            return f"Multiple ({len(self.state.current_run.strategies)})"

    def _format_time(self, dt: Optional[datetime]) -> str:
        """Format datetime for display."""
        if not dt:
            return "N/A"

        # Make timezone-aware if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.strftime("%H:%M:%S")

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
