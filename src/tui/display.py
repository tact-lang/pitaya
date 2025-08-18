"""
Rich-based TUI display for the orchestrator.

Implements the three-zone layout (header, dashboard, footer) with
adaptive display based on instance count.
"""

import asyncio
import time
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

from .models import TUIState, InstanceStatus, RunDisplay, StrategyDisplay, InstanceDisplay
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
        # Ensure we have a terminal-capable console for Live rendering (works under wrappers like uv)
        if console is not None:
            self.console = console
            try:
                if not getattr(self.console, "is_terminal", True):
                    self.console = Console(force_terminal=True)
            except Exception:
                # Fallback to force terminal
                self.console = Console(force_terminal=True)
        else:
            self.console = Console(force_terminal=True)
        self.refresh_rate = refresh_rate
        logger.info(f"TUI refresh rate: {refresh_rate}s ({1/refresh_rate}Hz)")
        self.state_poll_interval = state_poll_interval

        # Core components
        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.event_stream = AsyncEventStream(self.event_processor)
        self.adaptive_display = AdaptiveDisplay()
        # Remove env toggles; rely on CLI and defaults
        self._force_display_mode_env = None
        self._alt_screen = True
        # Optional CLI override for display mode
        self._force_display_mode_cli: Optional[str] = None

        # Display state
        self._live: Optional[Live] = None
        self._shutdown = False
        self._orchestrator = None  # Kept for offline state reconciliation
        # Render on the primary console; stdout is redirected by Live to avoid corruption
        # Immutable snapshot for rendering to avoid races at completion bursts
        self._render_run: Optional[RunDisplay] = None
        # UI session start time (authoritative origin for global Runtime)
        from datetime import datetime as _dt
        self._ui_started_at = _dt.now(timezone.utc)

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

        # Prime state immediately to avoid initial '--:--:--' duration delay
        try:
            if self._orchestrator and hasattr(self._orchestrator, "get_current_state"):
                st = self._orchestrator.get_current_state()
                if st:
                    self._reconcile_state(st)
        except Exception:
            pass

        # Start event stream - specification requires file watching for all components
        await self.event_stream.start(events_file, from_offset)

        # Start state polling
        asyncio.create_task(self._state_polling_loop())

        # Start display (single Live session; no watchdog restarts)
        with Live(
            self._layout,
            console=self.console,
            refresh_per_second=10,  # 10Hz default
            screen=self._alt_screen,  # Alternate screen unless disabled
            vertical_overflow="crop",  # Crop content that goes beyond screen
            transient=self._alt_screen,  # Clear on exit only in alt-screen mode
            # Capture stdout to prevent stray prints from corrupting the Live screen
            redirect_stdout=True,
            redirect_stderr=False,
        ) as live:
            self._live = live

            # Force initial refresh
            live.update(self._layout)

            # Run render loop
            await self._render_loop()

    # Connected display mode removed

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
                if iterations % 50 == 0:  # Log periodically, not too chatty
                    logger.debug(f"Render loop iteration {iterations}")

                t0 = time.perf_counter()
                # Synchronize all duration calculations to a single timestamp
                from datetime import datetime as _dt
                self._frame_now = _dt.now(timezone.utc)
                # Build immutable snapshot for this frame to avoid shared-state mutations
                self._snapshot_state_for_render()
                t_snap = time.perf_counter()

                # Update layout components
                self._update_header()
                t_head = time.perf_counter()
                self._update_dashboard()
                t_body = time.perf_counter()
                self._update_footer()
                t_foot = time.perf_counter()

                # Force refresh the live display
                if self._live:
                    # Explicit update + refresh to force repaint even under TTY wrappers
                    self._live.update(self._layout)
                    try:
                        self._live.refresh()
                    except Exception:
                        pass
                t_paint = time.perf_counter()

                # Wait for next render cycle
                # Emit diagnostics for this frame
                run = self._render_run
                mode = run.get_display_mode() if run else "-"
                insts = len(run.instances) if run else 0
                stats = {}
                try:
                    stats = self.event_stream.get_stats()
                except Exception:
                    pass
                logger.debug(
                    "frame_ms=%.2f snap_ms=%.2f head_ms=%.2f body_ms=%.2f foot_ms=%.2f paint_ms=%.2f "
                    "mode=%s instances=%d events_processed=%d last_off=%d qsize=%s enq=%s proc=%s rot=%s trunc=%s pos=%s size=%s"
                    % (
                        (t_paint - t0) * 1000.0,
                        (t_snap - t0) * 1000.0,
                        (t_head - t_snap) * 1000.0 if t_head and t_snap else 0.0,
                        (t_body - t_head) * 1000.0 if t_body and t_head else 0.0,
                        (t_foot - t_body) * 1000.0 if t_foot and t_body else 0.0,
                        (t_paint - t_foot) * 1000.0 if t_paint and t_foot else 0.0,
                        mode,
                        insts,
                        self.state.events_processed,
                        self.state.last_event_start_offset,
                        stats.get("queue_size", -1),
                        stats.get("lines_enqueued", -1),
                        stats.get("lines_processed", -1),
                        stats.get("rotations", -1),
                        stats.get("truncations", -1),
                        stats.get("last_position", -1),
                        stats.get("prev_size", -1),
                    )
                )

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
            except Exception as e:
                # Catch-all to prevent the render loop from dying on transient races
                self.state.add_error(f"Render loop error: {type(e).__name__}: {e}")
                logger.error("Render loop exception", exc_info=True)

    async def _state_polling_loop(self) -> None:
        """Periodic state reconciliation loop."""
        while not self._shutdown:
            try:
                # Poll orchestrator state (direct instance only)
                if self._orchestrator and hasattr(self._orchestrator, "get_current_state"):
                    state = self._orchestrator.get_current_state()
                    if state:
                        self._reconcile_state(state)

                # Apply forced display mode (CLI only) if provided
                if self.state.current_run and not self.state.current_run.force_detail_level:
                    mode = (self._force_display_mode_cli or "").strip().lower() if self._force_display_mode_cli else None
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

    # Network polling loop removed

    def _reconcile_state(self, orchestrator_state: Any) -> None:
        """Reconcile orchestrator state with TUI state."""
        # This ensures we catch any missed events
        # The orchestrator state is authoritative

        if not self.state.current_run:
            return

        run = self.state.current_run

        # Update aggregate metrics (support dict- or attr-style state)
        try:
            if isinstance(orchestrator_state, dict):
                run.total_cost = orchestrator_state.get("total_cost", run.total_cost)
                run.total_tokens = orchestrator_state.get("total_tokens", run.total_tokens)
                run.total_instances = orchestrator_state.get("total_instances", run.total_instances)
                run.completed_instances = orchestrator_state.get("completed_instances", run.completed_instances)
                run.failed_instances = orchestrator_state.get("failed_instances", run.failed_instances)
                # Times (parse ISO strings to datetime when needed)
                try:
                    from datetime import datetime as _dt
                    sa = orchestrator_state.get("started_at")
                    ca = orchestrator_state.get("completed_at")
                    if isinstance(sa, str):
                        run.started_at = _dt.fromisoformat(sa.replace("Z", "+00:00"))
                    elif sa is not None:
                        run.started_at = sa
                    if isinstance(ca, str):
                        run.completed_at = _dt.fromisoformat(ca.replace("Z", "+00:00"))
                    elif ca is not None:
                        run.completed_at = ca
                except Exception:
                    pass
            else:
                run.total_cost = getattr(orchestrator_state, "total_cost", run.total_cost)
                run.total_tokens = getattr(orchestrator_state, "total_tokens", run.total_tokens)
                run.total_instances = getattr(orchestrator_state, "total_instances", run.total_instances)
                run.completed_instances = getattr(orchestrator_state, "completed_instances", run.completed_instances)
                run.failed_instances = getattr(orchestrator_state, "failed_instances", run.failed_instances)
                # Times
                try:
                    run.started_at = getattr(orchestrator_state, "started_at", run.started_at)
                    run.completed_at = getattr(orchestrator_state, "completed_at", run.completed_at)
                except Exception:
                    pass
        except Exception:
            # Keep run metrics unchanged on reconciliation error
            pass

        # Update per-instance basics so UI reflects progress without legacy events
        try:
            instances_map = (
                orchestrator_state.get("instances") if isinstance(orchestrator_state, dict) else getattr(orchestrator_state, "instances", {})
            )
            # Normalize mapping (id -> info)
            if isinstance(instances_map, dict):
                for iid, info in instances_map.items():
                    inst = run.instances.get(iid)
                    if not inst:
                        # Create a placeholder if missing
                        inst = InstanceDisplay(instance_id=iid, strategy_name=getattr(info, "strategy_name", ""))
                        run.instances[iid] = inst
                    # Map state
                    try:
                        st = getattr(info, "state", None)
                        st_val = st.value if hasattr(st, "value") else (st if isinstance(st, str) else None)
                        if st_val:
                            from .models import InstanceStatus as TStatus
                            if st_val in ("queued", "running", "interrupted", "completed", "failed"):
                                inst.status = TStatus(st_val)
                    except Exception:
                        pass
                    # Times
                    try:
                        inst.started_at = getattr(info, "started_at", inst.started_at) or inst.started_at
                        inst.completed_at = getattr(info, "completed_at", inst.completed_at) or inst.completed_at
                    except Exception:
                        pass
        except Exception:
            pass

    def _update_header(self) -> None:
        """Update header zone."""
        try:
            run_src = self._render_run
            if not run_src:
                # No active run
                header_content = Text(
                    "Orchestrator TUI - No Active Run", style="bold yellow"
                )
            else:
                run = run_src
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
                        first = next(iter(list(run.strategies.values())))
                        model = first.config.get("model") if hasattr(first, "config") else None
                    if model:
                        header_content.append(" | ")
                        header_content.append(f"Model: {model}", style="magenta")
                except Exception:
                    pass
                header_content.append(" | ")
                # Show UI session start as authoritative global start (immediate)
                header_content.append(
                    f"Started: {self._format_time(self._ui_started_at)}", style="dim"
                )
                # Total runtime inline
                try:
                    st = self._ui_started_at
                    if st and st.tzinfo is None:
                        st = st.replace(tzinfo=timezone.utc)
                    if st:
                        now_dt = self._frame_now or datetime.now(timezone.utc)
                        elapsed = now_dt - st
                        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
                        minutes, seconds = divmod(rem, 60)
                        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        header_content.append(" | ")
                        header_content.append(f"Runtime: {duration}", style="dim")
                except Exception:
                    pass

                # Instance counts summary in header for quick glance
                try:
                    inst_list = list(run.instances.values())
                    total = len(inst_list)
                    running = sum(1 for i in inst_list if i.status == InstanceStatus.RUNNING)
                    completed = sum(1 for i in inst_list if i.status == InstanceStatus.COMPLETED)
                    failed = sum(1 for i in inst_list if i.status == InstanceStatus.FAILED)
                    interrupted = sum(1 for i in inst_list if i.status == InstanceStatus.INTERRUPTED)
                    header_content.append(" | ")
                    header_content.append(
                        f"Instances: {total} (R:{running} C:{completed} F:{failed} I:{interrupted})",
                        style="cyan",
                    )
                except Exception:
                    pass

            # Update layout with simple panel
            self._layout["header"].update(
                Panel(Align.center(header_content), style="blue", height=3)
            )
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            # Fallback on error
            self._layout["header"].update(Panel(f"Header Error: {e}", style="red"))

    def _update_dashboard(self) -> None:
        """Update dashboard zone."""
        try:
            run_src = self._render_run
            if not run_src:
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
            display_mode = run_src.get_display_mode()

            # Get dashboard content from adaptive display (synchronized to frame time)
            dashboard_content = self.adaptive_display.render_dashboard(run_src, display_mode, frame_now=getattr(self, "_frame_now", None))

            self._layout["body"].update(dashboard_content)
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            self._layout["body"].update(Panel(f"Dashboard Error: {e}", style="red"))

    def _update_footer(self) -> None:
        """Update footer zone with aggregate metrics."""
        try:
            # Build footer text as simple string
            footer_lines = []

            # Show stats as per specification
            run_src = self._render_run
            if run_src:
                run = run_src

                # Count instances by status
                inst_list = list(run.instances.values())
                active_count = sum(1 for i in inst_list if i.status == InstanceStatus.RUNNING)
                completed_count = sum(1 for i in inst_list if i.status == InstanceStatus.COMPLETED)
                failed_count = sum(1 for i in inst_list if i.status == InstanceStatus.FAILED)
                interrupted_count = sum(1 for i in inst_list if i.status == InstanceStatus.INTERRUPTED)

                # Global Duration: strictly from UI session start (immediate and independent)
                duration = "--:--:--"
                try:
                    start_dt = self._ui_started_at
                    if start_dt:
                        now_dt = self._frame_now or datetime.now(timezone.utc)
                        # Align tz
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        elapsed = now_dt - start_dt
                        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                except Exception:
                    pass

                # Get total cost
                total_cost = sum(i.cost for i in inst_list)

                # First line: instance counts and duration
                line1 = (
                    f"Instances: {len(run.instances)} | Active: {active_count} | Completed: {completed_count} | "
                    f"Failed: {failed_count} | Interrupted: {interrupted_count} | Duration: {duration}"
                )
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
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            # Log the error for debugging
            logger.error(f"Footer rendering error: {e}", exc_info=True)
            self._layout["footer"].update(Panel(f"Footer Error: {e}", style="red"))

    def _get_strategy_summary(self) -> str:
        """Get strategy summary for header."""
        if not self._render_run:
            return "-"

        # Prefer strategy names from strategies map
        strategy_names = set()
        try:
            for s in self._render_run.strategies.values():
                if s.strategy_name and s.strategy_name.lower() != "unknown":
                    strategy_names.add(s.strategy_name)
        except Exception:
            pass

        # Fallback: derive from instances if strategies map is empty
        if not strategy_names:
            try:
                for inst in self._render_run.instances.values():
                    if inst.strategy_name and inst.strategy_name.lower() != "unknown":
                        strategy_names.add(inst.strategy_name)
            except Exception:
                pass

        if not strategy_names:
            return "-"

        if len(strategy_names) == 1:
            name = next(iter(strategy_names))
            # Count strategy executions when available
            exec_count = len(self._render_run.strategies) or sum(
                1 for i in self._render_run.instances.values() if i.strategy_name == name
            )
            return f"{name} (x{exec_count})" if exec_count > 1 else name
        return f"Multiple ({len(strategy_names)})"

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

    def _snapshot_state_for_render(self) -> None:
        """Build an immutable snapshot of the current state for rendering.

        This avoids shared-state mutation during completion bursts causing stale frames.
        """
        try:
            src = self.state.current_run
            if not src:
                self._render_run = None
                return

            # Copy RunDisplay (shallow for simple fields)
            run = RunDisplay(
                run_id=src.run_id,
                prompt=src.prompt,
                repo_path=src.repo_path,
                base_branch=src.base_branch,
                strategies={},
                instances={},
                total_cost=src.total_cost,
                total_tokens=src.total_tokens,
                total_instances=src.total_instances,
                active_instances=src.active_instances,
                completed_instances=src.completed_instances,
                failed_instances=src.failed_instances,
                started_at=src.started_at,
                completed_at=src.completed_at,
                ui_started_at=self._ui_started_at,
                force_detail_level=(src.force_detail_level or self._force_display_mode_cli or self._force_display_mode_env),
            )

            # Copy strategies
            for sid, s in src.strategies.items():
                run.strategies[sid] = StrategyDisplay(
                    strategy_id=s.strategy_id,
                    strategy_name=s.strategy_name,
                    config=dict(s.config) if s.config else {},
                    instance_ids=list(s.instance_ids),
                    total_instances=s.total_instances,
                    completed_instances=s.completed_instances,
                    failed_instances=s.failed_instances,
                    started_at=s.started_at,
                    completed_at=s.completed_at,
                    is_complete=s.is_complete,
                )

            # Copy instances
            for iid, inst in src.instances.items():
                run.instances[iid] = InstanceDisplay(
                    instance_id=inst.instance_id,
                    strategy_name=inst.strategy_name,
                    status=inst.status,
                    branch_name=inst.branch_name,
                    prompt=inst.prompt,
                    model=inst.model,
                    current_activity=inst.current_activity,
                    last_tool_use=inst.last_tool_use,
                    duration_seconds=inst.duration_seconds,
                    cost=inst.cost,
                    total_tokens=inst.total_tokens,
                    input_tokens=inst.input_tokens,
                    output_tokens=inst.output_tokens,
                    started_at=inst.started_at,
                    completed_at=inst.completed_at,
                    last_updated=inst.last_updated,
                    error=inst.error,
                    error_type=inst.error_type,
                    metadata=dict(inst.metadata) if inst.metadata else {},
                )

            self._render_run = run
        except Exception as e:
            logger.debug(f"Snapshot build failed: {e}")
            # Fallback to direct reference (riskier but better than blank)
            self._render_run = self.state.current_run

    def set_forced_display_mode(self, mode: Optional[str]) -> None:
        """Set a CLI-forced display mode (detailed|compact|dense)."""
        if mode and mode in {"detailed", "compact", "dense"}:
            self._force_display_mode_cli = mode
        else:
            self._force_display_mode_cli = None
