"""
Core TUI display wiring: composes mixins for layout, rendering, snapshotting,
and loops. Heavy rendering logic lives in display_components to keep this
module small and readable.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live

from .adaptive import AdaptiveDisplay
from .event_handler import AsyncEventStream, EventProcessor
from .models import RunDisplay, TUIState
from .display_components import (
    BodyMixin,
    FooterMixin,
    FormattingMixin,
    HeaderMixin,
    LayoutMixin,
    RenderLoopMixin,
    SnapshotMixin,
)


class TUIDisplay(
    RenderLoopMixin,
    BodyMixin,
    HeaderMixin,
    FooterMixin,
    SnapshotMixin,
    FormattingMixin,
    LayoutMixin,
):
    """Main TUI display using Rich."""

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_rate: float = 0.1,
        state_poll_interval: float = 3.0,
    ):
        self._logger = logging.getLogger(__name__)
        self.console = self._ensure_console(console)
        self.refresh_rate = refresh_rate
        self.state_poll_interval = state_poll_interval

        self.state = TUIState()
        self.event_processor = EventProcessor(self.state)
        self.event_stream = AsyncEventStream(self.event_processor)
        self.adaptive_display = AdaptiveDisplay()

        self._force_display_mode_env = None
        self._alt_screen = True
        self._force_display_mode_cli: Optional[str] = None
        self._details_mode: str = "none"
        self._body_split: bool = False

        self._live: Optional[Live] = None
        self._shutdown = False
        self._orchestrator = None
        self._render_run: Optional[RunDisplay] = None

        self._ui_started_at = datetime.now(timezone.utc)
        self._last_render_events: int = -1
        self._last_runtime_tick: int = -1
        self._last_updated_iid: Optional[str] = None
        self._frame_now: Optional[datetime] = None
        self._ids_full: bool = False

        self._layout = self._create_layout()

    def _ensure_console(self, console: Optional[Console]) -> Console:
        if console is not None:
            try:
                if not getattr(console, "is_terminal", True):
                    return Console(force_terminal=True)
                return console
            except Exception:
                return Console(force_terminal=True)
        return Console(force_terminal=True)

    def set_forced_display_mode(self, mode: Optional[str]) -> None:
        if mode and mode in {"detailed", "compact", "dense"}:
            self._force_display_mode_cli = mode

    def set_details_mode(self, mode: str) -> None:
        m = (mode or "none").strip().lower()
        self._details_mode = m if m in ("none", "right") else "none"

    def set_ids_full(self, full: bool) -> None:
        self._ids_full = bool(full)

    def set_details_messages(self, n: int) -> None:
        try:
            if hasattr(self.event_processor, "set_details_messages"):
                self.event_processor.set_details_messages(n)
        except Exception:
            pass

    async def run(self, orchestrator, events_file: Path, from_offset: int = 0) -> None:
        self._logger.info("TUI starting with events file: %s", events_file)
        self._orchestrator = orchestrator

        try:
            if self._orchestrator and hasattr(self._orchestrator, "get_current_state"):
                st = self._orchestrator.get_current_state()
                if st:
                    self._reconcile_state(st)
        except Exception:
            pass

        await self.event_stream.start(events_file, from_offset)
        asyncio.create_task(self._state_polling_loop())

        with Live(
            self._layout,
            console=self.console,
            refresh_per_second=10,
            screen=self._alt_screen,
            vertical_overflow="crop",
            transient=self._alt_screen,
            redirect_stdout=True,
            redirect_stderr=False,
        ) as live:
            self._live = live
            live.update(self._layout)
            await self._render_loop()

    async def stop(self) -> None:
        self._logger.info("TUI stop() called")
        self._shutdown = True
        await self.event_stream.stop()
        if self._live:
            self._live.stop()


__all__ = ["TUIDisplay"]
