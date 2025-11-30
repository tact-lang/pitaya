"""Base helpers shared by adaptive dashboard renderers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict

from rich.console import RenderableType

from ..models import RunDisplay
from ..event_processing.logging_config import logger


class AdaptiveBase:
    """Provides shared helpers and dispatch for adaptive renderers."""

    def __init__(self) -> None:
        self._color_scheme = "default"  # default|accessible
        self._mode_renderers: Dict[
            str, Callable[[RunDisplay, datetime | None], RenderableType]
        ] = {}

    def _aware(self, dt: datetime | None) -> datetime:
        """Return a timezone-aware datetime (UTC) for safe comparisons/sorts."""
        if dt is None:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        hours = seconds / 3600
        return f"{hours:.1f}h"

    def set_color_scheme(self, scheme: str) -> None:
        s = (scheme or "default").strip().lower()
        if s not in ("default", "accessible"):
            s = "default"
        self._color_scheme = s

    def render_dashboard(
        self, run: RunDisplay, display_mode: str, frame_now: datetime | None = None
    ) -> RenderableType:
        """Render dashboard based on display mode."""
        renderer = self._mode_renderers.get(
            display_mode, self._mode_renderers.get("compact")
        )
        try:
            logger.debug(
                "render_dashboard mode=%s strategies=%s instances=%s",
                display_mode,
                len(run.strategies),
                len(run.instances),
            )
        except Exception:
            pass
        if renderer is None:
            raise ValueError(f"No renderer registered for mode {display_mode}")
        return renderer(run, frame_now=frame_now)


__all__ = ["AdaptiveBase"]
