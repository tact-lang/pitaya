"""Small formatting helpers shared across display mixins."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


class FormattingMixin:
    """Duration/time formatting helpers."""

    def _format_time(self, dt: Optional[datetime]) -> str:
        """Format datetime for display."""
        if not dt:
            return "N/A"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%H:%M:%S")

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        hours = seconds / 3600
        return f"{hours:.1f}h"


__all__ = ["FormattingMixin"]
