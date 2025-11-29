"""Timeout-specific cleanup helpers."""

from __future__ import annotations

class TimeoutCleanupMixin:
    """Provides timeout cleanup that preserves workspace for inspection."""

    async def _cleanup_timeout(self) -> None:
        # Stop heartbeat and clean container but keep workspace for debugging.
        if getattr(self, "container_manager", None):
            await self.container_manager.stop_and_cleanup(remove_home_volume=False)
        # Intentionally do not delete workspace_dir so operators can inspect/resume.
        return
