"""Filesystem watcher for events.jsonl updates."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .logging_config import logger

# Watchdog is optional; fall back to pure polling when unavailable
WATCHDOG_AVAILABLE = True
try:  # pragma: no cover - import-time environment dependent
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
except Exception:  # ImportError or other
    WATCHDOG_AVAILABLE = False

    class FileSystemEventHandler:  # type: ignore
        """Stub used when watchdog is unavailable."""

        pass

    class FileModifiedEvent:  # type: ignore
        """Stub used when watchdog is unavailable."""

        pass


class EventFileWatcher(FileSystemEventHandler):
    """Watches events.jsonl file for changes."""

    def __init__(
        self,
        file_path: Path,
        callback: Callable[[str], None],
        initial_position: int = 0,
        *,
        get_position: Optional[Callable[[], int]] = None,
        set_position: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize file watcher.

        Args:
            file_path: Path to events.jsonl
            callback: Function to call with new lines
            initial_position: Starting position in file
        """
        self.file_path = file_path
        self.callback = callback
        # Share position with owner if provided to avoid duplicate reads
        self._get_position = get_position
        self._set_position = set_position
        self._last_position = initial_position

    def on_modified(self, event):
        """Handle file modification."""
        if not isinstance(event, FileModifiedEvent):
            return
        try:
            if Path(event.src_path).resolve() != self.file_path.resolve():
                return
        except Exception:
            return

        try:
            with open(self.file_path, "rb") as f:
                # Seek to last position
                last_pos = (
                    self._get_position() if self._get_position else self._last_position
                )
                f.seek(last_pos)

                # Read new lines
                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", errors="ignore").strip()
                    if s:
                        self.callback(s)

                # Update shared position
                new_pos = f.tell()
                if self._set_position:
                    self._set_position(new_pos)
                else:
                    self._last_position = new_pos
        except (OSError, IOError) as e:
            logger.error(f"Error reading events file: {e}")


__all__ = [
    "EventFileWatcher",
    "WATCHDOG_AVAILABLE",
    "FileSystemEventHandler",
    "FileModifiedEvent",
]
