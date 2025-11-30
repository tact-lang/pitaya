"""Async event stream reader with file watching and polling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from .file_watcher import EventFileWatcher, WATCHDOG_AVAILABLE
from .logging_config import logger
from .stream_file_ops import StreamFileOps

if TYPE_CHECKING:
    from watchdog.observers import Observer  # type: ignore
else:
    try:
        from watchdog.observers import Observer  # type: ignore
    except Exception:
        Observer = None  # type: ignore


class AsyncEventStream(StreamFileOps):
    """Async event stream reader with file watching."""

    def __init__(self, event_processor):
        """
        Initialize async event stream.

        Args:
            event_processor: Event processor to handle events
        """
        self.event_processor = event_processor
        self._observer: Optional[Observer] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # File change tracking
        self._prev_inode: Optional[int] = None
        self._prev_size: int = 0
        # Diagnostics
        self._lines_enqueued_total: int = 0
        self._lines_processed_total: int = 0
        self._rotations: int = 0
        self._truncations: int = 0
        self._bytes_read_total: int = 0

    async def start(self, events_file: Path, from_offset: int = 0) -> None:
        """
        Start watching events file.

        Args:
            events_file: Path to events.jsonl
            from_offset: Byte offset to start from
        """
        self._events_file = events_file
        self._last_position = from_offset
        # Capture the running loop for thread-safe callbacks
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        logger.info(
            f"Starting event stream for file: {events_file}, offset: {from_offset}"
        )

        # Start event processing loop
        asyncio.create_task(self._process_events())

        # Read any existing events first if file exists
        if events_file.exists():
            await self._read_existing_events(events_file, from_offset)

        # Start file watcher for real-time updates
        self._start_file_watcher(events_file)

        # Also start polling loop as a backup; sharing the same offset with the watcher avoids duplicates
        asyncio.create_task(self._polling_loop())

    async def stop(self) -> None:
        """Stop watching events."""
        self._shutdown = True

        if self._observer:
            self._observer.stop()
            self._observer.join()

    async def _read_existing_events(self, events_file: Path, from_offset: int) -> None:
        """Read existing events from file."""
        if not events_file.exists():
            logger.warning(f"Events file does not exist: {events_file}")
            return

        logger.info(
            f"Reading existing events from {events_file}, starting at offset {from_offset}"
        )
        try:
            with open(events_file, "rb") as f:
                # Seek to offset
                if from_offset > 0:
                    f.seek(from_offset - 1)
                    prev = f.read(1)
                    if prev != b"\n":
                        # advance to next newline to align
                        while True:
                            b = f.read(1)
                            if not b or b == b"\n":
                                break
                # finally set start position
                f.seek(max(0, from_offset))

                # Read all existing lines
                lines_read = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", errors="ignore").strip()
                    if s:
                        lines_read += 1
                        logger.debug(f"Reading existing event: {s[:100]}...")
                        await self._event_queue.put(s)
                        self._lines_enqueued_total += 1

                # Update position for polling to continue from here
                self._last_position = f.tell()
                self.event_processor.state.last_event_start_offset = self._last_position
                logger.info(
                    f"Read {lines_read} existing events, new position: {self._last_position}"
                )
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading existing events: {e}")

    def _start_file_watcher(self, events_file: Path) -> None:
        """Start watching events file for changes."""

        if not WATCHDOG_AVAILABLE or Observer is None:
            logger.info(
                "Watchdog not available; skipping file watcher and using polling only"
            )
            return

        def new_line_callback(line: str):
            # Put line in queue for async processing (thread-safe)
            try:
                loop = self._loop or asyncio.get_event_loop()
                loop.call_soon_threadsafe(self._event_queue.put_nowait, line)
                self._lines_enqueued_total += 1
            except Exception as e:
                logger.error(f"File watcher enqueue failed: {e}")

        # Pass shared position getters/setters so watcher and poller use one offset
        event_handler = EventFileWatcher(
            events_file,
            new_line_callback,
            self._last_position,
            get_position=lambda: self._last_position,
            set_position=lambda pos: setattr(self, "_last_position", pos),
        )

        try:
            self._observer = Observer()  # type: ignore[call-arg]
            self._observer.schedule(
                event_handler, str(events_file.parent), recursive=False
            )  # type: ignore[union-attr]
            self._observer.start()  # type: ignore[union-attr]
            logger.info(
                f"Started file watcher for {events_file} at position {self._last_position}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to start watchdog observer, falling back to polling: {e}"
            )
            self._observer = None

    async def _polling_loop(self) -> None:
        """Poll the events file for new content."""
        logger.info(f"Starting polling loop for file: {self._events_file}")

        # Wait for file to exist first
        while not self._shutdown and not self._events_file.exists():
            logger.debug(f"Waiting for events file to exist: {self._events_file}")
            await asyncio.sleep(0.1)

        logger.info(
            f"Events file exists, starting to poll from position {self._last_position}"
        )

        while not self._shutdown:
            try:
                if self._events_file.exists():
                    self._detect_rotation_or_truncation()
                    await self._read_new_lines_from_file()

                # Poll every 100ms
                await asyncio.sleep(0.1)

            except (OSError, IOError) as e:
                logger.error(f"Error polling events file: {e}")
                await asyncio.sleep(1.0)

    async def _process_events(self) -> None:
        """Process events from queue."""
        logger.info("Starting event processing loop")
        while not self._shutdown:
            try:
                # Get event with timeout to check shutdown
                import time

                t_wait0 = time.perf_counter()
                line = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                q_after_get = self._event_queue.qsize()

                # Parse and process event
                try:
                    t_p0 = time.perf_counter()
                    event = json.loads(line)
                    t_p1 = time.perf_counter()
                    e_type = event.get("type", "unknown")
                    self.event_processor.process_event(event)
                    t_h1 = time.perf_counter()
                    self._lines_processed_total += 1
                    logger.debug(
                        f"queue_event type={e_type} qsize={q_after_get} parse_ms={(t_p1 - t_p0) * 1000:.2f} handle_ms={(t_h1 - t_p1) * 1000:.2f} wait_ms={(t_p0 - t_wait0) * 1000:.2f}"
                    )

                    # Yield control to allow UI to update
                    await asyncio.sleep(0)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in event: {e}, line: {line}")
                    self.event_processor.state.add_error(f"JSON error: {e}")
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    logger.error(f"Error processing event: {e}")
                    self.event_processor.state.add_error(f"Event error: {e}")

            except asyncio.TimeoutError:
                # This is normal - just checking for shutdown
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of stream diagnostics."""
        try:
            qsize = self._event_queue.qsize()
        except Exception:
            qsize = -1
        return {
            "queue_size": qsize,
            "last_position": getattr(self, "_last_position", 0),
            "prev_size": self._prev_size,
            "rotations": self._rotations,
            "truncations": self._truncations,
            "lines_enqueued": self._lines_enqueued_total,
            "lines_processed": self._lines_processed_total,
            "bytes_read": self._bytes_read_total,
        }


__all__ = ["AsyncEventStream"]
