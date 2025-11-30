"""Helper mixin for AsyncEventStream file operations."""

from __future__ import annotations

import asyncio

from .logging_config import logger


class StreamFileOps:
    """File reading and rotation detection helpers for event streams."""

    def _detect_rotation_or_truncation(self) -> None:
        """Detect file rotation or truncation and adjust read offsets."""
        try:
            st = self._events_file.stat()
            inode = getattr(st, "st_ino", None)
            size = st.st_size
            if self._prev_inode is None:
                self._prev_inode = inode
                self._prev_size = size
            else:
                rotated = inode is not None and inode != self._prev_inode
                truncated = size < self._last_position
                if rotated or truncated:
                    logger.warning(
                        f"Events file {'rotated' if rotated else 'truncated'}; resetting read position"
                    )
                    self._last_position = size
                    self._prev_inode = inode
                    self._prev_size = size
                    if rotated:
                        self._rotations += 1
                    if truncated:
                        self._truncations += 1
        except Exception as e:
            logger.debug(f"Stat failed for events file: {e}")

    async def _read_new_lines_from_file(self) -> None:
        """Read new lines from the events file and enqueue them."""
        try:
            with open(self._events_file, "rb") as f:
                f.seek(self._last_position)

                lines_read = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", errors="ignore").strip()
                    if s:
                        lines_read += 1
                        logger.debug(f"Read event line: {s[:100]}...")
                        await self._event_queue.put(s)
                        self._lines_enqueued_total += 1
                        await asyncio.sleep(0)

                new_position = f.tell()
                if new_position > self._last_position and lines_read > 0:
                    logger.info(
                        f"Read {lines_read} lines, position {self._last_position} -> {new_position}"
                    )
                self._last_position = new_position
                try:
                    self._bytes_read_total += max(0, new_position - self._last_position)
                except Exception:
                    pass

                self.event_processor.state.last_event_start_offset = self._last_position
                try:
                    st2 = self._events_file.stat()
                    self._prev_size = st2.st_size
                except Exception:
                    pass
        except (OSError, IOError) as e:
            logger.error(f"Error reading events file: {e}")


__all__ = ["StreamFileOps"]
