"""Cross-platform repository lock with bounded wait and diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import IO, Any, Optional

from . import GitError
from .git_common import read_lock_holder

logger = logging.getLogger(__name__)

DEFAULT_REPO_LOCK_WAIT = 300.0
LOCK_LOG_PERIOD = 2.0


class RepoLock:
    """Async context manager that serializes git imports per repository."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: Optional[dict[str, Any]] = None,
        max_wait_seconds: float = DEFAULT_REPO_LOCK_WAIT,
    ) -> None:
        self._path = path
        self._metadata = metadata or {}
        self._max_wait = max_wait_seconds
        self._fh: Optional[IO[str]] = None

    async def __aenter__(self) -> "RepoLock":
        self._fh = await asyncio.to_thread(self._path.open, "a+")
        await self._acquire()
        await self._write_metadata()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._release()

    async def _acquire(self) -> None:
        start = time.monotonic()
        last_log = start
        if sys.platform.startswith("win"):
            await self._acquire_windows(start, last_log)
        else:
            await self._acquire_posix(start, last_log)

    async def _acquire_posix(self, start: float, last_log: float) -> None:
        import fcntl

        assert self._fh is not None
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                last_log = await self._wait_or_fail(start, last_log)

    async def _acquire_windows(self, start: float, last_log: float) -> None:
        import msvcrt

        assert self._fh is not None
        while True:
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except Exception:
                last_log = await self._wait_or_fail(start, last_log)

    async def _wait_or_fail(self, start: float, last_log: float) -> float:
        waited = time.monotonic() - start
        if waited >= self._max_wait:
            raise GitError(
                f"Timeout waiting for repo import lock after {waited:.1f}s "
                f"(holder={read_lock_holder(self._path)})"
            )

        now = time.monotonic()
        if now - last_log >= LOCK_LOG_PERIOD:
            logger.info(
                "waiting_for_repo_lock path=%s held_by=%s waited_s=%.1f",
                self._path,
                read_lock_holder(self._path),
                waited,
            )
            last_log = now
        await asyncio.sleep(0.2)
        return last_log

    async def _write_metadata(self) -> None:
        if not self._fh:
            return
        try:
            payload = {
                "pid": os.getpid(),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **self._metadata,
            }
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.write(json.dumps(payload))
            self._fh.flush()
        except Exception:
            logger.debug("Failed to write lock metadata for %s", self._path)

    async def _release(self) -> None:
        if not self._fh:
            return

        try:
            if sys.platform.startswith("win"):
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            logger.debug("Failed to release repo lock %s", self._path)

        try:
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.flush()
        except Exception:
            logger.debug("Failed to truncate repo lock %s", self._path)

        try:
            self._fh.close()
        except Exception:
            logger.debug("Failed to close repo lock %s", self._path)
