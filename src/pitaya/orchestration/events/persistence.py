"""Persistence and flushing utilities for EventBus."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TextIO

from pitaya.orchestration.events.redaction import EventRedactor

logger = logging.getLogger(__name__)


class EventPersistence:
    """Handle canonical/runner event persistence and flushing."""

    def __init__(
        self,
        *,
        persist_path: Optional[Path],
        run_id: Optional[str],
        redactor: EventRedactor,
        flush_policy: str = "interval",
        flush_interval_ms: int = 50,
        flush_max_batch: int = 256,
        runner_flush_interval_ms: int = 10,
        runner_flush_max_batch: int = 512,
    ) -> None:
        self.persist_path = persist_path
        self.run_id = run_id
        self.redactor = redactor

        self._persist_file: Optional[TextIO] = None
        self._runner_file: Optional[TextIO] = None
        self._lock_file: Optional[TextIO] = None
        self._current_offset = 0

        self._runner_queue: deque[bytes] = deque()
        self._runner_writer_task: Optional[asyncio.Task] = None
        self._runner_flush_interval_ms = runner_flush_interval_ms
        self._runner_flush_max_batch = runner_flush_max_batch

        self._flush_policy = flush_policy
        self._flush_interval_ms = flush_interval_ms
        self._flush_max_batch = flush_max_batch
        self._pending_canonical: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        self._pending_lock = asyncio.Lock()
        self._flusher_task: Optional[asyncio.Task] = None

    @property
    def current_offset(self) -> int:
        return self._current_offset

    def open(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        if self.persist_path.exists():
            self._current_offset = self.persist_path.stat().st_size
        self._lock_file = self._acquire_lock()
        self._persist_file = open(self.persist_path, "ab", buffering=0)
        try:
            runner_path = self.persist_path.parent / "runner.jsonl"
            self._runner_file = open(runner_path, "ab", buffering=0)
        except Exception:
            self._runner_file = None

        if self._flush_policy == "interval":
            try:
                self._flusher_task = asyncio.create_task(self._flush_loop())
            except RuntimeError:
                self._flusher_task = None

    def write_runner_event(self, event_type: str, event: Dict[str, Any]) -> None:
        if not self._runner_file:
            return
        if not isinstance(event_type, str):
            return
        if not (event_type.startswith("instance.") or event_type.startswith("runner.")):
            return

        line = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        if event_type == "runner.instance.completed":
            try:
                self._runner_file.write(line)
                self._runner_file.flush()
                os.fsync(self._runner_file.fileno())
            except Exception as exc:
                logger.debug("runner.completed sync write error: %s", exc)
            return

        self._runner_queue.append(line)
        if not self._runner_writer_task or self._runner_writer_task.done():
            try:
                self._runner_writer_task = asyncio.create_task(
                    self._runner_flush_loop()
                )
            except RuntimeError:
                try:
                    self._runner_file.write(line)
                    self._runner_file.flush()
                except Exception:
                    pass

    def enqueue_canonical(
        self, record: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        if not self._persist_file:
            return
        if self._flush_policy == "per_event":
            self._write_canonical_immediate(record, payload)
            return

        async def _enqueue() -> None:
            async with self._pending_lock:
                self._pending_canonical.append((record, payload))

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_enqueue())
        except RuntimeError:
            self._write_canonical_immediate(record, payload)

        if not self._flusher_task or self._flusher_task.done():
            try:
                self._flusher_task = asyncio.create_task(self._flush_loop())
            except RuntimeError:
                pass

    def flush_pending(self) -> None:
        if not self._persist_file:
            return
        pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        try:
            if self._pending_canonical:
                pending = list(self._pending_canonical)
                self._pending_canonical.clear()
        except Exception:
            pending = []
        for record, payload in pending:
            self._write_canonical_immediate(record, payload)
        try:
            self._persist_file.flush()
            os.fsync(self._persist_file.fileno())
        except Exception:
            pass

    def close(self) -> None:
        for task in (self._flusher_task, self._runner_writer_task):
            try:
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass

        if self._runner_file and self._runner_queue:
            try:
                blob = b"".join(self._runner_queue)
                self._runner_queue.clear()
                self._runner_file.write(blob)
                self._runner_file.flush()
            except Exception:
                pass

        if self._persist_file:
            try:
                pending = list(self._pending_canonical)
                self._pending_canonical.clear()
                for record, payload in pending:
                    self._write_canonical_immediate(record, payload)
            except Exception:
                pass
            try:
                self._persist_file.close()
            except Exception:
                pass
            self._persist_file = None

        if self._runner_file:
            try:
                self._runner_file.close()
            except Exception:
                pass
            self._runner_file = None

        if self._lock_file:
            try:
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

    def read_events_since(
        self,
        *,
        timestamp: Optional[datetime] = None,
        offset: Optional[int] = None,
        event_types: Optional[set[str]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not (self.persist_path and self.persist_path.exists()):
            return [], self._current_offset
        events: List[Dict[str, Any]] = []
        try:
            with open(self.persist_path, "rb") as fh:
                current_offset = self._offset_from_timestamp(fh, timestamp, offset)
                fh.seek(current_offset)
                while True:
                    if limit and len(events) >= limit:
                        break
                    pos_before = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        try:
                            fh.seek(pos_before)
                        except Exception:
                            pass
                        break
                    try:
                        event = json.loads(line.decode("utf-8").strip())
                    except json.JSONDecodeError:
                        current_offset += len(line)
                        continue
                    if not event_types or event.get("type") in event_types:
                        events.append(event)
                    current_offset += len(line)
                return events, current_offset
        except (OSError, IOError) as exc:
            logger.error("Error reading events from file: %s", exc)
            return events, self._current_offset

    # Internal helpers -------------------------------------------------

    def _write_canonical_immediate(
        self, record: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        try:
            start_offset = self._current_offset
            record_with_payload = {
                **record,
                "start_offset": start_offset,
                "payload": self.redactor.sanitize(payload),
            }
            line = (
                json.dumps(record_with_payload, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._persist_file.write(line)
            self._persist_file.flush()
            os.fsync(self._persist_file.fileno())
            self._current_offset += len(line)
        except Exception as exc:
            logger.error("Failed to persist canonical event: %s", exc)

    def _write_canonical_batch(self, batch: list[tuple[dict, dict]]) -> None:
        if not self._persist_file or not batch:
            return
        try:
            base = self._current_offset
            lines: list[bytes] = []
            offset = base
            for record, payload in batch:
                record_with_payload = {
                    **record,
                    "start_offset": offset,
                    "payload": self.redactor.sanitize(payload),
                }
                line = (
                    json.dumps(record_with_payload, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                lines.append(line)
                offset += len(line)
            self._persist_file.write(b"".join(lines))
            self._persist_file.flush()
            os.fsync(self._persist_file.fileno())
            self._current_offset = offset
        except Exception as exc:
            logger.error("Failed to persist canonical batch: %s", exc)

    def _acquire_lock(self) -> Optional[TextIO]:
        if not self.persist_path:
            return None
        lock_path = self.persist_path.with_suffix(self.persist_path.suffix + ".lock")
        try:
            handle = open(lock_path, "a+")
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                meta = {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at_iso": self._utc_ts_ms_z(),
                    "run_id": self.run_id,
                }
                handle.seek(0)
                handle.truncate(0)
                handle.write(json.dumps(meta))
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                pass
            return handle
        except Exception as exc:
            raise RuntimeError(f"Failed to acquire events.jsonl lock: {exc}") from exc

    def _utc_ts_ms_z(self) -> str:
        dt = datetime.now(timezone.utc)
        try:
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except TypeError:
            s = dt.isoformat()
            if s.endswith("+00:00"):
                s = s[:-6] + "Z"
            return s

    def _offset_from_timestamp(
        self, fh: Any, timestamp: Optional[datetime], offset: Optional[int]
    ) -> int:
        if timestamp is None and offset:
            return offset
        if timestamp is None:
            return 0
        try:
            import datetime as _dt

            target = (
                timestamp
                if timestamp.tzinfo
                else timestamp.replace(tzinfo=timezone.utc)
            )
            fh.seek(0)
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                try:
                    rec = json.loads(line.decode("utf-8").strip())
                    ts_str = rec.get("ts")
                    if ts_str:
                        t = _dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if t >= target:
                            return rec.get("start_offset", pos)
                except Exception:
                    continue
            fh.seek(0, os.SEEK_END)
            return fh.tell()
        except Exception:
            return offset or 0

    async def _flush_loop(self) -> None:
        try:
            while self._persist_file and self._flush_policy == "interval":
                await asyncio.sleep(self._flush_interval_ms / 1000.0)
                try:
                    async with self._pending_lock:
                        if not self._pending_canonical:
                            continue
                        batch = self._pending_canonical[: self._flush_max_batch]
                        del self._pending_canonical[: self._flush_max_batch]
                    await asyncio.to_thread(self._write_canonical_batch, batch)
                except Exception as exc:
                    logger.error("Error in event flusher: %s", exc)
        except asyncio.CancelledError:
            return

    async def _runner_flush_loop(self) -> None:
        try:
            while self._runner_file is not None:
                if not self._runner_queue:
                    await asyncio.sleep(self._runner_flush_interval_ms / 1000.0)
                    if not self._runner_queue:
                        continue
                batch: list[bytes] = []
                for _ in range(
                    min(self._runner_flush_max_batch, len(self._runner_queue))
                ):
                    try:
                        batch.append(self._runner_queue.popleft())
                    except IndexError:
                        break
                if not batch:
                    continue
                blob = b"".join(batch)
                try:
                    await asyncio.to_thread(self._runner_file.write, blob)
                    await asyncio.to_thread(self._runner_file.flush)
                except Exception as exc:
                    logger.debug("runner.flush error: %s", exc)
        except asyncio.CancelledError:
            return
