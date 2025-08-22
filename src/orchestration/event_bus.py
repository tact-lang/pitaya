"""
Event bus for component communication.

The event bus enables loose coupling between components by providing
a publish-subscribe mechanism for events. It uses an in-memory ring
buffer to prevent memory issues with long-running orchestrations.
"""

import asyncio
import json
import logging
import os
import socket
import uuid
import re as _re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TextIO, Pattern


logger = logging.getLogger(__name__)


# Two-stage redaction patterns
_SENSITIVE_KEYS = (
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "cookie",
)
_PATTERNS = [
    _re.compile(r"(?i)(authorization\s*:\s*Bearer)\s+[A-Za-z0-9._\-]+"),
    _re.compile(r"sk-[A-Za-z0-9]{16,}"),
    _re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),  # ghp_, gho_, ghs_, gh u
    _re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT-like
    _re.compile(r"(?i)(api[_-]?key|token)\s*[:=]\s*[A-Za-z0-9\-]{8,}"),
    _re.compile(r"(?i)Basic\s+[A-Za-z0-9+/=]{20,}"),
]


class EventBus:
    """
    In-memory event bus with ring buffer and persistence.

    Events flow unidirectionally upward:
    - Instance Runner emits fine-grained events
    - Orchestration adds strategy-level events
    - TUI consumes events for display
    """

    def __init__(
        self,
        max_events: int = 10000,
        persist_path: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize event bus.

        Args:
            max_events: Maximum events to keep in memory (ring buffer size)
            persist_path: Optional path to persist events to disk
        """
        self.max_events = max_events
        self.persist_path = persist_path

        # Ring buffer for events
        self.events: deque[Dict[str, Any]] = deque(maxlen=max_events)

        # Subscribers by event type
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Global subscribers (receive all events)
        self.global_subscribers: List[Callable] = []

        # Event statistics
        self.event_counts: Dict[str, int] = defaultdict(int)

        # Persistence file handle and offset tracking
        self._persist_file: Optional[TextIO] = None
        # Separate file for runner-level events (instance.*), per spec
        self._runner_file: Optional[TextIO] = None
        # Buffered writer state for runner.jsonl to avoid event-loop blocking
        self._runner_queue: deque[bytes] = deque()
        self._runner_writer_task: Optional[asyncio.Task] = None
        # Minimal defaults (no env overrides)
        self._runner_flush_interval_ms = 10
        self._runner_flush_max_batch = 512
        self._current_offset = 0
        self._lock_file: Optional[TextIO] = None
        self._run_id = run_id
        if persist_path:
            self._open_persist_file()

        # File watching support
        self._file_watchers: List[asyncio.Task] = []

        # Flush policy (fixed sensible defaults)
        self._flush_policy = "interval"
        self._flush_interval_ms = 50
        self._flush_max_batch = 256
        self._pending_canonical: List[Tuple[Dict[str, Any], Dict[str, Any]]] = (
            []
        )  # (envelope record without start_offset/payload, payload)
        self._pending_lock = asyncio.Lock()
        self._flusher_task: Optional[asyncio.Task] = None
        # Custom redaction patterns (compiled regex)
        self._custom_redaction_patterns: List[Pattern] = []

    def _open_persist_file(self) -> None:
        """Open persistence file for appending events."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            # Check if file exists to get current offset
            if self.persist_path.exists():
                self._current_offset = self.persist_path.stat().st_size
            # Acquire single-writer lock for duration of run
            try:
                lock_path = self.persist_path.with_suffix(
                    self.persist_path.suffix + ".lock"
                )
                self._lock_file = open(lock_path, "a+")
                # Cross-platform strict lock
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write lock metadata for diagnostics (best-effort)
                try:
                    meta = {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "started_at_iso": self._utc_ts_ms_z(),
                        "run_id": self._run_id,
                    }
                    self._lock_file.seek(0)
                    self._lock_file.truncate(0)
                    self._lock_file.write(json.dumps(meta))
                    self._lock_file.flush()
                    os.fsync(self._lock_file.fileno())
                except Exception:
                    pass
            except Exception as e:
                # Strict enforcement: fail fast if lock cannot be acquired
                raise RuntimeError(f"Failed to acquire events.jsonl lock: {e}")

            # Open file in binary append mode to track byte offsets precisely
            self._persist_file = open(self.persist_path, "ab", buffering=0)
            # Open runner.jsonl alongside for instance.* events (best-effort)
            try:
                runner_path = self.persist_path.parent / "runner.jsonl"
                self._runner_file = open(runner_path, "ab", buffering=0)
            except Exception:
                self._runner_file = None
            # Start flusher for interval policy
            if self._flush_policy == "interval":
                try:
                    self._flusher_task = asyncio.create_task(self._flush_loop())
                except RuntimeError:
                    # No running loop yet; will start when emit_canonical is called within a loop
                    self._flusher_task = None

    def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        instance_id: Optional[str] = None,
    ) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event_type: Type of event (e.g., "instance.started")
            data: Event data
            instance_id: Optional instance ID for correlation
        """
        # Create event with metadata
        event = {
            "type": event_type,
            "timestamp": self._utc_ts_ms_z(),
            "data": data,
        }

        if instance_id:
            event["instance_id"] = instance_id

        # Add to ring buffer first
        self.events.append(event)

        # Update statistics
        self.event_counts[event_type] += 1

        # Only persist canonical events to events.jsonl; runner-level events go to runner.jsonl
        try:
            if (
                self._runner_file
                and isinstance(event_type, str)
                and (
                    event_type.startswith("instance.")
                    or event_type.startswith("runner.")
                )
            ):
                line = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
                # For critical runner completion records, write synchronously to ensure durability before cleanup
                if event_type == "runner.instance.completed":
                    try:
                        self._runner_file.write(line)
                        self._runner_file.flush()
                        os.fsync(self._runner_file.fileno())
                    except Exception as e:
                        logger.debug(f"runner.completed sync write error: {e}")
                else:
                    # Enqueue for background flushing to avoid blocking the event loop
                    self._runner_queue.append(line)
                    if not self._runner_writer_task or self._runner_writer_task.done():
                        try:
                            self._runner_writer_task = asyncio.create_task(
                                self._runner_flush_loop()
                            )
                        except RuntimeError:
                            # If no running loop yet, write synchronously as fallback
                            try:
                                self._runner_file.write(line)
                                self._runner_file.flush()
                            except Exception:
                                pass
        except Exception:
            pass

        # Notify subscribers
        self._notify_subscribers(event)

        # Minimal debug logging only when logger configured for DEBUG externally
        if logger.isEnabledFor(logging.DEBUG):
            try:
                keys = (
                    ",".join(sorted(list(data.keys())))
                    if isinstance(data, dict)
                    else "-"
                )
                iid = instance_id or event.get("instance_id") or "-"
                logger.debug(f"Emitted event: {event_type} (iid={iid}, keys={keys})")
            except Exception:
                pass

    async def _runner_flush_loop(self) -> None:
        """Background flush loop for runner.jsonl writes.

        Drains a small batch every few milliseconds and writes it off-thread
        in a single call to minimize event-loop blocking.
        """
        try:
            while self._runner_file is not None:
                # If nothing to write, sleep briefly
                if not self._runner_queue:
                    await asyncio.sleep(self._runner_flush_interval_ms / 1000.0)
                    # Double-check before continuing
                    if not self._runner_queue:
                        continue
                # Drain a batch
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
                    # Perform write + flush off-thread
                    await asyncio.to_thread(self._runner_file.write, blob)
                    await asyncio.to_thread(self._runner_file.flush)
                except Exception as e:
                    logger.debug(f"runner.flush error: {e}")
        except asyncio.CancelledError:
            return

    def _utc_ts_ms_z(self) -> str:
        """Return UTC ISO-8601 timestamp with milliseconds and trailing Z."""
        dt = datetime.now(timezone.utc)
        # Python 3.11+: timespec='milliseconds'
        try:
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except TypeError:
            s = dt.isoformat()
            if s.endswith("+00:00"):
                s = s[:-6] + "Z"
            return s

    def emit_canonical(
        self,
        *,
        type: str,
        run_id: str,
        strategy_execution_id: Optional[str] = None,
        key: Optional[str] = None,
        payload: Dict[str, Any],
    ) -> None:
        """Emit a canonical event line into events.jsonl with strict envelope.

        Fields: {id, type, ts, run_id, strategy_execution_id, key?, start_offset, payload}
        """
        ts = self._utc_ts_ms_z()
        record = {
            "id": str(uuid.uuid4()),
            "type": type,
            "ts": ts,
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            **({"key": key} if key else {}),
            # start_offset added after write
        }
        # Queue for flush according to policy
        if self._persist_file:
            try:
                if self._flush_policy == "per_event":
                    # Write immediately
                    self._write_canonical_immediate(record, payload)
                else:
                    # Interval policy: queue and ensure flusher running
                    async def _enqueue():
                        async with self._pending_lock:
                            self._pending_canonical.append((record, payload))

                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(_enqueue())
                    except RuntimeError:
                        # No running loop; fall back to immediate write
                        self._write_canonical_immediate(record, payload)
                    # Start flusher if not running
                    if not self._flusher_task or self._flusher_task.done():
                        try:
                            self._flusher_task = asyncio.create_task(self._flush_loop())
                        except RuntimeError:
                            pass
            except Exception as e:
                logger.error(f"Failed to enqueue canonical event: {e}")
        # Also push an in-memory mirror for UI in canonical form
        mirror = {
            "type": type,
            "ts": ts,
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            **({"key": key} if key else {}),
            "payload": self._sanitize(payload),
        }
        self.events.append(mirror)
        self._notify_subscribers(mirror)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"canonical.emit: type={type} key={key or '-'} sid={strategy_execution_id or '-'} ts={ts}"
            )

    def _sanitize(self, obj: Any) -> Any:
        """Redact secrets recursively: field-name redaction + pattern sweep."""
        try:
            if isinstance(obj, dict):
                out: Dict[str, Any] = {}
                for k, v in obj.items():
                    kl = str(k).lower()
                    if any(s in kl for s in _SENSITIVE_KEYS):
                        # Field-name redaction: value replaced regardless of type
                        out[k] = "[REDACTED]"
                    else:
                        out[k] = self._sanitize(v)
                return out
            if isinstance(obj, list):
                return [self._sanitize(v) for v in obj]
            if isinstance(obj, str):
                s = obj
                # Pattern sweep across strings (built-in + custom)
                for pat in list(_PATTERNS) + list(self._custom_redaction_patterns):
                    try:
                        if pat.pattern.lower().startswith("(?i)(authorization"):
                            # Preserve header name; redact token tail
                            s = pat.sub(r"\1 [REDACTED]", s)
                        else:
                            s = pat.sub("[REDACTED]", s)
                    except Exception:
                        continue
                return s
            return obj
        except Exception:
            return obj

    def set_custom_redaction_patterns(self, patterns: List[str]) -> None:
        """Set additional regex patterns to redact in event payloads/logs."""
        out: List[Pattern] = []
        for p in patterns or []:
            try:
                out.append(_re.compile(p))
            except Exception:
                continue
        self._custom_redaction_patterns = out

    def _write_canonical_immediate(
        self, record: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        try:
            start_offset = self._current_offset
            record_with_payload = {
                **record,
                "start_offset": start_offset,
                "payload": self._sanitize(payload),
            }
            line = (
                json.dumps(record_with_payload, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._persist_file.write(line)
            # Per-event durability
            self._persist_file.flush()
            os.fsync(self._persist_file.fileno())
            self._current_offset += len(line)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"canonical.persist: type={record.get('type')} key={record.get('key','-')} start={start_offset} bytes={len(line)}"
                )
        except Exception as e:
            logger.error(f"Failed to persist canonical event: {e}")

    def flush_pending(self) -> None:
        """Synchronously flush any pending canonical events (best-effort).

        Ensures events are written and fsynced to disk even when using interval policy.
        Safe to call during shutdown or immediately after critical emissions.
        """
        try:
            if not self._persist_file:
                return
            # Drain pending list without awaiting the async lock (best-effort)
            pending = []
            try:
                if self._pending_canonical:
                    pending = list(self._pending_canonical)
                    self._pending_canonical.clear()
            except Exception:
                pending = []
            for record, payload in pending:
                self._write_canonical_immediate(record, payload)
            # Final fsync barrier
            try:
                self._persist_file.flush()
                os.fsync(self._persist_file.fileno())
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"flush_pending error: {e}")

    async def _flush_loop(self) -> None:
        """Periodic flusher for interval policy.

        Avoids blocking the event loop with per-event fsync by batching writes
        and offloading the actual file I/O to a background thread.
        """
        try:
            while self._persist_file and self._flush_policy == "interval":
                await asyncio.sleep(self._flush_interval_ms / 1000.0)
                try:
                    # Drain a batch under lock, then write it out off-thread
                    async with self._pending_lock:
                        if not self._pending_canonical:
                            continue
                        batch = self._pending_canonical[: self._flush_max_batch]
                        del self._pending_canonical[: self._flush_max_batch]
                    # Perform a single batched write + fsync off-thread
                    await asyncio.to_thread(self._write_canonical_batch, batch)
                except Exception as e:
                    logger.error(f"Error in event flusher: {e}")
        except asyncio.CancelledError:
            return

    def _write_canonical_batch(self, batch: list[tuple[dict, dict]]) -> None:
        """Write a batch of canonical events in one I/O operation.

        Each entry in batch is (record_without_payload, payload). This method
        computes start_offset per line, appends payload, writes all lines at
        once, flushes, fsyncs once, and advances the current offset.
        """
        if not self._persist_file or not batch:
            return
        try:
            base = self._current_offset
            lines: list[bytes] = []
            offset = base
            for record, payload in batch:
                # Compute start_offset for this record
                record_with_payload = {
                    **record,
                    "start_offset": offset,
                    "payload": self._sanitize(payload),
                }
                line = (
                    json.dumps(record_with_payload, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                lines.append(line)
                offset += len(line)
            # Write all lines at once
            self._persist_file.write(b"".join(lines))
            # Single flush/fsync barrier for the entire batch
            self._persist_file.flush()
            os.fsync(self._persist_file.fileno())
            # Advance current offset
            self._current_offset = offset
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"canonical.persist.batch: count={len(lines)} start={base} bytes={sum(len(line) for line in lines)}"
                )
        except Exception as e:
            logger.error(f"Failed to persist canonical batch: {e}")

    def _notify_subscribers(self, event: Dict[str, Any]) -> None:
        """Notify all relevant subscribers of an event."""
        event_type = event["type"]

        # Notify type-specific subscribers
        for callback in self.subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Error in event callback for {event_type}: {e}")

        # Notify global subscribers
        for callback in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Error in global event callback: {e}")

    def subscribe(
        self,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> Callable[[], None]:
        """
        Subscribe to events.

        This sets up file watching on events.jsonl for real-time updates
        when persist_path is configured.

        Args:
            event_type: Event type to subscribe to (None for all events)
            callback: Function to call when event occurs

        Returns:
            Unsubscribe function
        """
        # Add to in-memory subscribers
        if event_type is None:
            self.global_subscribers.append(callback)

            def unsubscribe_memory():
                self.global_subscribers.remove(callback)

        else:
            self.subscribers[event_type].append(callback)

            def unsubscribe_memory():
                self.subscribers[event_type].remove(callback)

        # If we have a persist file, also set up file watching
        unsubscribe_file = None
        if self.persist_path and self.persist_path.exists():
            # Use subscribe_to_file for file watching
            unsubscribe_file = self.subscribe_to_file(
                self.persist_path,
                event_type,
                callback,
                poll_interval=0.1,  # Fast polling for responsiveness
            )

        # Combined unsubscribe function
        def unsubscribe():
            unsubscribe_memory()
            if unsubscribe_file:
                unsubscribe_file()

        logger.debug(f"Subscribed to {event_type or 'all events'}")
        return unsubscribe

    def get_events_since(
        self,
        timestamp: Optional[datetime] = None,
        offset: Optional[int] = None,
        event_types: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get events since a given timestamp or offset.

        Args:
            timestamp: Only return events after this time (None for all)
            offset: Only return events after this byte offset
            event_types: Filter to specific event types (None for all)
            limit: Maximum number of events to return

        Returns:
            List of events matching criteria
        """
        events: List[Dict[str, Any]] = []

        # If offset specified, read from file (in binary mode to track byte offsets)
        if (
            (offset is not None or timestamp is not None)
            and self.persist_path
            and self.persist_path.exists()
        ):
            try:
                # Open in binary mode to ensure tell/seek operate on byte offsets
                with open(self.persist_path, "rb") as f:
                    # Derive starting offset from timestamp if provided
                    current_offset = 0
                    if timestamp is not None:
                        # Scan to find first record with ts >= timestamp
                        # Note: this is O(N); acceptable for v1
                        from datetime import datetime as _dt

                        try:
                            # Ensure tz-aware parse; if naive, treat as UTC
                            ts_target = (
                                timestamp
                                if timestamp.tzinfo
                                else timestamp.replace(tzinfo=timezone.utc)
                            )
                        except Exception:
                            ts_target = timestamp
                        f.seek(0)
                        while True:
                            pos = f.tell()
                            line = f.readline()
                            if not line:
                                break
                            try:
                                rec = json.loads(line.decode("utf-8").strip())
                                ts_str = rec.get("ts")
                                if ts_str:
                                    try:
                                        t = _dt.fromisoformat(
                                            ts_str.replace("Z", "+00:00")
                                        )
                                        if t >= ts_target:
                                            current_offset = rec.get(
                                                "start_offset", pos
                                            )
                                            break
                                    except Exception:
                                        pass
                            except json.JSONDecodeError:
                                pass
                        # If not found, set to EOF
                        if current_offset == 0:
                            f.seek(0, os.SEEK_END)
                            return [], f.tell()
                    else:
                        current_offset = offset or 0
                    # Validate offset is at a line boundary
                    if current_offset > 0:
                        f.seek(current_offset - 1)
                        prev_byte = f.read(1)
                        if prev_byte != b"\n":
                            while True:
                                b = f.read(1)
                                if not b or b == b"\n":
                                    break
                            current_offset = f.tell()
                    else:
                        f.seek(0)

                    # Read lines from validated offset
                    while True:
                        if limit and len(events) >= limit:
                            break
                        pos_before = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            try:
                                f.seek(pos_before)
                            except Exception:
                                pass
                            break
                        try:
                            event = json.loads(line.decode("utf-8").strip())
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping malformed event at offset {current_offset}"
                            )
                            # Advance offset by the line length even if malformed
                            current_offset += len(line)
                            continue

                        # Apply filters
                        if not event_types or event.get("type") in event_types:
                            events.append(event)
                        # Update offset by bytes read
                        current_offset += len(line)

                    # Return events and the next offset
                    return events, current_offset
            except (OSError, IOError) as e:
                logger.error(f"Error reading events from file: {e}")

        # Otherwise use in-memory buffer
        for event in self.events:
            if limit and len(events) >= limit:
                break

            # Check timestamp
            if timestamp:
                try:
                    ts_s = event.get("ts") or event.get("timestamp")
                    event_time = (
                        datetime.fromisoformat(str(ts_s).replace("Z", "+00:00"))
                        if ts_s
                        else None
                    )
                    if event_time and event_time <= timestamp:
                        continue
                except Exception:
                    pass

            # Offset filtering applies to file-backed events; in-memory mirror has no byte offsets

            # Check event type
            if event_types and event["type"] not in event_types:
                continue

            events.append(event)

        # For in-memory events, return current offset as next
        return events, self._current_offset

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary with event counts and other stats
        """
        return {
            "total_events": sum(self.event_counts.values()),
            "events_in_buffer": len(self.events),
            "event_counts": dict(self.event_counts),
            "subscriber_count": sum(len(subs) for subs in self.subscribers.values()),
            "global_subscriber_count": len(self.global_subscribers),
        }

    def subscribe_to_file(
        self,
        file_path: Path,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
        poll_interval: float = 1.0,
    ) -> Callable[[], None]:
        """
        Subscribe to events from a file with periodic polling.

        This enables external processes to consume events by watching
        the event log file for new entries.

        Args:
            file_path: Path to event file to watch
            event_type: Optional event type filter
            callback: Function to call for each new event
            poll_interval: How often to check for new events (seconds)

        Returns:
            Unsubscribe function
        """

        async def file_watcher():
            last_offset = 0

            while True:
                try:
                    # Check if file exists
                    if file_path.exists():
                        # Read new events from last offset (binary mode for byte-accurate offsets)
                        with open(file_path, "rb") as f:
                            f.seek(last_offset)
                            while True:
                                pos_before = f.tell()
                                line = f.readline()
                                if not line:
                                    break
                                # Enforce newline boundary per spec: if partial line (no trailing \n), rewind and wait
                                if not line.endswith(b"\n"):
                                    try:
                                        f.seek(pos_before)
                                    except Exception:
                                        pass
                                    break
                                try:
                                    s = line.decode("utf-8", errors="ignore").strip()
                                    if not s:
                                        last_offset += len(line)
                                        continue
                                    event = json.loads(s)
                                except json.JSONDecodeError:
                                    # Skip malformed lines but still advance offset
                                    last_offset += len(line)
                                    continue

                                # Filter by event type if specified
                                if not event_type or event.get("type") == event_type:
                                    try:
                                        callback(event)
                                    except Exception as cb_err:
                                        logger.error(
                                            f"Error in file watcher callback: {cb_err}"
                                        )
                                # Advance offset by bytes read for this line
                                last_offset += len(line)

                    # Sleep before next check
                    await asyncio.sleep(poll_interval)

                except asyncio.CancelledError:
                    break
                except (OSError, json.JSONDecodeError) as e:
                    logger.error(f"Error in file watcher: {e}")
                    await asyncio.sleep(poll_interval)

        # Create watcher task
        task = asyncio.create_task(file_watcher())
        self._file_watchers.append(task)

        def unsubscribe():
            task.cancel()
            self._file_watchers.remove(task)

        return unsubscribe

    def close(self) -> None:
        """Close persistence file if open."""
        # Cancel file watchers
        for task in self._file_watchers:
            task.cancel()
        self._file_watchers.clear()

        # Flush any pending events (best-effort)
        try:
            if self._flusher_task and not self._flusher_task.done():
                self._flusher_task.cancel()
        except Exception:
            pass
        # Stop runner writer and drain queue
        try:
            if self._runner_writer_task and not self._runner_writer_task.done():
                self._runner_writer_task.cancel()
        except Exception:
            pass
        try:
            if self._runner_file and self._runner_queue:
                # Drain synchronously
                try:
                    blob = b"".join(self._runner_queue)
                    self._runner_queue.clear()
                    self._runner_file.write(blob)
                    self._runner_file.flush()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # Synchronously flush pending if file open
            if self._persist_file:
                # Drain pending queue
                try:
                    pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                    try:
                        # best-effort no-await acquire
                        if self._pending_canonical:
                            pending = list(self._pending_canonical)
                            self._pending_canonical.clear()
                    except Exception:
                        pass
                    for record, payload in pending:
                        self._write_canonical_immediate(record, payload)
                except Exception:
                    pass
        finally:
            pass

        if self._persist_file:
            self._persist_file.close()
            self._persist_file = None
        # Close runner file
        try:
            if self._runner_file:
                self._runner_file.close()
        except Exception:
            pass
        self._runner_file = None
        # Release lock file
        try:
            if self._lock_file:
                self._lock_file.close()
        except Exception:
            pass
        self._lock_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
