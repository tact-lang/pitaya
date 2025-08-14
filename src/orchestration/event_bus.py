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
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TextIO


logger = logging.getLogger(__name__)


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
        self._current_offset = 0
        self._lock_file: Optional[TextIO] = None
        self._run_id = run_id
        if persist_path:
            self._open_persist_file()

        # File watching support
        self._file_watchers: List[asyncio.Task] = []

    def _open_persist_file(self) -> None:
        """Open persistence file for appending events."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            # Check if file exists to get current offset
            if self.persist_path.exists():
                self._current_offset = self.persist_path.stat().st_size
            # Acquire single-writer lock for duration of run
            try:
                lock_path = self.persist_path.with_suffix(self.persist_path.suffix + ".lock")
                self._lock_file = open(lock_path, "a+")
                # Cross-platform best-effort lock
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except Exception:
                    # If lock cannot be acquired, proceed but log
                    logger.warning("Could not acquire events.jsonl lock; proceeding best-effort")
            except Exception as e:
                logger.warning(f"Failed to open lock file: {e}")

            # Open file in binary append mode to track byte offsets precisely
            self._persist_file = open(self.persist_path, "ab", buffering=0)

    def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        instance_id: Optional[str] = None,
        *,
        persist: bool = True,
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        if instance_id:
            event["instance_id"] = instance_id

        # Add to ring buffer first
        self.events.append(event)

        # Update statistics
        self.event_counts[event_type] += 1

        # Persist legacy event if configured (for backward-compatible consumers)
        if persist and self._persist_file:
            line = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
            try:
                start_offset = self._current_offset
                self._persist_file.write(line)
                self._persist_file.flush()
                os.fsync(self._persist_file.fileno())
                self._current_offset += len(line)
                event["offset"] = start_offset
            except (OSError, IOError) as e:
                logger.error(f"Failed to persist event: {e}")

        # Notify subscribers
        self._notify_subscribers(event)

        # Enrich debug logging with brief context to aid troubleshooting
        try:
            keys = ",".join(sorted(list(data.keys()))) if isinstance(data, dict) else "-"
            iid = instance_id or event.get("instance_id") or "-"
            logger.debug(f"Emitted event: {event_type} (iid={iid}, keys={keys})")
        except Exception:
            # Fallback to original single-line message if anything goes wrong
            logger.debug(f"Emitted event: {event_type}")

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
        ts = datetime.now(timezone.utc).isoformat()
        record = {
            "id": str(uuid.uuid4()),
            "type": type,
            "ts": ts,
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            **({"key": key} if key else {}),
            # start_offset added after write
        }
        if self._persist_file:
            try:
                start_offset = self._current_offset
                record_with_payload = {**record, "start_offset": start_offset, "payload": payload}
                line = (json.dumps(record_with_payload, separators=(",", ":")) + "\n").encode("utf-8")
                self._persist_file.write(line)
                self._persist_file.flush()
                os.fsync(self._persist_file.fileno())
                self._current_offset += len(line)
            except Exception as e:
                logger.error(f"Failed to persist canonical event: {e}")
        # Also push an in-memory mirror for UI (map to legacy fields)
        mirror = {
            "type": type,
            "timestamp": ts,
            "data": payload,
        }
        self.events.append(mirror)
        self._notify_subscribers(mirror)

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
        if offset is not None and self.persist_path and self.persist_path.exists():
            try:
                # Open in binary mode to ensure tell/seek operate on byte offsets
                with open(self.persist_path, "rb") as f:
                    # Validate offset is at a line boundary
                    if offset > 0:
                        f.seek(offset - 1)
                        # Check if previous byte is newline
                        prev_byte = f.read(1)
                        if prev_byte != b"\n":
                            # Not at line boundary, scan forward to next newline
                            while True:
                                b = f.read(1)
                                if not b or b == b"\n":
                                    break
                            offset = f.tell()
                    else:
                        f.seek(0)

                    # Read lines from validated offset
                    current_offset = offset
                    while True:
                        if limit and len(events) >= limit:
                            break
                        line = f.readline()
                        if not line:
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
                event_time = datetime.fromisoformat(event["timestamp"])
                if event_time <= timestamp:
                    continue

            # Check offset
            if offset is not None and event.get("offset", 0) <= offset:
                continue

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
                                line = f.readline()
                                if not line:
                                    break
                                try:
                                    event = json.loads(line.decode("utf-8").strip())
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

        if self._persist_file:
            self._persist_file.close()
            self._persist_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
