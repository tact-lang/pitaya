"""
Event bus for component communication.

The event bus enables loose coupling between components by providing
a publish-subscribe mechanism for events. It uses an in-memory ring
buffer to prevent memory issues with long-running orchestrations.
"""

import asyncio
import json
import logging
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
            self._persist_file = open(
                self.persist_path, "a", buffering=1
            )  # Line buffered

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        if instance_id:
            event["instance_id"] = instance_id

        # Add to ring buffer first
        self.events.append(event)

        # Update statistics
        self.event_counts[event_type] += 1

        # Persist if configured (synchronous to ensure data integrity)
        if self._persist_file:
            event_json = json.dumps(event) + "\n"
            try:
                # Synchronous write to prevent data loss
                self._persist_file.write(event_json)
                self._persist_file.flush()
                # Add offset AFTER successful write
                event["offset"] = self._current_offset
                # Update offset for next event
                self._current_offset += len(event_json.encode("utf-8"))
            except (OSError, IOError) as e:
                logger.error(f"Failed to persist event: {e}")
                # Don't add offset if write failed

        # Notify subscribers
        self._notify_subscribers(event)

        logger.debug(f"Emitted event: {event_type}")

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

        # If offset specified, read from file
        if offset is not None and self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, "r") as f:
                    # Validate offset is at a line boundary
                    if offset > 0:
                        f.seek(offset - 1)
                        # Check if previous character is newline
                        prev_char = f.read(1)
                        if prev_char != "\n":
                            # Not at line boundary, scan forward to next line
                            while True:
                                char = f.read(1)
                                if not char or char == "\n":
                                    break
                            offset = f.tell()
                    else:
                        f.seek(0)

                    # Read lines from validated offset
                    current_offset = offset
                    for line in f:
                        if limit and len(events) >= limit:
                            break
                        try:
                            event = json.loads(line.strip())
                            # Apply filters
                            if event_types and event["type"] not in event_types:
                                continue
                            events.append(event)
                            current_offset = f.tell()
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping malformed event at offset {current_offset}"
                            )
                            continue

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
                        # Read new events from last offset
                        with open(file_path, "r") as f:
                            f.seek(last_offset)
                            for line in f:
                                try:
                                    event = json.loads(line.strip())
                                    # Filter by event type if specified
                                    if event_type and event.get("type") != event_type:
                                        continue
                                    # Call callback
                                    callback(event)
                                except json.JSONDecodeError:
                                    continue
                            # Update offset
                            last_offset = f.tell()

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
