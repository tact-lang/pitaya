"""Event bus for intra-orchestrator communication."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .event_persistence import EventPersistence
from .event_redaction import EventRedactor

logger = logging.getLogger(__name__)


class EventBus:
    """In-memory event bus with persistence and redaction."""

    def __init__(
        self,
        max_events: int = 10000,
        persist_path: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        self.max_events = max_events
        self.events: deque[Dict[str, Any]] = deque(maxlen=max_events)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.global_subscribers: List[Callable] = []
        self.event_counts: Dict[str, int] = defaultdict(int)

        self._redactor = EventRedactor()
        self._persistence = EventPersistence(
            persist_path=persist_path, run_id=run_id, redactor=self._redactor
        )
        if persist_path:
            self._persistence.open()

        self._file_watchers: List[asyncio.Task] = []

    # ------------------------------------------------------------------ #
    # Persistence lifecycle
    # ------------------------------------------------------------------ #
    @property
    def persist_path(self) -> Optional[Path]:
        return self._persistence.persist_path

    def reconfigure_persistence(
        self, persist_path: Path, run_id: Optional[str]
    ) -> None:
        """Point the event bus at a new persistence file."""
        self._persistence.close()
        self._persistence = EventPersistence(
            persist_path=persist_path, run_id=run_id, redactor=self._redactor
        )
        self._persistence.open()

    # ------------------------------------------------------------------ #
    # Emission
    # ------------------------------------------------------------------ #
    def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        instance_id: Optional[str] = None,
    ) -> None:
        event = {
            "type": event_type,
            "timestamp": self._utc_ts_ms_z(),
            "data": data,
        }
        if instance_id:
            event["instance_id"] = instance_id

        self.events.append(event)
        self.event_counts[event_type] += 1

        try:
            self._persistence.write_runner_event(event_type, event)
        except Exception:
            pass

        self._notify_subscribers(event)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                keys = (
                    ",".join(sorted(list(data.keys())))
                    if isinstance(data, dict)
                    else "-"
                )
                iid = instance_id or event.get("instance_id") or "-"
                logger.debug(
                    "Emitted event: %s (iid=%s, keys=%s)", event_type, iid, keys
                )
            except Exception:
                pass

    def emit_canonical(
        self,
        *,
        type: str,
        run_id: str,
        strategy_execution_id: Optional[str] = None,
        key: Optional[str] = None,
        payload: Dict[str, Any],
    ) -> None:
        ts = self._utc_ts_ms_z()
        record = {
            "id": str(uuid.uuid4()),
            "type": type,
            "ts": ts,
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            **({"key": key} if key else {}),
        }
        try:
            self._persistence.enqueue_canonical(record, payload)
        except Exception as exc:
            logger.error("Failed to enqueue canonical event: %s", exc)

        mirror = {
            "type": type,
            "ts": ts,
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            **({"key": key} if key else {}),
            "payload": self._redactor.sanitize(payload),
        }
        self.events.append(mirror)
        self._notify_subscribers(mirror)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "canonical.emit: type=%s key=%s sid=%s ts=%s",
                type,
                key or "-",
                strategy_execution_id or "-",
                ts,
            )

    # ------------------------------------------------------------------ #
    # Redaction
    # ------------------------------------------------------------------ #
    def set_custom_redaction_patterns(self, patterns: List[str]) -> None:
        self._redactor.set_custom_patterns(patterns)

    def _sanitize(self, obj: Any) -> Any:
        """Back-compat shim for callers expecting _sanitize on the bus."""
        return self._redactor.sanitize(obj)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def get_events_since(
        self,
        timestamp: Optional[datetime] = None,
        offset: Optional[int] = None,
        event_types: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        # Read from disk when offset/timestamp provided and persistence exists
        if (
            timestamp is not None or offset is not None
        ) and self._persistence.persist_path:
            events, next_offset = self._persistence.read_events_since(
                timestamp=timestamp, offset=offset, event_types=event_types, limit=limit
            )
            return events, next_offset

        events: List[Dict[str, Any]] = []
        for event in self.events:
            if limit and len(events) >= limit:
                break
            if timestamp and not self._event_after(event, timestamp):
                continue
            if event_types and event["type"] not in event_types:
                continue
            events.append(event)
        return events, self._persistence.current_offset

    # ------------------------------------------------------------------ #
    # Subscription
    # ------------------------------------------------------------------ #
    def subscribe(
        self,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> Callable[[], None]:
        if event_type is None:
            self.global_subscribers.append(callback)

            def unsubscribe_memory() -> None:
                self.global_subscribers.remove(callback)

        else:
            self.subscribers[event_type].append(callback)

            def unsubscribe_memory() -> None:
                self.subscribers[event_type].remove(callback)

        unsubscribe_file = None
        if self._persistence.persist_path and self._persistence.persist_path.exists():
            unsubscribe_file = self.subscribe_to_file(
                self._persistence.persist_path,
                event_type,
                callback,
                poll_interval=0.1,
            )

        def unsubscribe() -> None:
            unsubscribe_memory()
            if unsubscribe_file:
                unsubscribe_file()

        logger.debug("Subscribed to %s", event_type or "all events")
        return unsubscribe

    def subscribe_to_file(
        self,
        file_path: Path,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
        poll_interval: float = 1.0,
    ) -> Callable[[], None]:
        async def file_watcher() -> None:
            last_offset = 0
            while True:
                try:
                    if file_path.exists():
                        with open(file_path, "rb") as fh:
                            fh.seek(last_offset)
                            while True:
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
                                    event = json.loads(
                                        line.decode("utf-8", errors="ignore").strip()
                                    )
                                except json.JSONDecodeError:
                                    last_offset += len(line)
                                    continue
                                if not event_type or event.get("type") == event_type:
                                    try:
                                        callback(event)
                                    except Exception as cb_err:
                                        logger.error(
                                            "Error in file watcher callback: %s", cb_err
                                        )
                                last_offset += len(line)
                    await asyncio.sleep(poll_interval)
                except asyncio.CancelledError:
                    break
                except (OSError, json.JSONDecodeError) as exc:
                    logger.error("Error in file watcher: %s", exc)
                    await asyncio.sleep(poll_interval)

        task = asyncio.create_task(file_watcher())
        self._file_watchers.append(task)

        def unsubscribe() -> None:
            task.cancel()
            self._file_watchers.remove(task)

        return unsubscribe

    # ------------------------------------------------------------------ #
    # Flush / close
    # ------------------------------------------------------------------ #
    def flush_pending(self) -> None:
        self._persistence.flush_pending()

    def close(self) -> None:
        for task in self._file_watchers:
            task.cancel()
        self._file_watchers.clear()
        self._persistence.close()

    def __enter__(self) -> "EventBus":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _notify_subscribers(self, event: Dict[str, Any]) -> None:
        event_type = event["type"]
        for callback in self.subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except (AttributeError, TypeError, ValueError) as exc:
                logger.error("Error in event callback for %s: %s", event_type, exc)

        for callback in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except (AttributeError, TypeError, ValueError) as exc:
                logger.error("Error in global event callback: %s", exc)

    @staticmethod
    def _utc_ts_ms_z() -> str:
        dt = datetime.now(timezone.utc)
        try:
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except TypeError:
            s = dt.isoformat()
            if s.endswith("+00:00"):
                s = s[:-6] + "Z"
            return s

    @staticmethod
    def _event_after(event: Dict[str, Any], ts: datetime) -> bool:
        try:
            ts_s = event.get("ts") or event.get("timestamp")
            event_time = (
                datetime.fromisoformat(str(ts_s).replace("Z", "+00:00"))
                if ts_s
                else None
            )
            if not event_time:
                return True
            target = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return event_time > target
        except Exception:
            return True
