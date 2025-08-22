"""
Event handler for processing Pitaya events.

Transforms raw events from the event stream into TUI state updates.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
import asyncio
from .models import (
    TUIState,
    RunDisplay,
    StrategyDisplay,
    InstanceDisplay,
    InstanceStatus,
)

# Watchdog is optional; fall back to pure polling when unavailable
WATCHDOG_AVAILABLE = True
try:  # pragma: no cover - import-time environment dependent
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
except Exception:  # ImportError or other
    WATCHDOG_AVAILABLE = False

    class FileSystemEventHandler:  # type: ignore
        pass

    class FileModifiedEvent:  # type: ignore
        pass


if TYPE_CHECKING:
    from watchdog.observers import Observer  # type: ignore
else:
    try:
        from watchdog.observers import Observer  # type: ignore
    except Exception:
        Observer = None  # type: ignore


logger = logging.getLogger(__name__)


class EventProcessor:
    """Processes Pitaya events and updates TUI state."""

    def __init__(self, state: TUIState):
        """
        Initialize event processor.

        Args:
            state: TUI state to update
        """
        self.state = state
        self._event_handlers = self._setup_handlers()
        # Last-N public messages per instance for details pane
        from collections import defaultdict, deque

        self._messages = defaultdict(lambda: deque(maxlen=10))
        self._max_messages = 10

    def set_details_messages(self, n: int) -> None:
        try:
            n = max(1, int(n))
        except Exception:
            n = 10
        # Rebuild deques with new maxlen
        from collections import deque

        new = {}
        for iid, dq in self._messages.items():
            nd = deque(dq, maxlen=n)
            new[iid] = nd
        self._messages = new
        self._max_messages = n

    def _append_msg(self, iid: str, text: str) -> None:
        try:
            if not iid:
                return
            dq = self._messages.get(iid)
            if dq is None:
                from collections import deque

                dq = deque(maxlen=self._max_messages)
                self._messages[iid] = dq
            self._messages[iid].append(text)
        except Exception:
            pass

    def _setup_handlers(self) -> Dict[str, Callable]:
        """Setup event type to handler mapping."""
        return {
            # Run-level events
            "run.started": self._handle_run_started,
            "run.completed": self._handle_run_completed,
            "run.failed": self._handle_run_failed,
            # Strategy-level events
            "strategy.started": self._handle_strategy_started,
            "strategy.completed": self._handle_strategy_completed,
            "strategy.failed": self._handle_strategy_failed,
            # State snapshots
            "state.instance_registered": self._handle_state_instance_registered,
            # Canonical task events (map to minimal UI updates)
            "task.scheduled": self._handle_task_scheduled,
            "task.started": self._handle_task_started,
            "task.progress": self._handle_task_progress,
            "task.completed": self._handle_task_completed,
            "task.failed": self._handle_task_failed,
            "task.interrupted": self._handle_task_interrupted,
            # Instance-level events
            "instance.queued": self._handle_instance_queued,
            "instance.started": self._handle_instance_started,
            "instance.completed": self._handle_instance_completed,
            "instance.failed": self._handle_instance_failed,
            "instance.progress": self._handle_instance_progress,
            # Instance phase events
            "instance.workspace_preparing": self._handle_instance_workspace_preparing,
            "instance.container_creating": self._handle_instance_container_creating,
            "instance.container_env_preparing": self._handle_instance_container_env_preparing,
            "instance.container_env_prepared": self._handle_instance_container_env_prepared,
            "instance.container_create_call": self._handle_instance_container_creating,
            "instance.container_create_entry": self._handle_instance_container_creating,
            "instance.container_image_check": self._handle_instance_container_creating,
            "instance.container_config_ready": self._handle_instance_container_creating,
            "instance.container_create_attempt": self._handle_instance_container_creating,
            "instance.container_created": self._handle_instance_container_created,
            # Agent start
            "instance.agent_starting": self._handle_instance_agent_starting,
            "instance.result_collection_started": self._handle_instance_result_collection,
            # Agent stream events
            "instance.agent_system": self._handle_agent_system,
            "instance.agent_assistant": self._handle_agent_assistant,
            "instance.agent_tool_use": self._handle_agent_tool_use,
            "instance.agent_tool_result": self._handle_agent_tool_result,
            "instance.agent_result": self._handle_agent_result,
            # Cancellation / lifecycle
            "instance.canceled": self._handle_instance_canceled,
            "state.instance_updated": self._handle_state_instance_updated,
        }

    def process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single event.

        Args:
            event: Event dictionary with type, timestamp, data
        """
        # Normalize canonical events with 'payload' envelope into the internal shape
        event_type = event.get("type")
        # Canonical normalization: copy envelope fields + payload to a stable shape
        if "payload" in event and isinstance(event["payload"], dict):
            payload = event["payload"]
            norm: Dict[str, Any] = {
                "type": event_type,
                "timestamp": event.get("ts") or event.get("timestamp"),
                "data": payload,
            }
            # carry important envelope fields
            for k in ("run_id", "strategy_execution_id", "key"):
                if event.get(k) is not None:
                    norm[k] = event[k]
            if payload.get("instance_id"):
                norm["instance_id"] = payload.get("instance_id")
            event = norm
            event_type = norm.get("type")
        if not event_type:
            return

        # Update event tracking
        self.state.events_processed += 1

        # Initialize current run if missing using envelope run_id when available
        try:
            if not self.state.current_run and event.get("run_id"):
                self.state.current_run = RunDisplay(run_id=str(event.get("run_id")))
        except Exception:
            pass

        # Debug logging - more detailed for instance events
        if event_type.startswith("instance."):
            logger.info(
                f"Processing {event_type} for instance {event.get('instance_id', 'None')}"
            )

        # Get handler for event type
        handler = self._event_handlers.get(event_type)
        if handler:
            try:
                import time

                t0 = time.perf_counter()
                handler(event)
                t1 = time.perf_counter()
                iid = event.get("instance_id") or event.get("data", {}).get(
                    "instance_id"
                )
                logger.debug(
                    f"event_processed type={event_type} iid={iid or '-'} dur_ms={(t1 - t0)*1000:.2f}"
                )
            except (AttributeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"Error processing event {event_type}: {e}")
                self.state.add_error(f"Event processing error: {e}")
        else:
            # Accept canonical strategy/task events as pass-through for now
            logger.debug(f"No handler for event type: {event_type}")

    # Canonical task.* helpers for offline mode (maps minimal state)
    def _ensure_current_run(self) -> None:
        if not self.state.current_run:
            # Create a placeholder RunDisplay so task/strategy events can attach
            self.state.current_run = RunDisplay(
                run_id="unknown",
                prompt="",
                repo_path="",
                base_branch="main",
            )

    def _handle_task_scheduled(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        # Derive strategy from envelope when available
        sid = event.get("strategy_execution_id")
        strategy_name = "unknown"
        if sid and self.state.current_run and sid in self.state.current_run.strategies:
            strategy_name = self.state.current_run.strategies[sid].strategy_name
        inst = self.state.current_run.instances.get(iid)
        if not inst:
            inst = InstanceDisplay(
                instance_id=iid,
                strategy_name=strategy_name,
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name", ""),
            )
            self.state.current_run.instances[iid] = inst
        # Set model early if present to avoid showing default placeholder
        try:
            m = data.get("model")
            if m:
                inst.model = m
        except Exception:
            pass
        # Keep total_instances in sync with unique instances only
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        # Best-effort grouping by strategy name when available
        if sid and self.state.current_run and sid in self.state.current_run.strategies:
            strat = self.state.current_run.strategies[sid]
            if iid not in strat.instance_ids:
                strat.instance_ids.append(iid)
                strat.total_instances += 1
        # Fallback: prime run started_at on first canonical event if not set
        try:
            if self.state.current_run and not self.state.current_run.started_at:
                ts = event.get("timestamp")
                self.state.current_run.started_at = self._parse_timestamp(ts)
        except Exception:
            pass

    def _handle_task_started(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.RUNNING
            # Set start time from canonical envelope timestamp
            try:
                ts = event.get("timestamp") or event.get("ts")
                inst.started_at = self._parse_timestamp(ts)
            except Exception:
                pass
            # Ensure global run started_at is set ASAP
            try:
                if self.state.current_run and not self.state.current_run.started_at:
                    self.state.current_run.started_at = (
                        inst.started_at or self._parse_timestamp(event.get("timestamp"))
                    )
            except Exception:
                pass
            # Backfill strategy name when available
            sid = event.get("strategy_execution_id")
            if (
                sid
                and self.state.current_run
                and sid in self.state.current_run.strategies
            ):
                sname = self.state.current_run.strategies[sid].strategy_name
                if sname and (
                    not inst.strategy_name or inst.strategy_name == "unknown"
                ):
                    inst.strategy_name = sname
                strat = self.state.current_run.strategies[sid]
                if iid not in strat.instance_ids:
                    strat.instance_ids.append(iid)
                    strat.total_instances += 1
            # Append public message
            model = (event.get("data", {}) or {}).get("model")
            base = (event.get("data", {}) or {}).get("base_branch")
            parts = ["started"]
            if model:
                parts.append(f"model={model}")
            if base:
                parts.append(f"base={base}")
            self._append_msg(iid, " ".join(parts))
        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _handle_task_completed(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.COMPLETED
            art = data.get("artifact", {})
            if art.get("branch_final"):
                inst.branch_name = art.get("branch_final")
            # Persist completion time and duration if start known
            try:
                ts = event.get("timestamp") or event.get("ts")
                inst.completed_at = self._parse_timestamp(ts) or inst.completed_at
                if inst.started_at and inst.completed_at:
                    # Calculate in seconds (monotonic best-effort)
                    start = inst.started_at
                    end = inst.completed_at
                    try:
                        if end.tzinfo != start.tzinfo:
                            end = end.replace(tzinfo=start.tzinfo)
                    except Exception:
                        pass
                    inst.duration_seconds = max(0.0, (end - start).total_seconds())
            except Exception:
                pass
            # Capture metrics from canonical payload
            metrics = data.get("metrics", {})
            if metrics:
                try:
                    tc = metrics.get("total_cost")
                    inst.cost = float(tc) if isinstance(tc, (int, float)) else inst.cost
                except Exception:
                    pass
                try:
                    tt = metrics.get("total_tokens")
                    if isinstance(tt, (int, float)):
                        inst.total_tokens = int(tt)
                except Exception:
                    pass
                # Optional in/out tokens
                try:
                    it = metrics.get("input_tokens")
                    ot = metrics.get("output_tokens")
                    if isinstance(it, (int, float)):
                        inst.input_tokens = int(it)
                    if isinstance(ot, (int, float)):
                        inst.output_tokens = int(ot)
                except Exception:
                    pass
            # Capture final message info
            try:
                fm = data.get("final_message")
                if isinstance(fm, str):
                    inst.final_message = fm
                inst.final_message_truncated = bool(data.get("final_message_truncated"))
                fmp = data.get("final_message_path")
                if isinstance(fmp, str) and fmp:
                    inst.final_message_path = fmp
            except Exception:
                pass
            # Append public message
            art = data.get("artifact", {}) or {}
            branch = art.get("branch_final") or art.get("branch_planned")
            dur = metrics.get("duration_seconds") or data.get("duration_seconds")
            toks = metrics.get("total_tokens")
            parts = ["completed"]
            if dur is not None:
                parts.append(f"time={dur}")
            if toks is not None:
                parts.append(f"tok={toks}")
            if branch:
                parts.append(f"branch={branch}")
            self._append_msg(iid, " ".join(parts))
        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _handle_task_progress(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if not inst:
            # Create a placeholder if we somehow missed scheduled/started
            inst = InstanceDisplay(instance_id=iid, strategy_name="")
            self.state.current_run.instances[iid] = inst
        # Update current activity based on canonical payload
        phase = data.get("phase")
        activity = data.get("activity")
        tool = data.get("tool")
        if tool:
            inst.last_tool_use = tool
        if activity:
            inst.current_activity = activity
        elif phase:
            friendly = {
                "workspace_preparing": "Preparing workspace...",
                "container_creating": "Creating container...",
                "container_env_preparing": "Preparing container env...",
                "container_env_prepared": "Container env ready",
                "container_created": "Container created",
                "agent_starting": "Starting Agent...",
                "result_collection": "Collecting results...",
                "branch_imported": "Branch imported",
                "no_changes": "No changes",
                "cleanup": "Cleaning up...",
                "assistant": "Agent is thinking...",
                "system": "Agent connected",
                "tool_use": f"Using {tool}" if tool else "Tool use",
            }.get(phase)
            if friendly:
                inst.current_activity = friendly
        inst.last_updated = datetime.now()
        # Append progress message (phase/activity/tool)
        phase = data.get("phase")
        activity = data.get("activity")
        tool = data.get("tool")
        msg = None
        if activity:
            msg = f"progress activity={activity}"
        elif phase:
            msg = f"progress phase={phase}"
        if tool:
            msg = (msg + f" tool={tool}") if msg else f"progress tool={tool}"
        if msg:
            self._append_msg(iid, msg)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _handle_task_failed(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.FAILED
            inst.error = data.get("message")
        # Append failed message and offline egress hint (once per instance implied by de-dup in UI)
        etype = data.get("error_type")
        msg = (data.get("message") or "").strip().replace("\n", " ")
        parts = ["failed"]
        if etype:
            parts.append(f"type={etype}")
        if msg:
            parts.append(f'message="{msg[:80]}"')
        # offline hint
        if str(data.get("network_egress")) == "offline":
            parts.append("hint=egress=offline:set runner.network_egress=online/proxy")
        self._append_msg(iid, " ".join(parts))
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass
        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)

    def _handle_task_interrupted(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            # Terminal interrupted state
            inst.status = InstanceStatus.INTERRUPTED
            inst.last_updated = datetime.now()
        self._append_msg(iid, "interrupted")
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass
            # Adjust aggregates
            self.state.current_run.active_instances = max(
                0, self.state.current_run.active_instances - 1
            )
        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)

    # Run-level event handlers

    def _handle_run_started(self, event: Dict[str, Any]) -> None:
        """Handle run.started event."""
        data = event.get("data", {})

        self.state.current_run = RunDisplay(
            run_id=data.get("run_id", "unknown"),
            prompt=data.get("prompt", ""),
            repo_path=data.get("repo_path", ""),
            base_branch=data.get("base_branch", "main"),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.connected_to_orchestrator = True

    def _handle_run_completed(self, event: Dict[str, Any]) -> None:
        """Handle run.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        run = self.state.current_run

        run.completed_at = self._parse_timestamp(event.get("timestamp"))
        run.total_cost = data.get("total_cost", 0.0)
        run.total_tokens = data.get("total_tokens", 0)

    def _handle_run_failed(self, event: Dict[str, Any]) -> None:
        """Handle run.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        self.state.add_error(f"Run failed: {data.get('error', 'Unknown error')}")

    # Strategy-level event handlers

    def _handle_strategy_started(self, event: Dict[str, Any]) -> None:
        """Handle strategy.started event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        # Canonical envelope carries the strategy execution id
        strategy_id = event.get("strategy_execution_id") or data.get("strategy_id")
        if not strategy_id:
            return

        strategy = StrategyDisplay(
            strategy_id=strategy_id,
            strategy_name=data.get("name", data.get("strategy_name", "unknown")),
            config=data.get("params", data.get("config", {})),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.current_run.strategies[strategy_id] = strategy
        # Prime run start time on first strategy start
        try:
            if not self.state.current_run.started_at:
                self.state.current_run.started_at = strategy.started_at
        except Exception:
            pass
        # Log with the canonical 'name' when present
        logger.info(
            f"Created strategy {strategy_id} ({strategy.strategy_name or data.get('name','unknown')})"
        )

    def _handle_strategy_completed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = event.get("strategy_execution_id") or data.get("strategy_id")
        if not strategy_id or strategy_id not in self.state.current_run.strategies:
            return

        strategy = self.state.current_run.strategies[strategy_id]
        strategy.completed_at = self._parse_timestamp(event.get("timestamp"))
        strategy.is_complete = True

    def _handle_strategy_failed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = data.get("strategy_id")
        if strategy_id and strategy_id in self.state.current_run.strategies:
            strategy = self.state.current_run.strategies[strategy_id]
            strategy.is_complete = True
            self.state.add_error(
                f"Strategy {strategy.strategy_name} failed: {data.get('error', 'Unknown')}"
            )

    # Instance-level event handlers

    def _handle_instance_queued(self, event: Dict[str, Any]) -> None:
        """Handle instance.queued event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        # Try to get instance_id from both places (top level and in data)
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id:
            logger.warning(f"No instance_id in event: {event}")
            return

        # Avoid double-creating when canonical task.scheduled already handled it
        if instance_id not in self.state.current_run.instances:
            instance = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy", "unknown"),
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name"),
                last_updated=self._parse_timestamp(event.get("timestamp"))
                or datetime.now(),
            )
            self.state.current_run.instances[instance_id] = instance
        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)

        # Add instance to strategy tracking
        strategy_name = data.get("strategy", "unknown")
        for strategy in self.state.current_run.strategies.values():
            if strategy.strategy_name == strategy_name:
                strategy.instance_ids.append(instance_id)
                strategy.total_instances += 1
                logger.info(
                    f"Added instance {instance_id} to strategy {strategy.strategy_id}"
                )
                break

        logger.info(
            f"Created instance {instance_id} from queued event, total instances: {len(self.state.current_run.instances)}"
        )

    def _handle_instance_started(self, event: Dict[str, Any]) -> None:
        """Handle instance.started event."""
        if not self.state.current_run:
            logger.warning("Got instance.started but no current_run")
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.started without instance_id")
            return

        # Check if instance was already created by queued event
        if instance_id not in self.state.current_run.instances:
            logger.info(f"Creating instance {instance_id} from started event")
            # Create the instance if it doesn't exist
            data = event.get("data", {})
            instance = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy", "unknown"),
                status=InstanceStatus.RUNNING,
                started_at=self._parse_timestamp(event.get("timestamp")),
                prompt=data.get("prompt"),
                model=data.get("model", ""),
                current_activity="Starting...",
                last_updated=datetime.now(),
            )
            self.state.current_run.instances[instance_id] = instance
            self.state.current_run.total_instances += 1
            self.state.current_run.active_instances += 1
            logger.info(
                f"Instance {instance_id} started. Active: {self.state.current_run.active_instances}"
            )
        else:
            # Update existing instance
            data = event.get("data", {})
            instance = self.state.current_run.instances[instance_id]

            # Only increment active if status is changing from QUEUED to RUNNING
            was_queued = instance.status == InstanceStatus.QUEUED

            instance.status = InstanceStatus.RUNNING
            instance.started_at = self._parse_timestamp(event.get("timestamp"))
            instance.prompt = data.get("prompt")
            instance.model = data.get("model", "")
            # Backfill strategy name if available
            if instance.strategy_name == "unknown" and data.get("strategy"):
                instance.strategy_name = data.get("strategy")
            # Ensure we show a meaningful activity line
            instance.current_activity = instance.current_activity or "Starting..."
            instance.last_updated = datetime.now()

            if was_queued:
                self.state.current_run.active_instances += 1
                logger.info(
                    f"Instance {instance_id} started (was queued). Active: {self.state.current_run.active_instances}"
                )

    def _handle_instance_completed(self, event: Dict[str, Any]) -> None:
        """Handle instance.completed event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.completed without instance_id")
            return

        if instance_id not in self.state.current_run.instances:
            logger.warning(
                f"Got instance.completed for unknown instance: {instance_id}"
            )
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        # Only update to completed if not already completed (avoid race conditions)
        if instance.status != InstanceStatus.COMPLETED:
            instance.status = InstanceStatus.COMPLETED
            instance.completed_at = self._parse_timestamp(event.get("timestamp"))
            instance.branch_name = data.get("branch_name")
            instance.duration_seconds = data.get("duration_seconds", 0.0)
            instance.current_activity = "Completed"  # Clear the activity
            instance.last_updated = datetime.now()
            logger.info(
                f"Instance {instance_id} completed after activity: {instance.current_activity}"
            )

        # Update metrics from data
        metrics = data.get("metrics", {})
        instance.cost = metrics.get("total_cost", 0.0)
        instance.total_tokens = metrics.get("total_tokens", 0)
        instance.input_tokens = metrics.get("input_tokens", 0)
        instance.output_tokens = metrics.get("output_tokens", 0)

        # Update run totals
        self.state.current_run.completed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )
        self.state.current_run.total_cost += instance.cost
        self.state.current_run.total_tokens += instance.total_tokens

        logger.info(
            f"Instance {instance_id} completed. Run totals - Completed: {self.state.current_run.completed_instances}, Active: {self.state.current_run.active_instances}"
        )

        # Update strategy totals
        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.completed_instances += 1
                break

    def _handle_instance_failed(self, event: Dict[str, Any]) -> None:
        """Handle instance.failed event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.status = InstanceStatus.FAILED
        instance.completed_at = self._parse_timestamp(event.get("timestamp"))
        instance.error = data.get("error")
        instance.error_type = data.get("error_type")
        instance.last_updated = datetime.now()

        self.state.current_run.failed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

        # Update strategy totals
        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.failed_instances += 1
                break

    def _handle_instance_progress(self, event: Dict[str, Any]) -> None:
        """Handle instance.progress event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.current_activity = data.get("activity")
        instance.last_updated = datetime.now()

    def _handle_state_instance_registered(self, event: Dict[str, Any]) -> None:
        """Handle state.instance_registered snapshot to populate strategy name early."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id:
            return

        # Ensure instance exists
        inst = self.state.current_run.instances.get(instance_id)
        if not inst:
            inst = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy_name", "unknown"),
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name"),
            )
            self.state.current_run.instances[instance_id] = inst
        else:
            # Update strategy name/branch if missing
            if inst.strategy_name == "unknown" and data.get("strategy_name"):
                inst.strategy_name = data.get("strategy_name")
            if not inst.branch_name and data.get("branch_name"):
                inst.branch_name = data.get("branch_name")

        # Keep total_instances accurate
        self.state.current_run.total_instances = len(self.state.current_run.instances)

        # Add to strategy grouping if possible
        strategy_name = data.get("strategy_name")
        if strategy_name:
            for strategy in self.state.current_run.strategies.values():
                if (
                    strategy.strategy_name == strategy_name
                    and instance_id not in strategy.instance_ids
                ):
                    strategy.instance_ids.append(instance_id)
                    strategy.total_instances += 1
                    break

    # Agent/assistant event handlers

    def _handle_agent_system(self, event: Dict[str, Any]) -> None:
        """Handle agent system message (connection established)."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Agent connected"
        instance.last_updated = datetime.now()

    def _handle_agent_assistant(self, event: Dict[str, Any]) -> None:
        """Handle agent assistant message."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Agent is thinking..."
        instance.last_updated = datetime.now()

    def _handle_agent_tool_use(self, event: Dict[str, Any]) -> None:
        """Handle agent tool use."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        tool_name = data.get("tool", "unknown")
        instance.last_tool_use = tool_name

        # Make tool descriptions more user-friendly
        tool_descriptions = {
            "str_replace_editor": "Editing files",
            "bash": "Running commands",
            "str_replace_based_edit_tool": "Editing code",
            "read_file": "Reading files",
            "write_file": "Writing files",
            "list_files": "Listing files",
            "search_files": "Searching files",
            "find_files": "Finding files",
        }

        friendly_name = tool_descriptions.get(tool_name, f"Using {tool_name}")
        instance.current_activity = friendly_name
        instance.last_updated = datetime.now()

    def _handle_agent_tool_result(self, event: Dict[str, Any]) -> None:
        """Handle agent tool result."""
        # Could track success/failure of tool uses if needed
        pass

    def _handle_agent_result(self, event: Dict[str, Any]) -> None:
        """Handle agent final result."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        # Update final metrics if not already set
        metrics = data.get("metrics", {})
        if metrics:
            if instance.cost == 0.0:
                instance.cost = metrics.get("total_cost", 0.0)
            if instance.total_tokens == 0:
                instance.total_tokens = metrics.get("total_tokens", 0)
                instance.input_tokens = metrics.get("input_tokens", 0)
                instance.output_tokens = metrics.get("output_tokens", 0)

    def _handle_instance_canceled(self, event: Dict[str, Any]) -> None:
        """Handle instance canceled (treated as interrupted)."""
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]
        instance.status = InstanceStatus.INTERRUPTED
        instance.error = data.get("error") or "canceled"
        instance.error_type = data.get("error_type") or "canceled"
        instance.last_updated = datetime.now()
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

    def _handle_state_instance_updated(self, event: Dict[str, Any]) -> None:
        """Handle state.instance_updated snapshots for terminal transitions."""
        if not self.state.current_run:
            return
        data = event.get("data", {})
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        new_state = data.get("new_state")
        instance = self.state.current_run.instances[instance_id]
        if new_state == "interrupted":
            instance.status = InstanceStatus.INTERRUPTED
            instance.last_updated = datetime.now()
            self.state.current_run.active_instances = max(
                0, self.state.current_run.active_instances - 1
            )

    # Instance phase event handlers

    def _handle_instance_workspace_preparing(self, event: Dict[str, Any]) -> None:
        """Handle workspace preparation event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        previous_activity = instance.current_activity
        instance.current_activity = "Preparing workspace..."
        instance.last_updated = datetime.now()
        logger.info(
            f"Instance {instance_id}: '{previous_activity}' -> 'Preparing workspace...'"
        )

    def _handle_instance_container_creating(self, event: Dict[str, Any]) -> None:
        """Handle container creation event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Creating container..."
        instance.last_updated = datetime.now()

    def _handle_instance_container_env_preparing(self, event: Dict[str, Any]) -> None:
        """Handle container env preparing event."""
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Preparing container env..."
        instance.last_updated = datetime.now()

    def _handle_instance_container_env_prepared(self, event: Dict[str, Any]) -> None:
        """Handle container env prepared event."""
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Container env ready"
        instance.last_updated = datetime.now()

    def _handle_instance_container_created(self, event: Dict[str, Any]) -> None:
        """Handle container created event."""
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Container created"
        instance.last_updated = datetime.now()

    def _handle_instance_agent_starting(self, event: Dict[str, Any]) -> None:
        """Handle agent starting event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Starting Agent..."
        instance.last_updated = datetime.now()

    def _handle_instance_result_collection(self, event: Dict[str, Any]) -> None:
        """Handle result collection event."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Collecting results..."
        instance.last_updated = datetime.now()

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


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


class AsyncEventStream:
    """Async event stream reader with file watching."""

    def __init__(self, event_processor: EventProcessor):
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
            self._observer.schedule(event_handler, str(events_file.parent), recursive=False)  # type: ignore[union-attr]
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
                    # Detect rotation or truncation
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
                                # On rotation/truncation, jump to end of current file to avoid reprocessing duplicates
                                self._last_position = size
                                self._prev_inode = inode
                                self._prev_size = size
                                if rotated:
                                    self._rotations += 1
                                if truncated:
                                    self._truncations += 1
                    except Exception as e:
                        logger.debug(f"Stat failed for events file: {e}")

                    # Use regular file operations instead of aiofiles for more reliable reading
                    try:
                        with open(self._events_file, "rb") as f:
                            # Seek to last position
                            f.seek(self._last_position)

                            # Read new lines
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
                                    # Yield control after each event to allow processing
                                    await asyncio.sleep(0)

                            # Update position
                            new_position = f.tell()
                            if new_position > self._last_position and lines_read > 0:
                                logger.info(
                                    f"Read {lines_read} lines, position {self._last_position} -> {new_position}"
                                )
                            self._last_position = new_position
                            try:
                                self._bytes_read_total += max(
                                    0, new_position - self._last_position
                                )
                            except Exception:
                                pass

                            # Update state
                            self.event_processor.state.last_event_start_offset = (
                                self._last_position
                            )
                            # Update previous size
                            try:
                                st2 = self._events_file.stat()
                                self._prev_size = st2.st_size
                            except Exception:
                                pass
                    except (OSError, IOError) as e:
                        logger.error(f"Error reading events file: {e}")

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
                        f"queue_event type={e_type} qsize={q_after_get} parse_ms={(t_p1-t_p0)*1000:.2f} handle_ms={(t_h1-t_p1)*1000:.2f} wait_ms={(t_p0-t_wait0)*1000:.2f}"
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
