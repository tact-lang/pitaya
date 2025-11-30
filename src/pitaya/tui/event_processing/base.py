"""Base components and shared utilities for event processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

from ..models import InstanceDisplay, RunDisplay, TUIState
from .logging_config import logger


class EventProcessorBase:
    """Core plumbing for processing events and dispatching handlers."""

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
            "instance.agent_turn_complete": self._handle_agent_turn_complete,
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
                    f"event_processed type={event_type} iid={iid or '-'} dur_ms={(t1 - t0) * 1000:.2f}"
                )
            except (AttributeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"Error processing event {event_type}: {e}")
                self.state.add_error(f"Event processing error: {e}")
        else:
            # Accept canonical strategy/task events as pass-through for now
            logger.debug(f"No handler for event type: {event_type}")

    def _ensure_current_run(self) -> None:
        if not self.state.current_run:
            # Create a placeholder RunDisplay so task/strategy events can attach
            self.state.current_run = RunDisplay(
                run_id="unknown",
                prompt="",
                repo_path="",
                base_branch="main",
            )

    def _apply_usage_metrics(
        self,
        instance: InstanceDisplay,
        usage: Dict[str, Any],
        message_id: Optional[str] = None,
    ) -> None:
        """Incrementally apply token usage to an instance and run totals."""
        if message_id and message_id in instance.usage_message_ids:
            return

        def _as_int(value: Any) -> int:
            try:
                return int(value or 0)
            except Exception:
                return 0

        input_tokens = _as_int(usage.get("input_tokens"))
        cache_creation = _as_int(usage.get("cache_creation_input_tokens"))
        cache_read = _as_int(usage.get("cache_read_input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))
        output_tokens += _as_int(usage.get("reasoning_output_tokens"))

        cumulative_input = input_tokens + cache_creation + cache_read
        cumulative_output = output_tokens

        explicit_total = usage.get("tokens")
        if explicit_total is None:
            explicit_total = usage.get("total_tokens")
        if explicit_total is not None:
            try:
                cumulative_total = max(0, int(explicit_total))
            except Exception:
                cumulative_total = cumulative_input + cumulative_output
        else:
            cumulative_total = cumulative_input + cumulative_output

        fresh_input_total = max(0, cumulative_total - cumulative_output)
        cached_input_total = max(0, cumulative_input - fresh_input_total)

        prev_usage_total = getattr(instance, "usage_running_total", 0)
        prev_input_total = getattr(instance, "usage_input_running_total", 0)
        prev_output_total = getattr(instance, "usage_output_running_total", 0)
        prev_prompt_total = getattr(instance, "usage_prompt_running_total", 0)

        delta_total = max(0, cumulative_total - prev_usage_total)
        delta_prompt = max(0, fresh_input_total - prev_prompt_total)
        delta_output = max(0, cumulative_output - prev_output_total)

        if delta_total <= 0 and message_id:
            instance.usage_message_ids.add(message_id)
            return

        instance.usage_running_total = max(prev_usage_total, cumulative_total)
        instance.usage_input_running_total = max(prev_input_total, cumulative_input)
        instance.usage_output_running_total = max(prev_output_total, cumulative_output)
        instance.usage_prompt_running_total = max(prev_prompt_total, fresh_input_total)
        instance.cached_input_tokens = cached_input_total

        if delta_prompt:
            instance.input_tokens = instance.usage_prompt_running_total
        if delta_output:
            instance.output_tokens = instance.usage_output_running_total
        if delta_total:
            instance.total_tokens = instance.usage_running_total

        if message_id:
            instance.usage_message_ids.add(message_id)

        if delta_total > 0:
            if self.state.current_run:
                try:
                    self.state.current_run.total_tokens += delta_total
                except Exception:
                    pass
            instance.applied_run_tokens += delta_total

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


__all__ = ["EventProcessorBase"]
