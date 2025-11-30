"""Handlers for agent stream events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class AgentEventHandlers:
    """Agent/assistant event handlers."""

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

        data = event.get("data", {}) or {}
        usage = None
        message_id = None

        if isinstance(data, dict):
            maybe_usage = data.get("usage")
            if isinstance(maybe_usage, dict):
                usage = maybe_usage
            maybe_message_id = data.get("message_id")
            if isinstance(maybe_message_id, str):
                message_id = maybe_message_id

            if usage is None:
                message = data.get("message")
                if isinstance(message, dict):
                    maybe_usage = message.get("usage")
                    if isinstance(maybe_usage, dict):
                        usage = maybe_usage
                    if not message_id:
                        mid = message.get("id")
                        if isinstance(mid, str):
                            message_id = mid

        if not isinstance(usage, dict):
            return

        self._apply_usage_metrics(instance, usage, message_id)

    def _handle_agent_turn_complete(self, event: Dict[str, Any]) -> None:
        """Handle explicit turn completion events with token metrics."""
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {}) or {}
        metrics = data.get("turn_metrics", {}) if isinstance(data, dict) else {}
        if not isinstance(metrics, dict):
            return

        instance = self.state.current_run.instances[instance_id]

        def _as_int(val: Any) -> int:
            try:
                return int(val or 0)
            except Exception:
                return 0

        input_tokens = _as_int(metrics.get("input_tokens"))
        output_tokens = _as_int(metrics.get("output_tokens"))
        cache_creation = _as_int(metrics.get("cache_creation_input_tokens"))
        cache_read = _as_int(
            metrics.get("cache_read_input_tokens", metrics.get("cached_input_tokens"))
        )
        reasoning_tokens = _as_int(metrics.get("reasoning_output_tokens"))
        tokens = _as_int(metrics.get("tokens"))
        total_tokens = _as_int(metrics.get("total_tokens"))

        usage = {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "tokens": total_tokens or tokens,
        }
        if tokens:
            usage["tokens"] = tokens

        prev_total = instance.total_tokens
        turn_number = data.get("turn_number")
        message_id = (
            f"turn-{turn_number}"
            if isinstance(turn_number, (int, float, str))
            else None
        )

        self._apply_usage_metrics(instance, usage, message_id)

        post_usage_total = instance.total_tokens
        if total_tokens and total_tokens > instance.total_tokens:
            instance.total_tokens = total_tokens
            instance.usage_running_total = max(
                instance.usage_running_total, total_tokens
            )
            delta_total = total_tokens - post_usage_total
            if delta_total > 0:
                if self.state.current_run:
                    self.state.current_run.total_tokens += delta_total
                instance.applied_run_tokens += delta_total
        else:
            delta_total = instance.total_tokens - prev_total

        if input_tokens and input_tokens > instance.input_tokens:
            instance.input_tokens = input_tokens
            instance.usage_input_running_total = max(
                instance.usage_input_running_total, input_tokens
            )
        if output_tokens and output_tokens + reasoning_tokens > instance.output_tokens:
            instance.output_tokens = output_tokens + reasoning_tokens
            instance.usage_output_running_total = max(
                instance.usage_output_running_total,
                output_tokens + reasoning_tokens,
            )

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

        metrics = data.get("metrics", {})
        if metrics:
            prev_total = instance.total_tokens
            try:
                total_tokens = int(
                    metrics.get("total_tokens", instance.total_tokens)
                    or instance.total_tokens
                )
            except Exception:
                total_tokens = instance.total_tokens
            try:
                input_raw = int(
                    metrics.get("input_tokens", instance.input_tokens)
                    or instance.input_tokens
                )
            except Exception:
                input_raw = instance.input_tokens
            try:
                output_raw = int(
                    metrics.get("output_tokens", instance.output_tokens)
                    or instance.output_tokens
                )
            except Exception:
                output_raw = instance.output_tokens

            instance.total_tokens = total_tokens
            instance.usage_running_total = max(
                instance.usage_running_total, total_tokens
            )
            instance.usage_input_running_total = max(
                instance.usage_input_running_total, input_raw
            )
            instance.usage_output_running_total = max(
                instance.usage_output_running_total, output_raw
            )

            fresh_input = max(0, total_tokens - output_raw)
            instance.input_tokens = fresh_input
            instance.usage_prompt_running_total = max(
                instance.usage_prompt_running_total, fresh_input
            )
            instance.cached_input_tokens = max(
                instance.cached_input_tokens, max(0, input_raw - fresh_input)
            )
            instance.output_tokens = output_raw

            delta_total = total_tokens - prev_total
            if delta_total > 0:
                if self.state.current_run:
                    try:
                        self.state.current_run.total_tokens += delta_total
                    except Exception:
                        pass
                instance.applied_run_tokens += delta_total

            if instance.cost == 0.0:
                try:
                    instance.cost = float(metrics.get("total_cost", 0.0) or 0.0)
                except Exception:
                    pass


__all__ = ["AgentEventHandlers"]
