"""
Parser for the Anthropic agent stream-json output format.

Extracts events, metrics, and session information from the tool's structured
output while keeping parsing predictable and bounded.
"""

from __future__ import annotations

import logging
from typing import Optional

from .claude_helpers import (
    ClaudeParserHelpers,
    EventDict,
    EventPayload,
)

logger = logging.getLogger(__name__)

__all__ = ["ClaudeOutputParser"]


class ClaudeOutputParser(ClaudeParserHelpers):
    """Parses the Anthropic agent's stream-json output format."""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.total_tokens: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost: float = 0.0
        self.turn_count: int = 0
        self.last_message: Optional[str] = None

    def parse_line(self, line: str) -> Optional[EventDict]:
        """Parse a single JSONL line into an event dict or ``None``."""
        normalized_line = (line or "").strip()
        if not normalized_line:
            return None

        payload = self._decode_json(normalized_line)
        if payload is None:
            return None

        event_type = self._normalize_type(payload.get("type"))
        if event_type is None:
            return None

        handlers = {
            "assistant": self._handle_assistant,
            "user": self._handle_user,
            "system": self._handle_system,
            "result": self._handle_result,
            "session_started": self._handle_session_started,
            "turn_started": self._handle_turn_started,
            "turn_complete": self._handle_turn_complete,
            "tool_use": self._handle_tool_use_event,
            "tool_result": self._handle_tool_result_event,
            "message": self._handle_message_event,
            "error": self._handle_error_event,
            "session_complete": self._handle_session_complete,
            "task_update": self._handle_task_update,
            "cost_limit_warning": self._handle_cost_limit_warning,
        }

        handler = handlers.get(event_type, self._handle_unknown)
        return handler(payload, event_type)

    # Event handlers -----------------------------------------------------

    def _handle_assistant(self, payload: EventPayload, _: str) -> Optional[EventDict]:
        message = payload.get("message")
        usage_payload, message_id = self._extract_usage(message)

        content_items = message.get("content") if isinstance(message, dict) else None
        if isinstance(content_items, list):
            tool_event = self._extract_tool_use_from_content(
                content_items, payload, usage_payload, message_id
            )
            if tool_event:
                return tool_event
            self._update_last_message_from_content(content_items)

        event = self._base_event("assistant", payload)
        if usage_payload:
            event["usage"] = usage_payload
        if message_id:
            event["message_id"] = message_id
        if self.last_message:
            event["content"] = self._truncate_content(self.last_message)

        return event

    def _handle_user(self, payload: EventPayload, _: str) -> Optional[EventDict]:
        message = payload.get("message")
        content_items = message.get("content") if isinstance(message, dict) else None
        if isinstance(content_items, list):
            for item in content_items:
                if item.get("type") == "tool_result":
                    return self._build_tool_result_event(item, payload)
        return self._base_event("user", payload)

    def _handle_system(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("system", payload)
        if payload.get("subtype") == "init":
            self.session_id = payload.get("session_id")
            event["session_id"] = self.session_id
        return event

    def _handle_result(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("result", payload)
        event["session_id"] = self.session_id or payload.get("session_id")
        event["final_message"] = payload.get("result", self.last_message)

        usage = payload.get("usage") if isinstance(payload, dict) else None
        input_tokens = self._sum_input_tokens(usage) if isinstance(usage, dict) else 0
        output_tokens = (
            self._coerce_int(usage.get("output_tokens"))
            if isinstance(usage, dict)
            else 0
        )
        total_tokens = input_tokens + output_tokens or self.total_tokens

        if input_tokens or output_tokens:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.total_tokens = total_tokens

        event["metrics"] = {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": payload.get("total_cost_usd", self.total_cost),
            "turn_count": payload.get("num_turns", self.turn_count),
            "duration_ms": payload.get("duration_ms", 0),
            "duration_api_ms": payload.get("duration_api_ms", 0),
        }

        if "total_cost_usd" in payload:
            self.total_cost = self._coerce_float(payload.get("total_cost_usd"))
        if "num_turns" in payload:
            self.turn_count = self._coerce_int(payload.get("num_turns"))

        return event

    def _handle_session_started(self, payload: EventPayload, _: str) -> EventDict:
        self.session_id = payload.get("session_id") or self.session_id
        event = self._base_event("session_started", payload)
        event["session_id"] = self.session_id
        return event

    def _handle_turn_started(self, payload: EventPayload, _: str) -> EventDict:
        self.turn_count += 1
        event = self._base_event("turn_started", payload)
        event["turn_number"] = self.turn_count
        return event

    def _handle_turn_complete(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("turn_complete", payload)
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        tokens_used = self._coerce_int(metrics.get("tokens_used")) if metrics else 0
        cost = self._coerce_float(metrics.get("cost_usd")) if metrics else 0.0

        self.total_tokens += tokens_used
        self.total_cost += cost

        if metrics:
            event["turn_metrics"] = {
                "tokens": tokens_used,
                "cost": cost,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
            }
        return event

    def _handle_tool_use_event(self, payload: EventPayload, _: str) -> EventDict:
        tool = payload.get("tool") if isinstance(payload, dict) else {}
        tool_name = tool.get("name", "") if isinstance(tool, dict) else ""
        params = tool.get("parameters", {}) if isinstance(tool, dict) else {}
        return self._build_tool_use_event(tool_name, params, payload)

    def _handle_tool_result_event(self, payload: EventPayload, _: str) -> EventDict:
        success = payload.get("success", True)
        output = self._truncate_output(payload.get("output"))
        event = self._base_event("tool_result", payload)
        event["success"] = bool(success)
        if payload.get("error"):
            event["error"] = payload["error"]
        if output:
            event["output"] = output
        return event

    def _handle_message_event(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("message", payload)
        content = payload.get("content", "")
        self.last_message = content
        event["content"] = content
        return event

    def _handle_error_event(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("error", payload)
        event["error_type"] = payload.get("error_type", "unknown")
        event["error_message"] = payload.get("error_message", "")
        return event

    def _handle_session_complete(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("session_complete", payload)
        event["session_id"] = self.session_id
        event["final_message"] = payload.get("result", self.last_message)

        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            self.input_tokens = self._sum_input_tokens(usage)
            self.output_tokens = self._coerce_int(usage.get("output_tokens"))
            self.total_tokens = self.input_tokens + self.output_tokens

        if "total_cost_usd" in payload:
            self.total_cost = self._coerce_float(payload.get("total_cost_usd"))
        if "num_turns" in payload:
            self.turn_count = self._coerce_int(payload.get("num_turns"))

        event["metrics"] = {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": payload.get("total_cost_usd", self.total_cost),
            "turn_count": payload.get("num_turns", self.turn_count),
            "duration_ms": payload.get("duration_ms", 0),
            "duration_api_ms": payload.get("duration_api_ms", 0),
        }
        return event

    def _handle_task_update(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("task_update", payload)
        tasks = payload.get("tasks", [])
        event["tasks"] = tasks
        event["task_count"] = len(tasks)

        status_counts: dict[str, int] = {}
        for task in tasks:
            status = (
                task.get("status", "pending") if isinstance(task, dict) else "pending"
            )
            status_counts[status] = status_counts.get(status, 0) + 1
        event["status_counts"] = status_counts
        return event

    def _handle_cost_limit_warning(self, payload: EventPayload, _: str) -> EventDict:
        event = self._base_event("cost_limit_warning", payload)
        event["current_cost"] = self._coerce_float(payload.get("current_cost"))
        event["limit"] = self._coerce_float(payload.get("limit"))
        event["percentage"] = self._coerce_float(payload.get("percentage"))
        return event

    def _handle_unknown(self, payload: EventPayload, event_type: str) -> EventDict:
        event = self._base_event(event_type, payload)
        for key in ("status", "progress", "data"):
            if key in payload:
                event[key] = payload[key]
        return event

    def get_summary(self) -> EventDict:
        """Return summary of parsed session metrics and final content."""
        return {
            "session_id": self.session_id,
            "final_message": self.last_message,
            "metrics": {
                "total_tokens": self.total_tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_cost": self.total_cost,
                "turn_count": self.turn_count,
            },
        }
