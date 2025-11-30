"""
Shared helpers for parsing Claude stream-json events.

These utilities mutate parser state via subclass attributes and keep
individual helper functions small and testable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EventPayload = Dict[str, Any]
EventDict = Dict[str, Any]

MAX_OUTPUT_PREVIEW = 500
MAX_CONTENT_SNIPPET = 4000


class ClaudeParserHelpers:
    """Utility mixin for ClaudeOutputParser."""

    def _base_event(self, event_type: str, payload: EventPayload) -> EventDict:
        return {"type": event_type, "timestamp": self._timestamp(payload)}

    def _timestamp(self, payload: EventPayload) -> str:
        timestamp = payload.get("timestamp")
        if isinstance(timestamp, str) and timestamp.strip():
            return timestamp
        return datetime.now(timezone.utc).isoformat()

    def _decode_json(self, raw_line: str) -> Optional[EventPayload]:
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug(
                "claude_parser: unable to decode JSON line: %s", raw_line[:160]
            )
            return None
        if not isinstance(payload, dict):
            logger.debug("claude_parser: JSON line did not contain an object")
            return None
        return payload

    def _normalize_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized or None

    def _sum_input_tokens(self, usage: Dict[str, Any]) -> int:
        return (
            self._coerce_int(usage.get("input_tokens"))
            + self._coerce_int(usage.get("cache_creation_input_tokens"))
            + self._coerce_int(usage.get("cache_read_input_tokens"))
        )

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _truncate_output(self, output: Any) -> Optional[str]:
        if output is None:
            return None
        text = output if isinstance(output, str) else str(output)
        if len(text) > MAX_OUTPUT_PREVIEW:
            return text[:MAX_OUTPUT_PREVIEW]
        return text

    def _truncate_content(self, content: str) -> str:
        if len(content) > MAX_CONTENT_SNIPPET:
            return content[:MAX_CONTENT_SNIPPET] + "…"
        return content

    def _extract_usage(
        self, message: Any
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(message, dict):
            return None, None

        message_id = message.get("id") if isinstance(message.get("id"), str) else None
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return None, message_id

        input_tokens = self._coerce_int(usage.get("input_tokens"))
        cache_create = self._coerce_int(usage.get("cache_creation_input_tokens"))
        cache_read = self._coerce_int(usage.get("cache_read_input_tokens"))
        output_tokens = self._coerce_int(usage.get("output_tokens"))

        tokens_used = input_tokens + cache_create + cache_read + output_tokens
        self.total_tokens += tokens_used
        self.input_tokens += input_tokens + cache_create + cache_read
        self.output_tokens += output_tokens

        usage_payload = {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_create,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output_tokens,
        }
        return usage_payload, message_id

    def _extract_tool_use_from_content(
        self,
        items: list[Any],
        payload: EventPayload,
        usage_payload: Optional[Dict[str, Any]],
        message_id: Optional[str],
    ) -> Optional[EventDict]:
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            event = self._build_tool_use_event(
                str(item.get("name") or ""),
                item.get("input") if isinstance(item.get("input"), dict) else {},
                payload,
            )
            if usage_payload:
                event["usage"] = usage_payload
            if message_id:
                event["message_id"] = message_id
            return event
        return None

    def _build_tool_use_event(
        self, tool_name: str, tool_params: Dict[str, Any], payload: EventPayload
    ) -> EventDict:
        event = {
            "type": "tool_use",
            "timestamp": self._timestamp(payload),
            "tool": tool_name,
            "parameters": tool_params,
        }
        if tool_name == "Edit":
            event["file_path"] = tool_params.get("file_path")
            event["action"] = "edit"
        elif tool_name == "Write":
            event["file_path"] = tool_params.get("file_path")
            event["action"] = "write"
        elif tool_name == "Bash":
            event["command"] = tool_params.get("command")
            event["action"] = "bash"
        elif tool_name == "Read":
            event["file_path"] = tool_params.get("file_path")
            event["action"] = "read"
        return event

    def _build_tool_result_event(
        self, result_item: Dict[str, Any], parent_data: EventPayload
    ) -> EventDict:
        content = result_item.get("content", "")
        return {
            "type": "tool_result",
            "timestamp": self._timestamp(parent_data),
            "success": not result_item.get("is_error", False),
            "output": self._truncate_output(content) or "",
        }

    def _update_last_message_from_content(self, items: list[Any]) -> None:
        text_parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            self.last_message = " ".join(text_parts)
