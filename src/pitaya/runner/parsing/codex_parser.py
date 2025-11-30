"""
Parser for Codex 0.58+ ThreadEvent JSONL output.

Codex emits JSON lines with a stable schema (thread/turn/item/error events).
This parser maps those events into the internal Pitaya event set used by the
instance runner (assistant/tool_use/tool_result/turn_complete/error) and
accumulates token metrics and summary information.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .codex_item_handlers import handle_item_event

logger = logging.getLogger(__name__)

__all__ = ["CodexOutputParser"]

EventPayload = Dict[str, Any]
EventDict = Dict[str, Any]

MAX_OUTPUT_PREVIEW = 500


class CodexOutputParser:
    """Parse Codex ThreadEvents and derive summary metrics."""

    def __init__(self) -> None:
        """Initialize empty parser state for a Codex session."""
        self.session_id: Optional[str] = None
        self.last_message: Optional[str] = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.total_tokens: int = 0
        self.last_error: Optional[str] = None

    def parse_line(self, line: str) -> Optional[EventDict]:
        """Parse a raw JSONL line into an internal event dict or ``None``."""
        normalized_line = (line or "").strip()
        if not normalized_line:
            return None

        payload = self._decode_json(normalized_line)
        if payload is None:
            return None

        if isinstance(payload.get("msg"), dict):
            payload = payload["msg"]

        event_type = self._normalize_type(payload.get("type"))
        if event_type is None:
            return None

        direct_handlers = {
            "thread.started": self._handle_thread_started,
            "turn.started": self._handle_turn_started,
            "turn.completed": self._handle_turn_completed,
            "turn.failed": self._handle_turn_failed,
            "error": self._handle_stream_error,
        }

        handler = direct_handlers.get(event_type)
        if handler:
            return handler(payload)

        if event_type.startswith("item."):
            return self._handle_item_event(event_type, payload)

        logger.debug("codex_parser: unrecognized event type '%s'", event_type)
        return None

    def _handle_thread_started(self, payload: EventPayload) -> EventDict:
        thread_id = payload.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            self.session_id = thread_id
        return {
            "type": "system",
            "timestamp": self._timestamp(payload),
            "session_id": self.session_id,
            "subtype": "thread_started",
        }

    def _handle_turn_started(self, payload: EventPayload) -> EventDict:
        return {
            "type": "system",
            "timestamp": self._timestamp(payload),
            "session_id": self.session_id,
            "subtype": "turn_started",
        }

    def _handle_turn_completed(self, payload: EventPayload) -> EventDict:
        usage = payload.get("usage") or {}
        input_tokens = self._coerce_int(usage.get("input_tokens"))
        cached_tokens = self._coerce_int(usage.get("cached_input_tokens"))
        output_tokens = self._coerce_int(usage.get("output_tokens"))
        turn_input = input_tokens + cached_tokens
        turn_total = turn_input + output_tokens

        self.tokens_in += turn_input
        self.tokens_out += output_tokens
        self.total_tokens += turn_total

        return {
            "type": "turn_complete",
            "timestamp": self._timestamp(payload),
            "turn_metrics": {
                "input_tokens": turn_input,
                "output_tokens": output_tokens,
                "total_tokens": turn_total,
            },
        }

    def _handle_turn_failed(self, payload: EventPayload) -> EventDict:
        error_payload = payload.get("error") or {}
        message = (
            error_payload.get("message") or payload.get("message") or "unknown error"
        )
        self.last_error = str(message)
        return {
            "type": "error",
            "timestamp": self._timestamp(payload),
            "error_type": "codex",
            "error_message": self.last_error,
        }

    def _handle_stream_error(self, payload: EventPayload) -> EventDict:
        message = payload.get("message") or "unknown error"
        self.last_error = str(message)
        return {
            "type": "error",
            "timestamp": self._timestamp(payload),
            "error_type": "codex",
            "error_message": self.last_error,
        }

    def _handle_item_event(
        self, event_type: str, payload: EventPayload
    ) -> Optional[EventDict]:
        return handle_item_event(
            event_type,
            payload,
            timestamp_fn=self._timestamp,
            truncate_output=self._truncate_output,
            set_last_message=self._update_last_message,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Return cumulative session metrics and last known values."""
        return {
            "session_id": self.session_id,
            "final_message": self.last_message,
            "metrics": {
                "input_tokens": self.tokens_in,
                "output_tokens": self.tokens_out,
                "total_tokens": self.total_tokens,
            },
            "error": self.last_error,
        }

    def _timestamp(self, payload: EventPayload) -> str:
        timestamp = payload.get("timestamp")
        if isinstance(timestamp, str) and timestamp.strip():
            return timestamp
        return datetime.now(timezone.utc).isoformat()

    def _decode_json(self, raw_line: str) -> Optional[EventPayload]:
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug("codex_parser: unable to decode JSON line: %s", raw_line[:160])
            return None
        if not isinstance(payload, dict):
            logger.debug("codex_parser: JSON line did not contain an object")
            return None
        return payload

    def _normalize_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized or None

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _update_last_message(self, message: str) -> None:
        self.last_message = message

    def _truncate_output(self, output: Any) -> Optional[str]:
        if output is None:
            return None
        text = output if isinstance(output, str) else str(output)
        if len(text) > MAX_OUTPUT_PREVIEW:
            return text[:MAX_OUTPUT_PREVIEW]
        return text
