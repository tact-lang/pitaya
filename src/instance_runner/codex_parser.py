"""Parser for Codex ThreadEvent JSONL output."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .codex_parser_handlers import (
    handle_item_event,
    handle_stream_error,
    handle_thread_started,
    handle_turn_completed,
    handle_turn_failed,
    handle_turn_started,
)

logger = logging.getLogger(__name__)


class CodexOutputParser:
    """Parse Codex ThreadEvents and derive summary metrics."""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.last_message: Optional[str] = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.total_tokens: int = 0
        self.last_error: Optional[str] = None

    def _ts(self, payload: Dict[str, Any]) -> str:
        return payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        line = (line or "").strip()
        if not line:
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("codex_parser: unable to decode JSON line: %s", line[:160])
            return None

        if isinstance(payload.get("msg"), dict):
            payload = payload["msg"]

        event_type = payload.get("type")
        if not isinstance(event_type, str):
            return None

        normalized = event_type.strip().lower()
        if not normalized:
            return None

        if normalized == "thread.started":
            return handle_thread_started(self, payload)
        if normalized == "turn.started":
            return handle_turn_started(self, payload)
        if normalized == "turn.completed":
            return handle_turn_completed(self, payload)
        if normalized == "turn.failed":
            return handle_turn_failed(self, payload)
        if normalized.startswith("item."):
            return handle_item_event(self, normalized, payload)
        if normalized == "error":
            return handle_stream_error(self, payload)

        logger.debug("codex_parser: unrecognized event type '%s'", normalized)
        return None

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "last_message": self.last_message,
            "final_message": self.last_message,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "last_error": self.last_error,
            "error": self.last_error,
            "metrics": {
                "input_tokens": self.tokens_in,
                "output_tokens": self.tokens_out,
                "total_tokens": self.total_tokens,
            },
        }


__all__ = ["CodexOutputParser"]
