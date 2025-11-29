"""Claude parser facade."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .claude_parser_impl import parse_line_impl

logger = logging.getLogger(__name__)


class ClaudeOutputParser:
    """Parses the Anthropic agent's stream-json output format."""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.total_tokens: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost: float = 0.0
        self.turn_count: int = 0
        self.last_message: Optional[str] = None

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        return parse_line_impl(self, line)

    def _parse_tool_use(
        self, item: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool_event = {
            "type": "tool_use",
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "tool_name": item.get("name"),
            "tool_input": item.get("input"),
        }
        # Capture any response headers if present
        if "response_headers" in item:
            tool_event["response_headers"] = item.get("response_headers")
        return tool_event

    def _parse_tool_result(
        self, item: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        result_event = {
            "type": "tool_result",
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "tool_name": item.get("name"),
            "result": item.get("result"),
        }
        if "content" in item:
            result_event["content"] = item.get("content")
        return result_event

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "turn_count": self.turn_count,
        }


__all__ = ["ClaudeOutputParser"]
