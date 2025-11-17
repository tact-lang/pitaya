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

        # Some events can be wrapped in {"id": "...", "msg": {...}}
        if isinstance(payload.get("msg"), dict):
            payload = payload["msg"]

        event_type = payload.get("type")
        if not isinstance(event_type, str):
            return None

        normalized = event_type.strip().lower()
        if not normalized:
            return None

        if normalized == "thread.started":
            return self._handle_thread_started(payload)
        if normalized == "turn.started":
            return self._handle_turn_started(payload)
        if normalized == "turn.completed":
            return self._handle_turn_completed(payload)
        if normalized == "turn.failed":
            return self._handle_turn_failed(payload)
        if normalized.startswith("item."):
            return self._handle_item_event(normalized, payload)
        if normalized == "error":
            return self._handle_stream_error(payload)

        # Unknown event type; log for diagnostics but keep running.
        logger.debug("codex_parser: unrecognized event type '%s'", normalized)
        return None

    def _handle_thread_started(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = payload.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            self.session_id = thread_id
        return {
            "type": "system",
            "timestamp": self._ts(payload),
            "session_id": self.session_id,
            "subtype": "thread_started",
        }

    def _handle_turn_started(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "system",
            "timestamp": self._ts(payload),
            "session_id": self.session_id,
            "subtype": "turn_started",
        }

    def _handle_turn_completed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        cached_tokens = int(usage.get("cached_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        turn_input = input_tokens + cached_tokens
        turn_total = turn_input + output_tokens

        self.tokens_in += turn_input
        self.tokens_out += output_tokens
        self.total_tokens += turn_total

        return {
            "type": "turn_complete",
            "timestamp": self._ts(payload),
            "turn_metrics": {
                "input_tokens": turn_input,
                "output_tokens": output_tokens,
                "total_tokens": turn_total,
            },
        }

    def _handle_turn_failed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        error_payload = payload.get("error") or {}
        message = error_payload.get("message") or payload.get("message") or "unknown error"
        self.last_error = str(message)
        return {
            "type": "error",
            "timestamp": self._ts(payload),
            "error_type": "codex",
            "error_message": self.last_error,
        }

    def _handle_stream_error(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = payload.get("message") or "unknown error"
        self.last_error = str(message)
        return {
            "type": "error",
            "timestamp": self._ts(payload),
            "error_type": "codex",
            "error_message": self.last_error,
        }

    def _handle_item_event(
        self, event_type: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        item = payload.get("item")
        if not isinstance(item, dict):
            return None

        details = item.get("details")
        if not isinstance(details, dict):
            return None

        detail_type = details.get("type")
        if not isinstance(detail_type, str):
            return None

        detail_type = detail_type.strip().lower()
        if detail_type == "agent_message":
            return self._handle_agent_message(details, payload)
        if detail_type == "reasoning":
            return self._handle_reasoning(details, payload)
        if detail_type == "command_execution":
            return self._handle_command_event(event_type, details, payload)
        if detail_type == "file_change":
            return self._handle_file_change(details, payload)
        if detail_type == "mcp_tool_call":
            return self._handle_mcp_tool_event(event_type, details, payload)
        if detail_type == "web_search":
            return self._handle_web_search(event_type, details, payload)
        if detail_type == "error":
            return self._handle_item_error(details, payload)
        if detail_type == "todo_list":
            # Plans/To-do lists are currently ignored.
            return None

        logger.debug("codex_parser: unhandled item details type '%s'", detail_type)
        return None

    def _handle_agent_message(
        self, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = str(details.get("text") or "")
        if text:
            self.last_message = text
        return {
            "type": "assistant",
            "timestamp": self._ts(payload),
            "content": text,
        }

    def _handle_reasoning(
        self, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = str(details.get("text") or "")
        prefixed = f"[reasoning] {text}" if text else "[reasoning]"
        return {
            "type": "assistant",
            "timestamp": self._ts(payload),
            "content": prefixed,
        }

    def _handle_command_event(
        self, event_type: str, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        command = str(details.get("command") or "").strip()
        if event_type == "item.started":
            return {
                "type": "tool_use",
                "timestamp": self._ts(payload),
                "tool": "bash",
                "action": "bash",
                "command": command or None,
            }

        if event_type == "item.completed":
            exit_code = details.get("exit_code")
            status = str(details.get("status") or "").lower()
            success = status == "completed" and (exit_code == 0 or exit_code is None)
            output = details.get("aggregated_output")
            if isinstance(output, str):
                output = output[:500]
            elif output is not None:
                output = str(output)[:500]
            event: Dict[str, Any] = {
                "type": "tool_result",
                "timestamp": self._ts(payload),
                "success": success,
                "exit_code": exit_code,
            }
            if output:
                event["output"] = output
            if command:
                event["command"] = command
            return event

        return None

    def _handle_file_change(
        self, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        changes = details.get("changes") or []
        rendered_changes = []
        for change in changes:
            path = ""
            kind = ""
            if isinstance(change, dict):
                path = str(change.get("path") or "")
                kind = str(change.get("kind") or "")
            rendered_changes.append({"path": path, "kind": kind})
        status = str(details.get("status") or "")
        success = status.lower() == "completed"
        return {
            "type": "tool_result",
            "timestamp": self._ts(payload),
            "tool": "edit",
            "success": success,
            "changes": rendered_changes,
        }

    def _handle_mcp_tool_event(
        self, event_type: str, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        server = str(details.get("server") or "")
        tool_name = str(details.get("tool") or "")
        arguments = details.get("arguments")

        if event_type == "item.started":
            return {
                "type": "tool_use",
                "timestamp": self._ts(payload),
                "tool": "mcp",
                "action": f"{server}:{tool_name}".strip(":"),
                "arguments": arguments,
            }

        if event_type == "item.completed":
            status = str(details.get("status") or "").lower()
            success = status == "completed"
            result = details.get("result")
            error = details.get("error")
            event: Dict[str, Any] = {
                "type": "tool_result",
                "timestamp": self._ts(payload),
                "tool": "mcp",
                "success": success,
                "server": server or None,
                "tool_name": tool_name or None,
            }
            if result is not None:
                event["result"] = result
            if error:
                event["error"] = error
            return event

        return None

    def _handle_web_search(
        self, event_type: str, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        query = str(details.get("query") or "")
        if event_type == "item.started":
            return {
                "type": "tool_use",
                "timestamp": self._ts(payload),
                "tool": "web_search",
                "query": query,
            }
        if event_type == "item.completed":
            return {
                "type": "tool_result",
                "timestamp": self._ts(payload),
                "tool": "web_search",
                "success": True,
                "query": query,
            }
        return None

    def _handle_item_error(
        self, details: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        message = str(details.get("message") or "error")
        return {
            "type": "error",
            "timestamp": self._ts(payload),
            "error_type": "codex",
            "error_message": message,
        }

    def get_summary(self) -> Dict[str, Any]:
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
