"""Handlers for Codex parser events."""

from __future__ import annotations

from typing import Any, Dict, Optional


def handle_thread_started(parser, payload: Dict[str, Any]) -> Dict[str, Any]:
    parser.session_id = payload.get("thread_id") or payload.get("id")
    return {
        "type": "session_started",
        "session_id": parser.session_id,
        "timestamp": parser._ts(payload),
    }


def handle_turn_started(parser, payload: Dict[str, Any]) -> Dict[str, Any]:
    parser.last_message = None
    return {"type": "turn_started", "timestamp": parser._ts(payload)}


def handle_turn_completed(parser, payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") or payload.get("usage") or {}
    tokens_in = int(metrics.get("input_tokens", 0)) + int(
        metrics.get("cached_input_tokens", 0)
    )
    tokens_out = int(metrics.get("output_tokens", 0))
    parser.tokens_in += tokens_in
    parser.tokens_out += tokens_out
    parser.total_tokens += tokens_in + tokens_out
    if tokens_out:
        completion_text = payload.get("completion", {}).get("text")
        if completion_text:
            parser.last_message = completion_text
    return {
        "type": "turn_complete",
        "timestamp": parser._ts(payload),
        "metrics": metrics,
        "completion": payload.get("completion"),
    }


def handle_turn_failed(parser, payload: Dict[str, Any]) -> Dict[str, Any]:
    parser.last_error = payload.get("error", {}).get("message")
    return {
        "type": "turn_failed",
        "timestamp": parser._ts(payload),
        "error": payload.get("error"),
    }


def handle_stream_error(parser, payload: Dict[str, Any]) -> Dict[str, Any]:
    parser.last_error = payload.get("error", {}).get("message")
    return {
        "type": "error",
        "timestamp": parser._ts(payload),
        "error": payload.get("error"),
    }


def handle_item_event(
    parser, normalized: str, payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    item = payload.get("item", {}) or {}
    details = item.get("details", {}) or {}
    item_type = (item.get("type") or details.get("type") or "").lower()
    if details:
        merged = item.copy()
        merged.update(details)
        item = merged

    if item_type in ("message", "agent_message") and item.get("role") in (
        None,
        "assistant",
    ):
        return handle_agent_message(parser, item, payload)
    if item_type == "reasoning":
        return handle_reasoning(parser, item, payload)
    if item_type in ("command", "command_execution"):
        return handle_command_event(parser, item, payload)
    if item_type == "file_change":
        return handle_file_change(parser, item, payload)
    if item_type == "mcp_tool":
        return handle_mcp_tool_event(parser, item, payload)
    if item_type == "web_search":
        return handle_web_search(parser, item, payload)
    if item_type == "error":
        return handle_item_error(parser, item, payload)
    return None


def handle_agent_message(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    content = item.get("content") or {}
    text = (
        content.get("text") or item.get("text") or item.get("details", {}).get("text")
    )
    parser.last_message = text or parser.last_message
    tokens_in = int(item.get("input_tokens", 0))
    tokens_out = int(item.get("output_tokens", 0))
    parser.tokens_in += tokens_in
    parser.tokens_out += tokens_out
    parser.total_tokens += tokens_in + tokens_out
    event = {
        "type": "assistant",
        "timestamp": parser._ts(payload),
        "content": text,
        "message_id": item.get("id"),
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        },
    }
    if parser.last_error:
        event["previous_error"] = parser.last_error
    return event


def handle_reasoning(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    text = (
        item.get("content", {}).get("text")
        or item.get("text")
        or item.get("details", {}).get("text")
        or ""
    )
    return {
        "type": "assistant",
        "timestamp": parser._ts(payload),
        "content": text,
    }


def handle_command_event(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    status = item.get("status") or item.get("state")
    if status and status.lower() == "completed":
        return {
            "type": "tool_result",
            "timestamp": parser._ts(payload),
            "command": item.get("command") or item.get("input"),
            "result": item.get("aggregated_output") or item.get("output"),
            "exit_code": item.get("exit_code"),
        }

    event = {
        "type": "tool_use",
        "timestamp": parser._ts(payload),
        "tool_name": item.get("name") or "command_execution",
        "tool_input": item.get("input") or item.get("command"),
    }
    if item.get("id"):
        event["message_id"] = item.get("id")
    if item.get("display_name"):
        event["display_name"] = item.get("display_name")
    return event


def handle_file_change(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "file_change",
        "timestamp": parser._ts(payload),
        "path": item.get("path"),
        "change_type": item.get("change_type"),
        "summary": item.get("summary"),
    }


def handle_mcp_tool_event(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "timestamp": parser._ts(payload),
        "tool_name": item.get("tool_name") or "mcp_tool",
        "tool_input": item.get("arguments"),
        "tool_call_id": item.get("id"),
        "metadata": item.get("metadata", {}),
    }


def handle_web_search(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "timestamp": parser._ts(payload),
        "tool_name": "web_search",
        "tool_input": item.get("query"),
        "metadata": item.get("metadata", {}),
    }


def handle_item_error(
    parser, item: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    parser.last_error = item.get("message")
    return {
        "type": "error",
        "timestamp": parser._ts(payload),
        "error": item,
    }


__all__ = [
    "handle_thread_started",
    "handle_turn_started",
    "handle_turn_completed",
    "handle_turn_failed",
    "handle_stream_error",
    "handle_item_event",
]
