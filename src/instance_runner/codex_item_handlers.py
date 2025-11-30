"""
Handlers for Codex item.* detail events.

These helpers are used by CodexOutputParser to translate item-level Codex events
into the runner's internal event schema.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

EventPayload = Dict[str, Any]
EventDict = Dict[str, Any]
TimestampFn = Callable[[EventPayload], str]
TruncateFn = Callable[[Any], Optional[str]]
SetMessageFn = Callable[[str], None]

__all__ = ["handle_item_event"]

logger = logging.getLogger(__name__)


def handle_item_event(
    event_type: str,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,
    set_last_message: SetMessageFn,
) -> Optional[EventDict]:
    """Dispatch item.* events to the appropriate handler."""
    item = payload.get("item")
    if not isinstance(item, dict):
        return None

    details = item["details"] if isinstance(item.get("details"), dict) else item

    detail_type = details.get("type")
    if not isinstance(detail_type, str):
        return None

    detail_type = detail_type.strip().lower()
    handlers = {
        "agent_message": _agent_message,
        "reasoning": _reasoning,
        "command_execution": _command_execution,
        "file_change": _file_change,
        "mcp_tool_call": _mcp_tool_call,
        "web_search": _web_search,
        "error": _item_error,
    }

    if detail_type == "todo_list":
        return None

    handler = handlers.get(detail_type)
    if handler is None:
        logger.debug("codex_parser: unhandled item details type '%s'", detail_type)
        return None

    return handler(
        event_type,
        details,
        payload,
        timestamp_fn=timestamp_fn,
        truncate_output=truncate_output,
        set_last_message=set_last_message,
    )


def _agent_message(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused but kept for uniform signature
    set_last_message: SetMessageFn,
) -> EventDict:
    text = str(details.get("text") or "")
    if text:
        set_last_message(text)
    return {
        "type": "assistant",
        "timestamp": timestamp_fn(payload),
        "content": text,
    }


def _reasoning(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused but kept for uniform signature
    set_last_message: SetMessageFn,  # unused
) -> EventDict:
    text = str(details.get("text") or "")
    prefixed = f"[reasoning] {text}" if text else "[reasoning]"
    return {
        "type": "assistant",
        "timestamp": timestamp_fn(payload),
        "content": prefixed,
    }


def _command_execution(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,
    set_last_message: SetMessageFn,  # unused
) -> Optional[EventDict]:
    command = str(details.get("command") or "").strip()
    if event_type == "item.started":
        return {
            "type": "tool_use",
            "timestamp": timestamp_fn(payload),
            "tool": "bash",
            "action": "bash",
            "command": command or None,
        }

    if event_type == "item.completed":
        exit_code = details.get("exit_code")
        status = str(details.get("status") or "").lower()
        success = status == "completed" and (exit_code == 0 or exit_code is None)
        output = truncate_output(details.get("aggregated_output"))
        event: Dict[str, Any] = {
            "type": "tool_result",
            "timestamp": timestamp_fn(payload),
            "success": success,
            "exit_code": exit_code,
        }
        if output:
            event["output"] = output
        if command:
            event["command"] = command
        return event

    return None


def _file_change(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused
    set_last_message: SetMessageFn,  # unused
) -> EventDict:
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
        "timestamp": timestamp_fn(payload),
        "tool": "edit",
        "success": success,
        "changes": rendered_changes,
    }


def _mcp_tool_call(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused
    set_last_message: SetMessageFn,  # unused
) -> Optional[EventDict]:
    server = str(details.get("server") or "")
    tool_name = str(details.get("tool") or "")
    arguments = details.get("arguments")

    if event_type == "item.started":
        return {
            "type": "tool_use",
            "timestamp": timestamp_fn(payload),
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
            "timestamp": timestamp_fn(payload),
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


def _web_search(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused
    set_last_message: SetMessageFn,  # unused
) -> Optional[EventDict]:
    query = str(details.get("query") or "")
    if event_type == "item.started":
        return {
            "type": "tool_use",
            "timestamp": timestamp_fn(payload),
            "tool": "web_search",
            "query": query,
        }
    if event_type == "item.completed":
        return {
            "type": "tool_result",
            "timestamp": timestamp_fn(payload),
            "tool": "web_search",
            "success": True,
            "query": query,
        }
    return None


def _item_error(
    event_type: str,
    details: EventPayload,
    payload: EventPayload,
    *,
    timestamp_fn: TimestampFn,
    truncate_output: TruncateFn,  # unused
    set_last_message: SetMessageFn,  # unused
) -> EventDict:
    message = str(details.get("message") or "error")
    return {
        "type": "error",
        "timestamp": timestamp_fn(payload),
        "error_type": "codex",
        "error_message": message,
    }
