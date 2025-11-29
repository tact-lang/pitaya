"""Implementation helpers for Claude parser."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def parse_line_impl(parser, line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
        return None

    event_type = data.get("type")
    if not event_type:
        return None

    usage_payload: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None

    if event_type == "assistant" and "message" in data:
        message = data["message"]
        if "usage" in message:
            usage = message["usage"]
            message_id = message.get("id") if isinstance(message, dict) else None
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
            cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            tokens_used = input_tokens + cache_create + cache_read + output_tokens
            parser.total_tokens += tokens_used
            parser.input_tokens += input_tokens + cache_create + cache_read
            parser.output_tokens += output_tokens
            usage_payload = {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_tokens,
            }

        if isinstance(message.get("content"), list):
            text_parts = []
            for item in message["content"]:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "tool_use":
                    tool_event = parser._parse_tool_use(item, data)
                    if usage_payload:
                        tool_event["usage"] = usage_payload
                        if message_id:
                            tool_event["message_id"] = message_id
                    return tool_event
            if text_parts:
                parser.last_message = " ".join(text_parts)

    elif event_type == "user" and "message" in data:
        message = data["message"]
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "tool_result":
                    return parser._parse_tool_result(item, data)

    event = {
        "type": event_type,
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }
    if event_type == "assistant" and usage_payload:
        event["usage"] = usage_payload
        if message_id:
            event["message_id"] = message_id
    if event_type == "assistant" and parser.last_message:
        try:
            snippet = (
                parser.last_message
                if len(parser.last_message) <= 4000
                else (parser.last_message[:4000] + "…")
            )
        except Exception:
            snippet = parser.last_message
        event["content"] = snippet

    if event_type == "system" and data.get("subtype") == "init":
        parser.session_id = data.get("session_id")
        event["session_id"] = parser.session_id
        return event

    if event_type == "result":
        event["session_id"] = parser.session_id or data.get("session_id")
        event["final_message"] = data.get("result", parser.last_message)

        total_tokens = parser.total_tokens
        input_tokens = 0
        output_tokens = 0
        if "usage" in data:
            usage = data["usage"]
            input_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            parser.input_tokens = int(input_tokens)
            parser.output_tokens = int(output_tokens)

        event["metrics"] = {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": data.get("total_cost_usd", parser.total_cost),
            "turn_count": data.get("num_turns", parser.turn_count),
            "duration_ms": data.get("duration_ms", 0),
            "duration_api_ms": data.get("duration_api_ms", 0),
        }

        if "total_cost_usd" in data:
            parser.total_cost = data["total_cost_usd"]
        if "num_turns" in data:
            parser.turn_count = data["num_turns"]
        if total_tokens > 0:
            parser.total_tokens = total_tokens
            if not parser.input_tokens and input_tokens:
                parser.input_tokens = int(input_tokens)
            if not parser.output_tokens and output_tokens:
                parser.output_tokens = int(output_tokens)

        return event

    if event_type == "session_started":
        parser.session_id = data.get("session_id")
        event["session_id"] = parser.session_id
        return event

    if event_type == "turn_started":
        parser.turn_count += 1
        event["turn_number"] = parser.turn_count
        return event

    if event_type == "turn_complete":
        metrics = data.get("metrics", {})
        if metrics:
            tokens = metrics.get("tokens_used", 0)
            cost = metrics.get("cost_usd", 0.0)
            parser.total_tokens += int(tokens)
            parser.output_tokens += int(tokens)
            try:
                parser.total_cost += float(cost)
            except Exception:
                pass
            event["metrics"] = metrics
        return event

    if event_type == "message_delta":
        delta = data.get("delta", {})
        if isinstance(delta, dict):
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    parser.last_message = (parser.last_message or "") + text
                    event["content_delta"] = text
            elif delta.get("type") == "input_json_delta":
                event["input_json_delta"] = delta.get("partial_json", {})
        return event

    if event_type == "completion":
        event["completion_reason"] = data.get("reason")
        return event

    return None


__all__ = ["parse_line_impl"]
