"""
Parser for the Anthropic agent stream-json output format.

Extracts events, metrics, and session information from the tool's structured output.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ClaudeOutputParser:
    """Parses the Anthropic agent's stream-json output format."""

    def __init__(self) -> None:
        """Initialize parser state."""
        self.session_id: Optional[str] = None
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.turn_count: int = 0
        self.last_message: Optional[str] = None

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line of stream-json output.

        Args:
            line: JSON line from agent output

        Returns:
            Parsed event dict or None if not a valid event
        """
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
            return None

        # Extract event type
        event_type = data.get("type")
        if not event_type:
            return None

        # For assistant messages, check for tool use and capture text
        if event_type == "assistant" and "message" in data:
            message = data["message"]

            # Extract usage metrics from assistant messages
            if "usage" in message:
                usage = message["usage"]
                # Add up all token types
                tokens_used = (
                    usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                )
                self.total_tokens += tokens_used
                # Note: assistant messages don't have cost, only the final result does

            if isinstance(message.get("content"), list):
                # Extract text content for last_message
                text_parts = []
                for item in message["content"]:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_use":
                        return self._parse_tool_use(item, data)

                # Update last message if we have text
                if text_parts:
                    self.last_message = " ".join(text_parts)

        # For user messages, check for tool results
        elif event_type == "user" and "message" in data:
            message = data["message"]
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "tool_result":
                        return self._parse_tool_result(item, data)

        # Common event data
        event = {
            "type": event_type,
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        # Surface assistant text content in-stream for diagnostics/logging
        # This does not affect canonical events; it is only written to runner.jsonl
        if event_type == "assistant" and self.last_message:
            # Keep a concise but informative snapshot; avoid huge payloads
            try:
                snippet = (
                    self.last_message
                    if len(self.last_message) <= 4000
                    else (self.last_message[:4000] + "…")
                )
            except Exception:
                snippet = self.last_message
            event["content"] = snippet

        # Parse based on event type
        if event_type == "system" and data.get("subtype") == "init":
            # This is the session initialization
            self.session_id = data.get("session_id")
            event["session_id"] = self.session_id
            return event

        elif event_type == "result":
            # This is the final result - extract metrics from the SDK message
            event["session_id"] = self.session_id or data.get("session_id")
            event["final_message"] = data.get("result", self.last_message)

            # Extract total tokens from usage if provided
            total_tokens = self.total_tokens
            input_tokens = 0
            output_tokens = 0

            if "usage" in data:
                usage = data["usage"]
                # Track input and output separately
                input_tokens = (
                    usage.get("input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                )
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = input_tokens + output_tokens

            # Extract metrics from the result message (SDK format)
            event["metrics"] = {
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": data.get("total_cost_usd", self.total_cost),
                "turn_count": data.get("num_turns", self.turn_count),
                "duration_ms": data.get("duration_ms", 0),
                "duration_api_ms": data.get("duration_api_ms", 0),
            }

            # Update our tracking with SDK values if provided
            if "total_cost_usd" in data:
                self.total_cost = data["total_cost_usd"]
            if "num_turns" in data:
                self.turn_count = data["num_turns"]
            if total_tokens > 0:
                self.total_tokens = total_tokens

            return event

        elif event_type == "session_started":
            self.session_id = data.get("session_id")
            event["session_id"] = self.session_id
            return event

        elif event_type == "turn_started":
            self.turn_count += 1
            event["turn_number"] = self.turn_count
            return event

        elif event_type == "turn_complete":
            # Extract turn metrics
            metrics = data.get("metrics", {})
            if metrics:
                tokens = metrics.get("tokens_used", 0)
                cost = metrics.get("cost_usd", 0.0)
                self.total_tokens += tokens
                self.total_cost += cost

                event["turn_metrics"] = {
                    "tokens": tokens,
                    "cost": cost,
                    "total_tokens": self.total_tokens,
                    "total_cost": self.total_cost,
                }
            return event

        elif event_type == "tool_use":
            # Tool usage events (file edits, bash commands, etc.)
            tool_name = data.get("tool", {}).get("name")
            tool_params = data.get("tool", {}).get("parameters", {})

            event["tool"] = tool_name
            event["parameters"] = tool_params

            # Extract key information based on tool
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

        elif event_type == "tool_result":
            # Result from tool execution
            success = data.get("success", True)
            output = data.get("output", "")
            error = data.get("error")

            event["success"] = success
            if error:
                event["error"] = error
            if output:
                event["output"] = output[:500]  # Truncate long outputs

            return event

        elif event_type == "message":
            # Assistant messages
            content = data.get("content", "")
            self.last_message = content
            event["content"] = content
            return event

        elif event_type == "error":
            # Error events
            error_type = data.get("error_type", "unknown")
            error_message = data.get("error_message", "")

            event["error_type"] = error_type
            event["error_message"] = error_message
            return event

        elif event_type == "session_complete":
            # Session completion
            event["session_id"] = self.session_id
            event["final_message"] = data.get("result", self.last_message)

            # Extract metrics from SDK data if available
            event["metrics"] = {
                "total_tokens": self.total_tokens,
                "total_cost": data.get("total_cost_usd", self.total_cost),
                "turn_count": data.get("num_turns", self.turn_count),
                "duration_ms": data.get("duration_ms", 0),
                "duration_api_ms": data.get("duration_api_ms", 0),
            }

            # Update tracking if SDK provides values
            if "total_cost_usd" in data:
                self.total_cost = data["total_cost_usd"]
            if "num_turns" in data:
                self.turn_count = data["num_turns"]

            return event

        elif event_type == "task_update":
            # Task list updates
            tasks = data.get("tasks", [])
            event["tasks"] = tasks
            event["task_count"] = len(tasks)

            # Count by status
            status_counts: Dict[str, int] = {}
            for task in tasks:
                status = task.get("status", "pending")
                status_counts[status] = status_counts.get(status, 0) + 1
            event["status_counts"] = status_counts

            return event

        elif event_type == "cost_limit_warning":
            # Cost limit warnings
            current_cost = data.get("current_cost", 0.0)
            limit = data.get("limit", 0.0)
            percentage = data.get("percentage", 0.0)

            event["current_cost"] = current_cost
            event["limit"] = limit
            event["percentage"] = percentage
            return event

        else:
            # Unknown event type, pass through key data
            for key in ["status", "progress", "data"]:
                if key in data:
                    event[key] = data[key]
            return event

    def _parse_tool_use(
        self, tool_item: Dict[str, Any], parent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a tool use item from assistant message."""
        tool_name = tool_item.get("name", "")
        tool_params = tool_item.get("input", {})

        event = {
            "type": "tool_use",
            "timestamp": parent_data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "tool": tool_name,
            "parameters": tool_params,
        }

        # Extract key information based on tool
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

    def _parse_tool_result(
        self, result_item: Dict[str, Any], parent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a tool result from user message."""
        content = result_item.get("content", "")

        event = {
            "type": "tool_result",
            "timestamp": parent_data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "success": not result_item.get("is_error", False),
            "output": content[:500] if content else "",  # Truncate long outputs
        }

        return event

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of parsed session.

        Returns:
            Summary dict with metrics and session info
        """
        return {
            "session_id": self.session_id,
            "final_message": self.last_message,
            "metrics": {
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "turn_count": self.turn_count,
            },
        }
