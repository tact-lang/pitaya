"""
Parser for Codex CLI JSONL output.

This parser is tolerant to variations in Codex event naming. It maps Codex
events into the internal event set used by the runner (assistant/tool_use/
tool_result/result) and accumulates basic metrics (tokens in/out/total).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CodexOutputParser:
    """Parses Codex CLI JSONL events and derives a compact summary."""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.last_message: Optional[str] = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.total_tokens: int = 0
        self._seen_shutdown: bool = False

    def _ts(self, src: Dict[str, Any]) -> str:
        return src.get("timestamp") or datetime.now(timezone.utc).isoformat()

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        line = (line or "").strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("codex_parser: non-JSON line ignored: %s", line[:160])
            return None

        # Many Codex exec events are wrapped as {"id": "..", "msg": {"type": "...", ...}}
        data = obj
        if isinstance(obj.get("msg"), dict):
            data = obj.get("msg", {})

        et = str(data.get("type") or data.get("event") or "").strip().lower()
        if not et:
            return None

        # Session / config events
        if et in {"session_configured", "session_started", "session"}:
            sid = data.get("session_id") or obj.get("data", {}).get("session_id")
            if isinstance(sid, str) and sid:
                self.session_id = sid
            return {
                "type": "system",
                "timestamp": self._ts(obj),
                "session_id": self.session_id,
                "subtype": "init",
            }

        # Token counts
        if et in {"token_count", "tokens", "usage"}:
            tin = int(data.get("input_tokens", 0) or data.get("in", 0) or 0)
            tout = int(data.get("output_tokens", 0) or data.get("out", 0) or 0)
            # Include cached_input_tokens and reasoning_output_tokens if present
            try:
                tin += int(data.get("cached_input_tokens", 0) or 0)
            except Exception:
                pass
            try:
                tout += int(data.get("reasoning_output_tokens", 0) or 0)
            except Exception:
                pass
            tt = int(data.get("total_tokens", tin + tout) or (tin + tout))
            # Use maxima to avoid double-add if multiple reports roll in
            self.tokens_in = max(self.tokens_in, tin)
            self.tokens_out = max(self.tokens_out, tout)
            self.total_tokens = max(self.total_tokens, tt)
            return {
                "type": "turn_complete",
                "timestamp": self._ts(obj),
                "turn_metrics": {
                    "tokens": tt,
                    "total_tokens": self.total_tokens,
                },
            }

        # Agent/assistant messages
        if et in {"agent_message", "assistant_message", "assistant", "message"}:
            msg = (
                data.get("message")
                or obj.get("message")
                or obj.get("text")
                or obj.get("data", {}).get("message")
                or ""
            )
            if isinstance(msg, dict):
                msg = msg.get("text", "")
            if not isinstance(msg, str):
                msg = str(msg)
            self.last_message = msg
            return {
                "type": "assistant",
                "timestamp": self._ts(obj),
                "content": msg,
            }

        # Command execution (map to tool_use/tool_result)
        if et in {"exec_command_start", "exec_command_begin", "tool_start"}:
            cmd = data.get("command") or obj.get("command") or obj.get("cmd")
            if isinstance(cmd, list):
                try:
                    cmd = " ".join(str(c) for c in cmd)
                except Exception:
                    cmd = str(cmd)
            return {
                "type": "tool_use",
                "timestamp": self._ts(obj),
                "tool": "bash",
                "action": "bash",
                "command": str(cmd) if cmd else None,
            }

        if et in {"exec_command_end", "tool_end"}:
            rc = data.get("exit_code")
            success = (rc == 0) if rc is not None else bool(data.get("success", True))
            out = data.get("output") or data.get("stdout")
            err = data.get("error") or data.get("stderr")
            event: Dict[str, Any] = {
                "type": "tool_result",
                "timestamp": self._ts(obj),
                "success": success if isinstance(success, bool) else True,
            }
            if rc is not None:
                event["exit_code"] = rc
            if isinstance(out, str) and out:
                event["output"] = out[:500]
            if isinstance(err, str) and err:
                event["error"] = err[:500]
            return event

        # Patch / edit operations
        if et in {"patch_apply_start", "file_edit_start", "write_start"}:
            path = data.get("file") or obj.get("file") or obj.get("path")
            return {
                "type": "tool_use",
                "timestamp": self._ts(obj),
                "tool": "Edit",
                "action": "edit",
                "file_path": path,
            }

        if et in {"patch_apply_end", "file_edit_end", "write_end"}:
            ok = bool(data.get("success", True))
            err = data.get("error") or data.get("message")
            return {
                "type": "tool_result",
                "timestamp": self._ts(obj),
                "success": ok,
                **({"error": str(err)[:500]} if err else {}),
            }

        # Task completion
        if et in {"task_complete", "complete", "shutdown_complete"}:
            self._seen_shutdown = True
            # Prefer explicit last_agent_message, else last seen assistant message
            final = (
                data.get("last_agent_message")
                or data.get("message")
                or self.last_message
            )
            if not isinstance(final, str) and final is not None:
                final = str(final)
            # Update tokens from payload if present
            tin = int(data.get("input_tokens", self.tokens_in) or self.tokens_in)
            tout = int(data.get("output_tokens", self.tokens_out) or self.tokens_out)
            eff_in = max(self.tokens_in, tin)
            eff_out = max(self.tokens_out, tout)
            tt = int(data.get("total_tokens", eff_in + eff_out) or (eff_in + eff_out))
            eff_total = max(self.total_tokens, tt, eff_in + eff_out)
            self.tokens_in, self.tokens_out, self.total_tokens = (
                eff_in,
                eff_out,
                eff_total,
            )
            return {
                "type": "result",
                "timestamp": self._ts(obj),
                "session_id": self.session_id,
                "final_message": final,
                "metrics": {
                    "input_tokens": self.tokens_in,
                    "output_tokens": self.tokens_out,
                    "total_tokens": self.total_tokens,
                },
            }

        # Error pass-through
        if et == "error":
            emsg = data.get("error") or data.get("message") or obj.get("detail")
            return {
                "type": "error",
                "timestamp": self._ts(obj),
                "error_type": "codex",
                "error_message": str(emsg) if emsg else "unknown",
            }

        # Unrecognized -> minimal pass-through as assistant update for visibility
        try:
            snippet = obj.get("message") or obj.get("status") or json.dumps(obj)[:400]
        except Exception:
            snippet = ""
        return {
            "type": "assistant",
            "timestamp": self._ts(obj),
            "content": str(snippet) if snippet else f"[{et}]",
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "final_message": self.last_message,
            "metrics": {
                "input_tokens": self.tokens_in,
                "output_tokens": self.tokens_out,
                "total_tokens": self.total_tokens or (self.tokens_in + self.tokens_out),
            },
        }
