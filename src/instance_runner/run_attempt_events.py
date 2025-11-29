"""Event emitter helper for run attempts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict


def make_emit_event(
    event_callback: Callable[[Dict[str, Any]], None] | None,
    instance_id: str,
    final_message_ref: Dict[str, Any],
    metrics_ref: Dict[str, Any],
) -> Callable[[str, Dict[str, Any]], None]:
    def emit_event(event_type: str, data: Dict[str, Any]) -> None:
        if not event_callback:
            return
        try:
            if event_type.endswith("agent_result"):
                fm = data.get("final_message")
                if isinstance(fm, str) and fm:
                    final_message_ref["value"] = fm
                m = data.get("metrics") or {}
                if isinstance(m, dict) and m:
                    metrics_ref.update(m)
        except Exception:
            pass
        event_callback(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": instance_id,
                "data": data,
            }
        )

    return emit_event


__all__ = ["make_emit_event"]
