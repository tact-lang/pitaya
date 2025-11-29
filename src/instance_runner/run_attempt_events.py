"""Event emitter helper for run attempts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict


def make_emit_event(
    event_callback: Callable[[Dict[str, Any]], None] | None,
    instance_id: str,
    final_message_ref: Dict[str, Any],
    metrics_ref: Dict[str, Any],
) -> Callable[..., None]:
    def emit_event(
        event_type_or_payload: Any, data: Dict[str, Any] | None = None
    ) -> None:
        if not event_callback:
            return

        # Support both legacy signature emit_event(type, data) and single-payload callbacks
        if data is None and isinstance(event_type_or_payload, dict):
            payload = dict(event_type_or_payload)
            payload.setdefault("instance_id", instance_id)
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        else:
            payload = {
                "type": event_type_or_payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": instance_id,
                "data": data or {},
            }

        try:
            if payload.get("type", "").endswith("agent_result"):
                fm = (payload.get("data") or {}).get("final_message")
                if isinstance(fm, str) and fm:
                    final_message_ref["value"] = fm
                m = (payload.get("data") or {}).get("metrics") or {}
                if isinstance(m, dict) and m:
                    metrics_ref.update(m)
        except Exception:
            pass

        event_callback(payload)

    return emit_event


__all__ = ["make_emit_event"]
