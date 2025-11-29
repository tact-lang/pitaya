"""Helpers for instance event handling and progress mapping."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def strategy_execution_id(orchestrator, instance_id: str) -> Optional[str]:
    for sid, strat in orchestrator.state_manager.current_state.strategies.items():
        if instance_id in strat.instance_ids:
            return sid
    return None


def progress_payload(event: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    et = str(event.get("type", ""))
    data = event.get("data", {}) or {}
    phase = None
    activity = None
    tool = None
    if et == "instance.workspace_preparing":
        phase, activity = "workspace_preparing", "Preparing workspace..."
    elif et in {
        "instance.container_creating",
        "instance.container_create_call",
        "instance.container_create_attempt",
        "instance.container_image_check",
    }:
        phase, activity = "container_creating", "Creating container..."
    elif et == "instance.container_env_preparing":
        phase, activity = "container_env_preparing", "Preparing container env..."
    elif et == "instance.startup_waiting":
        phase, activity = "startup_waiting", "Waiting for startup slot..."
    elif et == "instance.container_env_prepared":
        phase, activity = "container_env_prepared", "Container env ready"
    elif et == "instance.container_created":
        phase, activity = "container_created", "Container created"
    elif et == "instance.container_adopted":
        phase, activity = "container_created", "Container adopted"
    elif et == "instance.agent_starting":
        phase, activity = "agent_starting", "Starting Agent..."
    elif et == "instance.result_collection_started":
        phase, activity = "result_collection", "Collecting results..."
    elif et == "instance.branch_imported":
        phase, activity = "branch_imported", f"Imported branch {data.get('branch_name','')}"
    elif et == "instance.no_changes":
        phase, activity = "no_changes", "No changes"
    elif et == "instance.workspace_cleaned":
        phase, activity = "cleanup", "Workspace cleaned"
    elif et == "instance.agent_tool_use":
        phase, tool = "tool_use", (data.get("tool") or data.get("data", {}).get("tool"))
        activity = f"Using {tool}" if tool else "Tool use"
    elif et == "instance.agent_assistant":
        phase, activity = "assistant", "Agent is thinking..."
    elif et == "instance.agent_system":
        phase, activity = "system", "Agent connected"
    extras: Dict[str, Any] = {}
    usage_payload = data.get("usage")
    if isinstance(usage_payload, dict):
        extras["usage"] = usage_payload
    message_id_val = data.get("message_id")
    if isinstance(message_id_val, str):
        extras["message_id"] = message_id_val
    if et == "instance.agent_turn_complete":
        phase = "assistant"
        activity = None
        tm = data.get("turn_metrics", {}) if isinstance(data, dict) else {}
        if isinstance(tm, dict):
            usage = {
                "tokens": int(tm.get("tokens", 0) or 0),
                "total_tokens": int(tm.get("total_tokens", 0) or 0),
            }
            if "input_tokens" in tm:
                usage["input_tokens"] = int(tm.get("input_tokens", 0) or 0)
            if "output_tokens" in tm:
                usage["output_tokens"] = int(tm.get("output_tokens", 0) or 0)
            extras.setdefault("usage", usage)
    payload: Dict[str, Any] = {}
    if activity:
        payload["activity"] = activity
    if tool:
        payload["tool"] = tool
    payload.update(extras)
    return phase, payload


def build_event_callback(orchestrator, instance_id: str, task_key: Optional[str]):
    def _callback(event: Dict[str, Any]) -> None:
        data = event.get("data", {})
        session_id = data.get("session_id")
        if session_id:
            try:
                orchestrator.state_manager.update_instance_session_id(instance_id, session_id)
            except Exception:
                pass
        try:
            orchestrator.event_bus.emit(
                event_type=event.get("type", "instance.event"),
                data=data,
                instance_id=instance_id,
            )
        except Exception:
            pass

        if not task_key:
            return

        try:
            phase, payload = progress_payload(event)
            if not phase:
                return
            orchestrator.event_bus.emit_canonical(
                type="task.progress",
                run_id=orchestrator.state_manager.current_state.run_id,
                strategy_execution_id=strategy_execution_id(orchestrator, instance_id),
                key=task_key,
                payload={
                    "key": task_key,
                    "instance_id": instance_id,
                    "phase": phase,
                    **payload,
                },
            )
        except Exception:
            pass

    return _callback


def truncate_final_message(
    full_msg: str, max_bytes: int, run_logs_dir, instance_id: str
) -> tuple[str, bool, str]:
    msg_bytes = full_msg.encode("utf-8", errors="ignore")
    if max_bytes <= 0 or len(msg_bytes) <= max_bytes:
        return full_msg, False, ""
    truncated = msg_bytes[:max_bytes].decode("utf-8", errors="ignore")
    try:
        dest_dir = run_logs_dir / "final_messages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{instance_id}.txt"
        with open(dest_dir / fname, "w", encoding="utf-8", errors="ignore") as fh:
            fh.write(full_msg)
        return truncated, True, f"final_messages/{fname}"
    except Exception:
        return truncated, True, ""


async def heartbeat_monitor(orchestrator, instance_id: str, interval: float = 2.0) -> None:
    """Emit periodic debug logs about last observed event."""
    try:
        while True:
            last_type = None
            last_ts = None
            if orchestrator.event_bus:
                try:
                    for ev in reversed(orchestrator.event_bus.events):
                        if ev.get("instance_id") == instance_id:
                            last_type = ev.get("type")
                            last_ts = ev.get("timestamp")
                            break
                except Exception:
                    pass
            if last_type and last_ts:
                logger.debug(
                    "Heartbeat: instance %s last_event=%s at %s",
                    instance_id,
                    last_type,
                    last_ts,
                )
            else:
                logger.debug("Heartbeat: instance %s awaiting first event", instance_id)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        return


def map_error_type(result, info) -> str:
    mapping = {
        "docker": "docker",
        "git": "git",
        "timeout": "timeout",
        "auth": "auth",
        "session_corrupted": "session_corrupted",
        "claude": "api",
        "orchestration": "unknown",
        "validation": "unknown",
        "system": "unknown",
        "unexpected": "unknown",
    }
    etype = (result.error_type or "unknown").lower()
    mapped = mapping.get(
        etype,
        (
            etype
            if etype
            in {
                "docker",
                "api",
                "network",
                "git",
                "timeout",
                "session_corrupted",
                "auth",
                "unknown",
            }
            else "unknown"
        ),
    )
    try:
        if (info.metadata or {}).get("network_egress") == "offline" and mapped != "canceled":
            mapped = "network"
    except Exception:
        pass
    return mapped
