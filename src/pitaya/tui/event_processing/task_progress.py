"""Handlers for task.progress events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..models import InstanceDisplay


class TaskProgressHandlers:
    """Task progress handler and helpers."""

    def _handle_task_progress(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self._ensure_progress_instance(iid)
        phase = data.get("phase")
        activity = data.get("activity")
        tool = data.get("tool")

        self._update_task_progress_activity(inst, phase, activity, tool)
        self._apply_progress_usage(inst, data)

        msg = self._build_progress_message(phase, activity, tool)
        if msg:
            self._append_msg(iid, msg)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _ensure_progress_instance(self, instance_id: str) -> InstanceDisplay:
        inst = self.state.current_run.instances.get(instance_id)
        if not inst:
            inst = InstanceDisplay(instance_id=instance_id, strategy_name="")
            self.state.current_run.instances[instance_id] = inst
        return inst

    def _update_task_progress_activity(
        self,
        instance: InstanceDisplay,
        phase: Optional[str],
        activity: Optional[str],
        tool: Optional[str],
    ) -> None:
        if tool:
            instance.last_tool_use = tool
        if activity:
            instance.current_activity = activity
        elif phase:
            friendly = {
                "workspace_preparing": "Preparing workspace...",
                "container_creating": "Creating container...",
                "container_env_preparing": "Preparing container env...",
                "container_env_prepared": "Container env ready",
                "container_created": "Container created",
                "agent_starting": "Starting Agent...",
                "result_collection": "Collecting results...",
                "branch_imported": "Branch imported",
                "no_changes": "No changes",
                "cleanup": "Cleaning up...",
                "assistant": "Agent is thinking...",
                "system": "Agent connected",
                "tool_use": f"Using {tool}" if tool else "Tool use",
            }.get(phase)
            if friendly:
                instance.current_activity = friendly
        instance.last_updated = datetime.now()

    def _apply_progress_usage(
        self, instance: InstanceDisplay, data: Dict[str, Any]
    ) -> None:
        if isinstance(data, dict):
            usage_payload = data.get("usage")
            if isinstance(usage_payload, dict):
                message_id = (
                    data.get("message_id")
                    if isinstance(data.get("message_id"), str)
                    else None
                )
                self._apply_usage_metrics(instance, usage_payload, message_id)

    def _build_progress_message(
        self,
        phase: Optional[str],
        activity: Optional[str],
        tool: Optional[str],
    ) -> Optional[str]:
        msg = None
        if activity:
            msg = f"progress activity={activity}"
        elif phase:
            msg = f"progress phase={phase}"
        if tool:
            msg = (msg + f" tool={tool}") if msg else f"progress tool={tool}"
        return msg


__all__ = ["TaskProgressHandlers"]
