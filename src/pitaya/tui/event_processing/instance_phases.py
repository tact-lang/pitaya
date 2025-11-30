"""Phase and lifecycle-adjacent instance handlers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..models import InstanceDisplay, InstanceStatus
from .logging_config import logger


class InstancePhaseHandlers:
    """Instance phase transitions, cancellations, and registration helpers."""

    def _handle_state_instance_registered(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        data = event.get("data", {})
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id:
            return

        inst = self.state.current_run.instances.get(instance_id)
        if not inst:
            inst = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy_name", "unknown"),
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name"),
            )
            self.state.current_run.instances[instance_id] = inst
        else:
            if inst.strategy_name == "unknown" and data.get("strategy_name"):
                inst.strategy_name = data.get("strategy_name")
            if not inst.branch_name and data.get("branch_name"):
                inst.branch_name = data.get("branch_name")

        self.state.current_run.total_instances = len(self.state.current_run.instances)

        strategy_name = data.get("strategy_name")
        if strategy_name:
            for strategy in self.state.current_run.strategies.values():
                if (
                    strategy.strategy_name == strategy_name
                    and instance_id not in strategy.instance_ids
                ):
                    strategy.instance_ids.append(instance_id)
                    strategy.total_instances += 1
                    break

    def _handle_instance_workspace_preparing(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        previous_activity = instance.current_activity
        instance.current_activity = "Preparing workspace..."
        instance.last_updated = datetime.now()
        logger.info(
            f"Instance {instance_id}: '{previous_activity}' -> 'Preparing workspace...'"
        )

    def _handle_instance_container_creating(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Creating container..."
        instance.last_updated = datetime.now()

    def _handle_instance_container_env_preparing(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Preparing container env..."
        instance.last_updated = datetime.now()

    def _handle_instance_container_env_prepared(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Container env ready"
        instance.last_updated = datetime.now()

    def _handle_instance_container_created(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Container created"
        instance.last_updated = datetime.now()

    def _handle_instance_agent_starting(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Starting Agent..."
        instance.last_updated = datetime.now()

    def _handle_instance_result_collection(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        instance = self.state.current_run.instances[instance_id]
        instance.current_activity = "Collecting results..."
        instance.last_updated = datetime.now()

    def _handle_instance_canceled(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return
        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]
        instance.status = InstanceStatus.INTERRUPTED
        instance.error = data.get("error") or "canceled"
        instance.error_type = data.get("error_type") or "canceled"
        instance.last_updated = datetime.now()
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

    def _handle_state_instance_updated(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return
        data = event.get("data", {})
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return
        new_state = data.get("new_state")
        instance = self.state.current_run.instances[instance_id]
        if new_state == "interrupted":
            instance.status = InstanceStatus.INTERRUPTED
            instance.last_updated = datetime.now()
            self.state.current_run.active_instances = max(
                0, self.state.current_run.active_instances - 1
            )


__all__ = ["InstancePhaseHandlers"]
