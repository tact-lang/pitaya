"""Lifecycle handlers for instance events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..models import InstanceDisplay, InstanceStatus
from .logging_config import logger


class InstanceLifecycleHandlers:
    """Instance lifecycle handlers (queued, started, completed, failed, progress)."""

    def _handle_instance_queued(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        data = event.get("data", {})
        instance_id = event.get("instance_id") or data.get("instance_id")
        if not instance_id:
            logger.warning(f"No instance_id in event: {event}")
            return

        if instance_id not in self.state.current_run.instances:
            instance = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy", "unknown"),
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name"),
                last_updated=self._parse_timestamp(event.get("timestamp"))
                or datetime.now(),
            )
            self.state.current_run.instances[instance_id] = instance
        self.state.current_run.total_instances = len(self.state.current_run.instances)

        strategy_name = data.get("strategy", "unknown")
        for strategy in self.state.current_run.strategies.values():
            if strategy.strategy_name == strategy_name:
                strategy.instance_ids.append(instance_id)
                strategy.total_instances += 1
                logger.info(
                    f"Added instance {instance_id} to strategy {strategy.strategy_id}"
                )
                break

        logger.info(
            f"Created instance {instance_id} from queued event, total instances: {len(self.state.current_run.instances)}"
        )

    def _handle_instance_started(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            logger.warning("Got instance.started but no current_run")
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.started without instance_id")
            return

        if instance_id not in self.state.current_run.instances:
            logger.info(f"Creating instance {instance_id} from started event")
            data = event.get("data", {})
            instance = InstanceDisplay(
                instance_id=instance_id,
                strategy_name=data.get("strategy", "unknown"),
                status=InstanceStatus.RUNNING,
                started_at=self._parse_timestamp(event.get("timestamp")),
                prompt=data.get("prompt"),
                model=data.get("model", ""),
                current_activity="Starting...",
                last_updated=datetime.now(),
            )
            self.state.current_run.instances[instance_id] = instance
            self.state.current_run.total_instances += 1
            self.state.current_run.active_instances += 1
            logger.info(
                f"Instance {instance_id} started. Active: {self.state.current_run.active_instances}"
            )
        else:
            data = event.get("data", {})
            instance = self.state.current_run.instances[instance_id]

            was_queued = instance.status == InstanceStatus.QUEUED

            instance.status = InstanceStatus.RUNNING
            instance.started_at = self._parse_timestamp(event.get("timestamp"))
            instance.prompt = data.get("prompt")
            instance.model = data.get("model", "")
            if instance.strategy_name == "unknown" and data.get("strategy"):
                instance.strategy_name = data.get("strategy")
            instance.current_activity = instance.current_activity or "Starting..."
            instance.last_updated = datetime.now()

            if was_queued:
                self.state.current_run.active_instances += 1
                logger.info(
                    f"Instance {instance_id} started (was queued). Active: {self.state.current_run.active_instances}"
                )

    def _handle_instance_completed(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id:
            logger.warning("Got instance.completed without instance_id")
            return

        if instance_id not in self.state.current_run.instances:
            logger.warning(
                f"Got instance.completed for unknown instance: {instance_id}"
            )
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        if instance.status != InstanceStatus.COMPLETED:
            self._set_instance_completed_state(instance, data, event, instance_id)

        metrics = data.get("metrics", {}) or {}
        self._apply_instance_completion_metrics(instance, metrics)

        self.state.current_run.completed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

        token_delta = self._calculate_token_delta(instance)
        if token_delta > 0:
            instance.applied_run_tokens += token_delta
            self.state.current_run.total_tokens += token_delta

        cost_delta = self._calculate_cost_delta(instance)
        if cost_delta > 0.0:
            instance.applied_run_cost += cost_delta
            self.state.current_run.total_cost += cost_delta

        logger.info(
            f"Instance {instance_id} completed. Run totals - Completed: {self.state.current_run.completed_instances}, Active: {self.state.current_run.active_instances}"
        )

        self._update_strategy_completion_totals(instance_id)

    def _handle_instance_failed(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.status = InstanceStatus.FAILED
        instance.completed_at = self._parse_timestamp(event.get("timestamp"))
        instance.error = data.get("error")
        instance.error_type = data.get("error_type")
        instance.last_updated = datetime.now()

        self.state.current_run.failed_instances += 1
        self.state.current_run.active_instances = max(
            0, self.state.current_run.active_instances - 1
        )

        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.failed_instances += 1
                break

    def _handle_instance_progress(self, event: Dict[str, Any]) -> None:
        if not self.state.current_run:
            return

        instance_id = event.get("instance_id")
        if not instance_id or instance_id not in self.state.current_run.instances:
            return

        data = event.get("data", {})
        instance = self.state.current_run.instances[instance_id]

        instance.current_activity = data.get("activity")
        instance.last_updated = datetime.now()

        if isinstance(data, dict):
            usage_payload = data.get("usage")
            if isinstance(usage_payload, dict):
                message_id = (
                    data.get("message_id")
                    if isinstance(data.get("message_id"), str)
                    else None
                )
                self._apply_usage_metrics(instance, usage_payload, message_id)

    def _set_instance_completed_state(
        self,
        instance: InstanceDisplay,
        data: Dict[str, Any],
        event: Dict[str, Any],
        instance_id: str,
    ) -> None:
        instance.status = InstanceStatus.COMPLETED
        instance.completed_at = self._parse_timestamp(event.get("timestamp"))
        instance.branch_name = data.get("branch_name")
        instance.duration_seconds = data.get("duration_seconds", 0.0)
        instance.current_activity = "Completed"
        instance.last_updated = datetime.now()
        logger.info(
            f"Instance {instance_id} completed after activity: {instance.current_activity}"
        )

    def _apply_instance_completion_metrics(
        self, instance: InstanceDisplay, metrics: Dict[str, Any]
    ) -> None:
        try:
            instance.cost = float(metrics.get("total_cost", instance.cost) or 0.0)
        except Exception:
            pass

        total_tokens_val = instance.total_tokens
        input_raw = instance.input_tokens
        output_raw = instance.output_tokens

        try:
            tt = metrics.get("total_tokens")
            if isinstance(tt, (int, float)):
                total_tokens_val = int(tt)
        except Exception:
            pass
        try:
            it = metrics.get("input_tokens")
            if isinstance(it, (int, float)):
                input_raw = int(it)
        except Exception:
            pass
        try:
            ot = metrics.get("output_tokens")
            if isinstance(ot, (int, float)):
                output_raw = int(ot)
        except Exception:
            pass

        fresh_input = max(0, total_tokens_val - output_raw)
        instance.total_tokens = total_tokens_val
        instance.input_tokens = fresh_input
        instance.output_tokens = output_raw
        instance.cached_input_tokens = max(
            instance.cached_input_tokens, max(0, input_raw - fresh_input)
        )

        instance.usage_running_total = max(
            instance.usage_running_total, total_tokens_val
        )
        instance.usage_input_running_total = max(
            instance.usage_input_running_total, input_raw
        )
        instance.usage_prompt_running_total = max(
            instance.usage_prompt_running_total, fresh_input
        )
        instance.usage_output_running_total = max(
            instance.usage_output_running_total, output_raw
        )

    def _calculate_token_delta(self, instance: InstanceDisplay) -> int:
        try:
            final_tokens = int(instance.total_tokens or 0)
        except Exception:
            final_tokens = 0
        return max(0, final_tokens - getattr(instance, "applied_run_tokens", 0))

    def _calculate_cost_delta(self, instance: InstanceDisplay) -> float:
        try:
            final_cost = float(instance.cost or 0.0)
        except Exception:
            final_cost = 0.0
        return max(0.0, final_cost - getattr(instance, "applied_run_cost", 0.0))

    def _update_strategy_completion_totals(self, instance_id: str) -> None:
        for strategy in self.state.current_run.strategies.values():
            if instance_id in strategy.instance_ids:
                strategy.completed_instances += 1
                break


__all__ = ["InstanceLifecycleHandlers"]
