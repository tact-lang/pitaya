"""Snapshot and reconciliation helpers for display rendering."""

from __future__ import annotations

from typing import Any

from ..models import InstanceDisplay, RunDisplay, StrategyDisplay
from ..event_processing.logging_config import logger


class SnapshotMixin:
    """Build immutable snapshots and reconcile orchestrator state."""

    def _reconcile_state(self, orchestrator_state: Any) -> None:
        """Reconcile orchestrator state with TUI state."""
        if not self.state.current_run:
            return

        run = self.state.current_run

        try:
            if isinstance(orchestrator_state, dict):
                run.total_cost = orchestrator_state.get("total_cost", run.total_cost)
                run.total_tokens = orchestrator_state.get(
                    "total_tokens", run.total_tokens
                )
                run.total_instances = orchestrator_state.get(
                    "total_instances", run.total_instances
                )
                run.completed_instances = orchestrator_state.get(
                    "completed_instances", run.completed_instances
                )
                run.failed_instances = orchestrator_state.get(
                    "failed_instances", run.failed_instances
                )
                self._reconcile_times_dict(run, orchestrator_state)
            else:
                run.total_cost = getattr(
                    orchestrator_state, "total_cost", run.total_cost
                )
                run.total_tokens = getattr(
                    orchestrator_state, "total_tokens", run.total_tokens
                )
                run.total_instances = getattr(
                    orchestrator_state, "total_instances", run.total_instances
                )
                run.completed_instances = getattr(
                    orchestrator_state, "completed_instances", run.completed_instances
                )
                run.failed_instances = getattr(
                    orchestrator_state, "failed_instances", run.failed_instances
                )
                self._reconcile_times_attr(run, orchestrator_state)
        except Exception:
            pass

        try:
            instances_map = (
                orchestrator_state.get("instances")
                if isinstance(orchestrator_state, dict)
                else getattr(orchestrator_state, "instances", {})
            )
            if isinstance(instances_map, dict):
                for iid, info in instances_map.items():
                    inst = run.instances.get(iid)
                    if not inst:
                        inst = InstanceDisplay(
                            instance_id=iid,
                            strategy_name=getattr(info, "strategy_name", ""),
                        )
                        run.instances[iid] = inst
                    self._apply_instance_basic_state(inst, info)
        except Exception:
            pass

    def _reconcile_times_dict(self, run, orchestrator_state: Any) -> None:
        try:
            from datetime import datetime as _dt

            sa = orchestrator_state.get("started_at")
            ca = orchestrator_state.get("completed_at")
            if isinstance(sa, str):
                run.started_at = _dt.fromisoformat(sa.replace("Z", "+00:00"))
            elif sa is not None:
                run.started_at = sa
            if isinstance(ca, str):
                run.completed_at = _dt.fromisoformat(ca.replace("Z", "+00:00"))
            elif ca is not None:
                run.completed_at = ca
        except Exception:
            pass

    def _reconcile_times_attr(self, run, orchestrator_state: Any) -> None:
        try:
            run.started_at = getattr(orchestrator_state, "started_at", run.started_at)
            run.completed_at = getattr(
                orchestrator_state, "completed_at", run.completed_at
            )
        except Exception:
            pass

    def _apply_instance_basic_state(self, inst: InstanceDisplay, info: Any) -> None:
        try:
            st = getattr(info, "state", None)
            st_val = (
                st.value
                if hasattr(st, "value")
                else (st if isinstance(st, str) else None)
            )
            if st_val:
                from ..models import InstanceStatus as TStatus

                if st_val in (
                    "queued",
                    "running",
                    "interrupted",
                    "completed",
                    "failed",
                ):
                    inst.status = TStatus(st_val)
        except Exception:
            pass
        try:
            inst.started_at = (
                getattr(info, "started_at", inst.started_at) or inst.started_at
            )
            inst.completed_at = (
                getattr(info, "completed_at", inst.completed_at) or inst.completed_at
            )
        except Exception:
            pass

    def _snapshot_state_for_render(self) -> None:
        """Build an immutable snapshot of the current state for rendering."""
        try:
            src = self.state.current_run
            if not src:
                self._render_run = None
                return

            run = RunDisplay(
                run_id=src.run_id,
                prompt=src.prompt,
                repo_path=src.repo_path,
                base_branch=src.base_branch,
                strategies={},
                instances={},
                total_cost=src.total_cost,
                total_tokens=src.total_tokens,
                total_instances=src.total_instances,
                active_instances=src.active_instances,
                completed_instances=src.completed_instances,
                failed_instances=src.failed_instances,
                started_at=src.started_at,
                completed_at=src.completed_at,
                ui_started_at=self._ui_started_at,
                force_detail_level=(
                    src.force_detail_level
                    or self._force_display_mode_cli
                    or self._force_display_mode_env
                ),
            )

            for sid, s in src.strategies.items():
                run.strategies[sid] = StrategyDisplay(
                    strategy_id=s.strategy_id,
                    strategy_name=s.strategy_name,
                    config=dict(s.config) if s.config else {},
                    instance_ids=list(s.instance_ids),
                    total_instances=s.total_instances,
                    completed_instances=s.completed_instances,
                    failed_instances=s.failed_instances,
                    started_at=s.started_at,
                    completed_at=s.completed_at,
                    is_complete=s.is_complete,
                )

            for iid, inst in src.instances.items():
                run.instances[iid] = InstanceDisplay(
                    instance_id=inst.instance_id,
                    strategy_name=inst.strategy_name,
                    status=inst.status,
                    branch_name=inst.branch_name,
                    prompt=inst.prompt,
                    model=inst.model,
                    current_activity=inst.current_activity,
                    last_tool_use=inst.last_tool_use,
                    duration_seconds=inst.duration_seconds,
                    cost=inst.cost,
                    total_tokens=inst.total_tokens,
                    input_tokens=inst.input_tokens,
                    output_tokens=inst.output_tokens,
                    usage_running_total=inst.usage_running_total,
                    usage_input_running_total=inst.usage_input_running_total,
                    usage_output_running_total=inst.usage_output_running_total,
                    usage_message_ids=set(inst.usage_message_ids),
                    started_at=inst.started_at,
                    completed_at=inst.completed_at,
                    last_updated=inst.last_updated,
                    error=inst.error,
                    error_type=inst.error_type,
                    metadata=dict(inst.metadata) if inst.metadata else {},
                )

            self._render_run = run
        except Exception as e:
            logger.debug(f"Snapshot build failed: {e}")
            self._render_run = self.state.current_run


__all__ = ["SnapshotMixin"]
