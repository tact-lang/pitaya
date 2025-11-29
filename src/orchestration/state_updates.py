"""Update helpers for StateManager."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..shared import InstanceResult, InstanceStatus
from .state_models import InstanceInfo, StrategyExecution

logger = logging.getLogger(__name__)


def register_instance(
    state_mgr,
    *,
    instance_id: str,
    strategy_name: str,
    prompt: str,
    base_branch: str,
    branch_name: str,
    container_name: str,
    metadata: Optional[Dict[str, Any]],
) -> None:
    state = state_mgr.current_state
    if not state:
        raise RuntimeError("No active run state")
    info = InstanceInfo(
        instance_id=instance_id,
        strategy_name=strategy_name,
        prompt=prompt,
        base_branch=base_branch,
        branch_name=branch_name,
        container_name=container_name,
        state=InstanceStatus.QUEUED,
        metadata=metadata or {},
    )
    state.instances[instance_id] = info
    state.total_instances += 1
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.instance_registered",
            {
                "instance_id": instance_id,
                "strategy_name": strategy_name,
                "branch_name": branch_name,
                "container_name": container_name,
                "state": InstanceStatus.QUEUED.value,
            },
            instance_id=instance_id,
        )


def register_task(
    state_mgr, key: str, fingerprint: str, canonical_input: Optional[str]
) -> None:
    if not state_mgr.current_state:
        raise RuntimeError("No active run state")
    existing = state_mgr.current_state.tasks.get(key)
    if existing and existing.get("fingerprint") != fingerprint:
        from ..exceptions import KeyConflictDifferentFingerprint

        raise KeyConflictDifferentFingerprint(
            f"KeyConflictDifferentFingerprint for key {key}"
        )
    entry = {"fingerprint": fingerprint}
    if canonical_input is not None:
        entry["input"] = canonical_input
    state_mgr.current_state.tasks[key] = entry
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.task_registered",
            {"key": key, "fingerprint": fingerprint},
        )


def update_instance_state(
    state_mgr, instance_id: str, state: InstanceStatus, result: Optional[InstanceResult]
) -> None:
    if not state_mgr.current_state:
        raise RuntimeError("No active run state")
    info = state_mgr.current_state.instances.get(instance_id)
    if not info:
        logger.warning("Unknown instance: %s", instance_id)
        return
    old_state = info.state
    info.state = state

    if state == InstanceStatus.RUNNING and not info.started_at:
        info.started_at = datetime.now(timezone.utc)
    elif state == InstanceStatus.INTERRUPTED:
        if old_state != InstanceStatus.INTERRUPTED:
            info.interrupted_at = datetime.now(timezone.utc)
            _emit_task_interrupted(state_mgr, instance_id, info)
        if result is not None:
            info.result = result
    elif state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
        if not info.completed_at:
            info.completed_at = datetime.now(timezone.utc)
        info.result = result
        if result and result.session_id:
            info.session_id = result.session_id
        _bump_counters(state_mgr, state, old_state)
        _update_metrics(state_mgr, info, result)

    _emit_instance_updated(state_mgr, instance_id, old_state, state, info, result)


def update_instance_session_id(
    state_mgr, instance_id: str, session_id: Optional[str]
) -> None:
    state = state_mgr.current_state
    if not state:
        return
    info = state.instances.get(instance_id)
    if not info:
        return
    info.session_id = session_id
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.instance_updated",
            {
                "instance_id": instance_id,
                "session_id": session_id,
                "new_state": info.state.value,
                "old_state": info.state.value,
            },
            instance_id=instance_id,
        )


def update_instance_metadata(
    state_mgr, instance_id: str, patch: Dict[str, Any]
) -> None:
    state = state_mgr.current_state
    if not state:
        return
    info = state.instances.get(instance_id)
    if not info:
        return
    try:
        info.metadata = {**(info.metadata or {}), **(patch or {})}
    except Exception:
        info.metadata = patch or {}
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.instance_updated",
            {
                "instance_id": instance_id,
                "metadata": info.metadata,
                "new_state": info.state.value,
                "old_state": info.state.value,
            },
            instance_id=instance_id,
        )


def update_instance_container_name(
    state_mgr, instance_id: str, container_name: str
) -> None:
    state = state_mgr.current_state
    if not state:
        return
    info = state.instances.get(instance_id)
    if not info:
        return
    info.container_name = container_name
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.instance_updated",
            {
                "instance_id": instance_id,
                "container_name": container_name,
                "new_state": info.state.value,
                "old_state": info.state.value,
            },
            instance_id=instance_id,
        )


def register_strategy(
    state_mgr, strategy_id: str, strategy_name: str, config: Dict[str, Any]
) -> None:
    if not state_mgr.current_state:
        raise RuntimeError("No active run state")
    state_mgr.current_state.strategies[strategy_id] = StrategyExecution(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        config=config,
    )
    if state_mgr.event_bus:
        state_mgr.event_bus.emit(
            "state.strategy_registered",
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "config": config,
            },
        )


def update_strategy_state(
    state_mgr, strategy_id: str, state: str, results: Optional[list[InstanceResult]]
) -> None:
    if not state_mgr.current_state:
        raise RuntimeError("No active run state")
    strategy = state_mgr.current_state.strategies.get(strategy_id)
    if not strategy:
        logger.warning("Unknown strategy: %s", strategy_id)
        return
    old_state = strategy.state
    strategy.state = state
    if state in ("completed", "failed"):
        strategy.completed_at = datetime.now(timezone.utc)
        if results:
            strategy.results = results
    if state_mgr.event_bus:
        event_data = {
            "strategy_id": strategy_id,
            "old_state": old_state,
            "new_state": state,
        }
        if state in ("completed", "failed"):
            event_data["completed_at"] = strategy.completed_at.isoformat()
            if results:
                event_data["result_count"] = len(results)
                event_data["branch_names"] = [
                    r.branch_name for r in results if r.branch_name
                ]
        state_mgr.event_bus.emit("state.strategy_updated", event_data)


def update_strategy_rand(state_mgr, strategy_id: str, seq: int, value: float) -> None:
    if not state_mgr.current_state:
        return
    state_mgr.current_state.strategy_rand[strategy_id] = {
        "seq": int(seq),
        "value": float(value),
    }
    if state_mgr.event_bus:
        try:
            state_mgr.event_bus.emit(
                "state.strategy_rand_updated",
                {
                    "strategy_id": strategy_id,
                    "seq": int(seq),
                    "value": float(value),
                },
            )
        except Exception:
            pass


def _emit_task_interrupted(state_mgr, instance_id: str, info: InstanceInfo) -> None:
    try:
        if state_mgr.event_bus and getattr(state_mgr.event_bus, "persist_path", None):
            key = (info.metadata or {}).get("key")
            if not key:
                return
            sid = None
            for s_id, strat in state_mgr.current_state.strategies.items():
                if instance_id in strat.instance_ids:
                    sid = s_id
                    break
            run_id = state_mgr.current_state.run_id
            state_mgr.event_bus.emit_canonical(
                type="task.interrupted",
                run_id=run_id,
                strategy_execution_id=sid,
                key=key,
                payload={"key": key, "instance_id": instance_id},
            )
            try:
                state_mgr.event_bus.flush_pending()
            except Exception:
                pass
    except Exception:
        pass


def _bump_counters(
    state_mgr, new_state: InstanceStatus, old_state: InstanceStatus
) -> None:
    if new_state == InstanceStatus.COMPLETED and old_state != InstanceStatus.COMPLETED:
        state_mgr.current_state.completed_instances += 1
    elif new_state == InstanceStatus.FAILED and old_state != InstanceStatus.FAILED:
        state_mgr.current_state.failed_instances += 1


def _update_metrics(
    state_mgr, info: InstanceInfo, result: Optional[InstanceResult]
) -> None:
    if not result or not result.metrics:
        return
    cost = float(result.metrics.get("total_cost", 0.0) or 0.0)
    tokens = int(result.metrics.get("total_tokens", 0) or 0)
    cost_delta = max(0.0, cost - getattr(info, "aggregated_cost", 0.0))
    token_delta = max(0, tokens - getattr(info, "aggregated_tokens", 0))
    if cost_delta:
        state_mgr.current_state.total_cost += cost_delta
    if token_delta:
        state_mgr.current_state.total_tokens += token_delta
    info.aggregated_cost = max(info.aggregated_cost, cost)
    info.aggregated_tokens = max(info.aggregated_tokens, tokens)


def _emit_instance_updated(
    state_mgr,
    instance_id: str,
    old_state: InstanceStatus,
    new_state: InstanceStatus,
    info: InstanceInfo,
    result: Optional[InstanceResult],
) -> None:
    if not state_mgr.event_bus:
        return
    event_data: Dict[str, Any] = {
        "instance_id": instance_id,
        "old_state": old_state.value,
        "new_state": new_state.value,
    }
    if new_state == InstanceStatus.RUNNING and info.started_at:
        event_data["started_at"] = info.started_at.isoformat()
    elif new_state == InstanceStatus.INTERRUPTED and info.interrupted_at:
        event_data["interrupted_at"] = info.interrupted_at.isoformat()
    elif (
        new_state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED)
        and info.completed_at
    ):
        event_data["completed_at"] = info.completed_at.isoformat()
        if result:
            event_data["branch_name"] = result.branch_name
            event_data["cost"] = result.metrics.get("total_cost", 0.0)
            event_data["tokens"] = result.metrics.get("total_tokens", 0)
    state_mgr.event_bus.emit(
        "state.instance_updated", event_data, instance_id=instance_id
    )
    try:
        logger.debug(
            "state.update_instance_state: iid=%s %s->%s",
            instance_id,
            old_state.value,
            new_state.value,
        )
    except Exception:
        pass
