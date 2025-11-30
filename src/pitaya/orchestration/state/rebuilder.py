"""Event replay helpers for StateManager."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from pitaya.shared import InstanceResult, InstanceStatus
from pitaya.orchestration.state.models import InstanceInfo, RunState, StrategyExecution

logger = logging.getLogger(__name__)


def rebuild_from_events(state: RunState, events: Iterable[Dict[str, Any]]) -> None:
    for ev in events:
        apply_event(state, ev)
    _crash_infer(state, events)


def apply_event(state: RunState, event: Dict[str, Any]) -> None:
    event_type = event.get("type")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else None
    data = payload if payload is not None else event.get("data", {})
    ts = _parse_ts(event.get("ts") or event.get("timestamp"))

    try:
        so = event.get("start_offset")
        if isinstance(so, int):
            state.last_event_start_offset = so
    except Exception:
        pass

    if event_type == "strategy.started":
        _on_strategy_started(state, data, event, ts)
    elif event_type == "strategy.completed":
        _on_strategy_completed(state, data, event, ts)
    elif event_type == "task.scheduled":
        _on_task_scheduled(state, data, event, ts)
    elif event_type == "task.started":
        _on_task_started(state, data, event, ts)
    elif event_type in ("task.completed", "task.failed", "task.interrupted"):
        _on_task_terminal(state, data, event, ts, event_type)
    elif event_type == "strategy.rand":
        sid = event.get("strategy_execution_id")
        if sid:
            state.strategy_rand[sid] = {
                "seq": data.get("seq"),
                "value": data.get("value"),
            }


def _on_strategy_started(
    state: RunState, data: Dict[str, Any], event: Dict[str, Any], ts: Optional[datetime]
) -> None:
    sid = event.get("strategy_execution_id") or data.get("strategy_id")
    if not sid or sid in state.strategies:
        return
    state.strategies[sid] = StrategyExecution(
        strategy_id=sid,
        strategy_name=data.get("name", data.get("strategy_name", "")),
        config=data.get("params", data.get("config", {})),
        started_at=ts or datetime.now(timezone.utc),
    )


def _on_strategy_completed(
    state: RunState, data: Dict[str, Any], event: Dict[str, Any], ts: Optional[datetime]
) -> None:
    sid = event.get("strategy_execution_id") or data.get("strategy_id")
    if not sid or sid not in state.strategies:
        return
    strat = state.strategies[sid]
    strat.completed_at = ts or datetime.now(timezone.utc)
    status = data.get("status")
    if status == "success":
        strat.state = "completed"
    elif status == "failed":
        strat.state = "failed"
    elif status == "canceled":
        strat.state = "canceled"


def _on_task_scheduled(
    state: RunState, data: Dict[str, Any], event: Dict[str, Any], ts: Optional[datetime]
) -> None:
    iid = data.get("instance_id")
    if not iid or iid in state.instances:
        return
    sname = ""
    sid = event.get("strategy_execution_id")
    if sid and sid in state.strategies:
        sname = state.strategies[sid].strategy_name
    info = InstanceInfo(
        instance_id=iid,
        strategy_name=sname,
        prompt=data.get("prompt", ""),
        base_branch=data.get("base_branch", "main"),
        branch_name=data.get("branch_name", ""),
        container_name=data.get("container_name", ""),
        state=InstanceStatus.QUEUED,
        metadata=data.get("metadata", {}),
        created_at=ts or datetime.now(timezone.utc),
    )
    state.instances[iid] = info
    state.total_instances = len(state.instances)
    if sid and sid in state.strategies:
        strat = state.strategies[sid]
        if iid not in strat.instance_ids:
            strat.instance_ids.append(iid)


def _on_task_started(
    state: RunState, data: Dict[str, Any], event: Dict[str, Any], ts: Optional[datetime]
) -> None:
    iid = data.get("instance_id")
    info = state.instances.get(iid) if iid else None
    if not info:
        return
    info.state = InstanceStatus.RUNNING
    info.started_at = ts or datetime.now(timezone.utc)


def _on_task_terminal(
    state: RunState,
    data: Dict[str, Any],
    event: Dict[str, Any],
    ts: Optional[datetime],
    event_type: str,
) -> None:
    iid = data.get("instance_id")
    info = state.instances.get(iid) if iid else None
    if not info:
        return

    if event_type == "task.interrupted":
        info.state = InstanceStatus.INTERRUPTED
        info.interrupted_at = ts or datetime.now(timezone.utc)
        return

    success = event_type == "task.completed"
    result_payload = data.get("artifact") or {}
    metrics = data.get("metrics") or {}
    final_message = data.get("final_message")
    result = InstanceResult(
        success=success,
        branch_name=result_payload.get("branch_final")
        or result_payload.get("branch_planned"),
        has_changes=result_payload.get("has_changes", False),
        metrics=metrics,
        status="success" if success else "failed",
        final_message=final_message,
        commit=result_payload.get("commit"),
        duplicate_of_branch=result_payload.get("duplicate_of_branch"),
        dedupe_reason=result_payload.get("dedupe_reason"),
    )
    info.result = result
    info.state = InstanceStatus.COMPLETED if success else InstanceStatus.FAILED
    info.completed_at = ts or datetime.now(timezone.utc)
    state.completed_instances = sum(
        1 for i in state.instances.values() if i.state == InstanceStatus.COMPLETED
    )
    state.failed_instances = sum(
        1 for i in state.instances.values() if i.state == InstanceStatus.FAILED
    )


def _crash_infer(state: RunState, events: Iterable[Dict[str, Any]]) -> None:
    try:
        terminal_iids: set[str] = set()
        for ev in events or []:
            et = str(ev.get("type", ""))
            iid = (
                ev.get("payload", {}).get("instance_id")
                if isinstance(ev.get("payload"), dict)
                else None
            )
            if not iid:
                iid = ev.get("instance_id")
            if iid and et in {"task.completed", "task.failed", "task.interrupted"}:
                terminal_iids.add(iid)

        for iid, info in list(state.instances.items()):
            if info.state == InstanceStatus.RUNNING and iid not in terminal_iids:
                info.state = InstanceStatus.INTERRUPTED
                info.interrupted_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.debug("Crash inference check failed: %s", exc)


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None
