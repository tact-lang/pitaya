"""Helper utilities for run resume logic."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from ..shared import InstanceResult, InstanceStatus
from .strategies import AVAILABLE_STRATEGIES as STRATEGIES
from .strategy_context import StrategyContext

logger = logging.getLogger(__name__)


def prepare_event_bus(orchestrator, run_id: str) -> None:
    """Point event bus to the resumed run and apply redaction."""
    event_log_path = orchestrator.logs_dir / run_id / "events.jsonl"
    if not orchestrator.event_bus:
        from .event_bus import EventBus

        orchestrator.event_bus = EventBus(
            max_events=orchestrator.event_buffer_size, persist_path=event_log_path
        )
    else:
        orchestrator.event_bus.reconfigure_persistence(event_log_path, run_id)
    try:
        if orchestrator._pending_redaction_patterns:
            orchestrator.event_bus.set_custom_redaction_patterns(
                list(orchestrator._pending_redaction_patterns)
            )
    except Exception:
        pass
    orchestrator.state_manager.event_bus = orchestrator.event_bus


def backfill_terminal(orchestrator, run_id: str) -> None:
    try:
        term_types = {"task.completed", "task.failed", "task.interrupted"}
        seen: set[str] = set()
        try:
            events, _ = orchestrator.event_bus.get_events_since(
                offset=0, event_types=term_types
            )
        except Exception:
            events = []
        for ev in events or []:
            payload = ev.get("payload") or {}
            iid = payload.get("instance_id") or ev.get("instance_id")
            if iid:
                seen.add(str(iid))

        for iid, info in list(
            orchestrator.state_manager.current_state.instances.items()
        ):
            task_key = (info.metadata or {}).get("key")
            if not task_key:
                continue
            if info.state.value not in ("completed", "failed", "interrupted"):
                continue
            if iid in seen:
                continue
            strategy_execution_id = None
            for (
                sid,
                strat,
            ) in orchestrator.state_manager.current_state.strategies.items():
                if iid in strat.instance_ids:
                    strategy_execution_id = sid
                    break
            if info.state.value == "completed" and info.result:
                res = info.result
                artifact = {
                    "type": "branch",
                    "branch_planned": info.branch_name,
                    "branch_final": res.branch_name,
                    "base": info.base_branch,
                    "commit": getattr(res, "commit", None) or "",
                    "has_changes": bool(res.has_changes),
                    "duplicate_of_branch": getattr(res, "duplicate_of_branch", None),
                    "dedupe_reason": getattr(res, "dedupe_reason", None),
                }
                full_msg = getattr(res, "final_message", None) or ""
                msg_bytes = full_msg.encode("utf-8", errors="ignore")
                max_bytes = 65536
                truncated_flag = False
                rel_path = ""
                out_msg = full_msg
                if max_bytes > 0 and len(msg_bytes) > max_bytes:
                    truncated_flag = True
                    out_msg = msg_bytes[:max_bytes].decode("utf-8", errors="ignore")
                    try:
                        run_logs = orchestrator.logs_dir / run_id
                        dest_dir = run_logs / "final_messages"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{iid}.txt"
                        with open(
                            dest_dir / fname, "w", encoding="utf-8", errors="ignore"
                        ) as fh:
                            fh.write(full_msg)
                        rel_path = f"final_messages/{fname}"
                    except Exception:
                        truncated_flag = False
                        rel_path = ""
                orchestrator.event_bus.emit_canonical(
                    type="task.completed",
                    run_id=run_id,
                    strategy_execution_id=strategy_execution_id,
                    key=task_key,
                    payload={
                        "key": task_key,
                        "instance_id": iid,
                        "artifact": artifact,
                        "metrics": res.metrics or {},
                        "final_message": out_msg,
                        "final_message_truncated": truncated_flag,
                        "final_message_path": rel_path,
                    },
                )
            elif info.state.value == "failed":
                res = info.result
                orchestrator.event_bus.emit_canonical(
                    type="task.failed",
                    run_id=run_id,
                    strategy_execution_id=strategy_execution_id,
                    key=task_key,
                    payload={
                        "key": task_key,
                        "instance_id": iid,
                        "error_type": (res.error_type if res else "unknown"),
                        "message": (res.error if res else ""),
                        "network_egress": (info.metadata or {}).get("network_egress"),
                    },
                )
            else:
                orchestrator.event_bus.emit_canonical(
                    type="task.interrupted",
                    run_id=run_id,
                    strategy_execution_id=strategy_execution_id,
                    key=task_key,
                    payload={"key": task_key, "instance_id": iid},
                )
    except Exception:
        pass


def reopen_strategies(orchestrator, run_id: str) -> None:
    try:
        for sid, strat in list(
            orchestrator.state_manager.current_state.strategies.items()
        ):
            if getattr(strat, "completed_at", None) is not None:
                old = strat.state
                strat.completed_at = None
                strat.state = "running"
                orchestrator.event_bus.emit(
                    "state.strategy_updated",
                    {"strategy_id": sid, "old_state": old, "new_state": "running"},
                )
                orchestrator.event_bus.emit_canonical(
                    type="strategy.started",
                    run_id=run_id,
                    strategy_execution_id=sid,
                    payload={
                        "name": strat.strategy_name,
                        "params": strat.config or {},
                        "resumed": True,
                    },
                )
    except Exception:
        pass


async def schedule_instances(orchestrator, saved_state) -> List[str]:
    from ..instance_runner.plugins import AVAILABLE_PLUGINS

    scheduled: List[str] = []
    manager = orchestrator.instance_manager

    async def _enqueue(iid: str) -> None:
        if iid not in manager.futures:
            manager.futures[iid] = asyncio.Future()
        await manager.enqueue(iid)

    for iid, info in saved_state.instances.items():
        try:
            if info.state in (InstanceStatus.RUNNING, InstanceStatus.INTERRUPTED):
                plugin_name = (info.metadata or {}).get("plugin_name", "claude-code")
                plugin_cls = (
                    AVAILABLE_PLUGINS.get(plugin_name)
                    or AVAILABLE_PLUGINS["claude-code"]
                )
                plugin = plugin_cls()
                can_resume = bool(
                    info.session_id and plugin.capabilities.supports_resume
                )
                if can_resume:
                    orchestrator.state_manager.update_instance_metadata(
                        iid, {"reuse_container": True, "operator_resume": True}
                    )
                    orchestrator.state_manager.update_instance_state(
                        iid, InstanceStatus.QUEUED
                    )
                    await _enqueue(iid)
                    scheduled.append(iid)
                    logger.info(
                        "resume_run: scheduled resume iid=%s container=%s session_id=%s plugin=%s",
                        iid,
                        info.container_name,
                        info.session_id,
                        plugin_name,
                    )
                else:
                    orchestrator.state_manager.update_instance_session_id(iid, None)
                    meta_patch = {
                        "reuse_container": False,
                        "operator_resume": True,
                        "resume_session_id": None,
                    }
                    try:
                        import uuid as _uuid

                        new_name = f"{info.container_name}_r{_uuid.uuid4().hex[:4]}"
                        orchestrator.state_manager.update_instance_container_name(
                            iid, new_name
                        )
                        meta_patch["container_name_override"] = new_name
                        orchestrator.state_manager.update_instance_metadata(
                            iid, meta_patch
                        )
                    except Exception:
                        pass
                    orchestrator.state_manager.update_instance_state(
                        iid, InstanceStatus.QUEUED
                    )
                    await _enqueue(iid)
                    scheduled.append(iid)
                    logger.info(
                        "resume_run: scheduled fresh iid=%s (no resumable session)", iid
                    )
            elif info.state == InstanceStatus.QUEUED:
                await _enqueue(iid)
                scheduled.append(iid)
                logger.info("resume_run: scheduled queued iid=%s", iid)
        except Exception as exc:
            logger.debug("Resume scheduling error for %s: %s", iid, exc)
    return scheduled


async def reenter_strategies(orchestrator, run_id: str) -> List[InstanceResult]:
    prompt = orchestrator.state_manager.current_state.prompt
    base_branch = orchestrator.state_manager.current_state.base_branch
    results: List[InstanceResult] = []

    async def _reenter_one(sid: str, strat_exec) -> List[InstanceResult]:
        if getattr(strat_exec, "completed_at", None):
            return []
        sname = strat_exec.strategy_name
        if sname not in STRATEGIES:
            return []
        strat_cls = STRATEGIES[sname]
        strat = strat_cls()
        try:
            strat.set_config_overrides(strat_exec.config or {})
        except Exception:
            pass
        ctx = StrategyContext(orchestrator, sname, sid)
        try:
            res = await strat.execute(prompt=prompt, base_branch=base_branch, ctx=ctx)
        except Exception as exc:
            logger.error("Strategy re-entry failed for %s: %s", sname, exc)
            res = []
        state_value = (
            "completed"
            if any(getattr(r, "success", False) for r in (res or []))
            else "failed"
        )
        orchestrator.state_manager.update_strategy_state(
            strategy_id=sid, state=state_value, results=res
        )
        status = "success" if state_value == "completed" else "failed"
        payload = {"status": status}
        if status == "failed":
            try:
                any_interrupted = any(
                    (
                        orchestrator.state_manager.current_state.instances.get(
                            iid
                        ).state.value
                        == "interrupted"
                    )
                    for iid in (
                        orchestrator.state_manager.current_state.strategies.get(
                            sid
                        ).instance_ids
                        or []
                    )
                    if iid in orchestrator.state_manager.current_state.instances
                )
            except Exception:
                any_interrupted = False
            if any_interrupted:
                payload["status"] = "canceled"
                payload["reason"] = "operator_interrupt"
            else:
                payload["reason"] = "no_successful_tasks"
        orchestrator.event_bus.emit_canonical(
            type="strategy.completed",
            run_id=run_id,
            strategy_execution_id=sid,
            payload=payload,
        )
        return res or []

    tasks: List[asyncio.Task] = []
    for sid, strat_exec in list(
        orchestrator.state_manager.current_state.strategies.items()
    ):
        tasks.append(asyncio.create_task(_reenter_one(sid, strat_exec)))
    if tasks:
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results_lists:
            if isinstance(res, list):
                results.extend(res)
    return results


def mark_run_completed(orchestrator) -> None:
    try:
        if orchestrator.state_manager and orchestrator.state_manager.current_state:
            orchestrator.state_manager.current_state.completed_at = datetime.now(
                timezone.utc
            )
    except Exception:
        pass
