"""Instance spawn helpers.

Separated from the main orchestrator to keep the spawn path readable and testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..shared import InstanceResult, InstanceStatus
from .state import InstanceInfo

logger = logging.getLogger(__name__)


def _short16_from(obj: dict[str, Any]) -> str:
    """Generate a stable 16-hex identifier from a canonical JSON dict."""
    encoded = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _strategy_index(state_manager, strategy_execution_id: str) -> int:
    strat = state_manager.current_state.strategies.get(strategy_execution_id)
    if not strat:
        return 1
    started_at = strat.started_at
    earlier = [
        s
        for s in state_manager.current_state.strategies.values()
        if s.started_at < started_at
    ]
    return len(earlier) + 1


def _build_names(
    strategy_name: str,
    run_id: str,
    strategy_index: int,
    durable_key: str,
) -> tuple[str, str]:
    """Return (container_name, branch_name) for an instance."""
    khash = hashlib.sha256(
        f"{strategy_index}|{durable_key}".encode("utf-8")
    ).hexdigest()[:8]
    try:
        strategy_segment = re.compile(r"[^A-Za-z0-9._-]+").sub(
            "-", str(strategy_name or "").strip()
        )
        strategy_segment = strategy_segment.strip("-/._") or "unknown"
    except Exception:
        strategy_segment = str(strategy_name or "unknown")

    container_name = f"pitaya_{run_id}_s{strategy_index}_k{khash}"
    branch_name = f"pitaya/{strategy_segment}/{run_id}/k{khash}"
    if len(branch_name) > 200:
        head = f"pitaya/{strategy_segment}/"
        tail = branch_name.split("/")[-1]
        room = 200 - (len(head) + 1 + len(tail))
        if room > 8:
            branch_name = f"{head}{run_id[:room]}/{tail}"
        else:
            branch_name = f"pitaya/unknown/{tail}"[-200:]
    return container_name, branch_name


def _register_instance_if_needed(
    state_manager,
    instance_id: str,
    *,
    strategy_name: str,
    prompt: str,
    base_branch: str,
    branch_name: str,
    container_name: str,
    metadata: Optional[Dict[str, Any]],
) -> InstanceInfo:
    if instance_id in state_manager.current_state.instances:
        return state_manager.current_state.instances[instance_id]

    state_manager.register_instance(
        instance_id=instance_id,
        strategy_name=strategy_name,
        prompt=prompt,
        base_branch=base_branch,
        branch_name=branch_name,
        container_name=container_name,
        metadata=metadata,
    )
    return state_manager.current_state.instances[instance_id]


def _attach_to_strategy(state_manager, strategy_execution_id: str, instance_id: str) -> None:
    if (
        strategy_execution_id
        and strategy_execution_id in state_manager.current_state.strategies
    ):
        ids = state_manager.current_state.strategies[strategy_execution_id].instance_ids
        if instance_id not in ids:
            ids.append(instance_id)


def _emit_terminal_if_missing(orchestrator, instance_id: str, info: InstanceInfo) -> None:
    """Emit canonical terminal events when resuming and the log is missing them."""
    task_key = (info.metadata or {}).get("key")
    if not task_key or not orchestrator.event_bus:
        return

    seen_terminal = False
    try:
        term_types = {"task.completed", "task.failed", "task.interrupted"}
        events, _ = orchestrator.event_bus.get_events_since(
            offset=0, event_types=term_types
        )
        for ev in events or []:
            payload = ev.get("payload") or {}
            iid = payload.get("instance_id") or ev.get("instance_id")
            if str(iid) == str(instance_id):
                seen_terminal = True
                break
    except Exception:
        seen_terminal = False
    if seen_terminal:
        return

    strategy_execution_id = None
    for sid, strat in orchestrator.state_manager.current_state.strategies.items():
        if instance_id in strat.instance_ids:
            strategy_execution_id = sid
            break

    if info.state == InstanceStatus.COMPLETED:
        result = info.result
        if not result:
            return
        artifact = {
            "type": "branch",
            "branch_planned": info.branch_name,
            "branch_final": result.branch_name,
            "base": info.base_branch,
            "commit": getattr(result, "commit", None) or "",
            "has_changes": bool(getattr(result, "has_changes", False)),
            "duplicate_of_branch": getattr(result, "duplicate_of_branch", None),
            "dedupe_reason": getattr(result, "dedupe_reason", None),
        }
        full_msg = getattr(result, "final_message", None) or ""
        msg_bytes = full_msg.encode("utf-8", errors="ignore")
        max_bytes = 65536
        truncated_flag = False
        rel_path = ""
        out_msg = full_msg
        if max_bytes > 0 and len(msg_bytes) > max_bytes:
            truncated_flag = True
            out_msg = msg_bytes[:max_bytes].decode("utf-8", errors="ignore")
            try:
                run_logs = orchestrator.logs_dir / orchestrator.state_manager.current_state.run_id
                dest_dir = run_logs / "final_messages"
                dest_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{instance_id}.txt"
                with open(dest_dir / fname, "w", encoding="utf-8", errors="ignore") as fh:
                    fh.write(full_msg)
                rel_path = f"final_messages/{fname}"
            except Exception:
                truncated_flag = False
                rel_path = ""
        orchestrator.event_bus.emit_canonical(
            type="task.completed",
            run_id=orchestrator.state_manager.current_state.run_id,
            strategy_execution_id=strategy_execution_id,
            key=task_key,
            payload={
                "key": task_key,
                "instance_id": instance_id,
                "artifact": artifact,
                "metrics": result.metrics or {},
                "final_message": out_msg,
                "final_message_truncated": truncated_flag,
                "final_message_path": rel_path,
            },
        )
    elif info.state == InstanceStatus.FAILED:
        orchestrator.event_bus.emit_canonical(
            type="task.failed",
            run_id=orchestrator.state_manager.current_state.run_id,
            strategy_execution_id=strategy_execution_id,
            key=task_key,
            payload={
                "key": task_key,
                "instance_id": instance_id,
                "error_type": info.result.error_type if info.result else "unknown",
                "message": info.result.error if info.result else "",
                "network_egress": (info.metadata or {}).get("network_egress"),
            },
        )
    elif info.state == InstanceStatus.INTERRUPTED:
        orchestrator.event_bus.emit_canonical(
            type="task.interrupted",
            run_id=orchestrator.state_manager.current_state.run_id,
            strategy_execution_id=strategy_execution_id,
            key=task_key,
            payload={"key": task_key, "instance_id": instance_id},
        )


async def spawn_instance(
    *,
    orchestrator,
    manager,
    prompt: str,
    repo_path: Path,
    base_branch: str,
    strategy_name: str,
    strategy_execution_id: str,
    instance_index: int,
    metadata: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None,
) -> str:
    """Spawn a new instance and enqueue it for execution."""
    logger.info(
        "spawn_instance: strategy=%s sid=%s key=%s",
        strategy_name,
        strategy_execution_id,
        (metadata or {}).get("key", "-"),
    )

    await orchestrator._check_disk_space()

    run_id = orchestrator.state_manager.current_state.run_id
    durable_key = key or f"i:{instance_index}"
    instance_id = _short16_from(
        {
            "run_id": run_id,
            "strategy_execution_id": strategy_execution_id,
            "key": durable_key,
        }
    )

    strategy_idx = _strategy_index(orchestrator.state_manager, strategy_execution_id)
    container_name, branch_name = _build_names(
        strategy_name, run_id, strategy_idx, durable_key
    )
    info = _register_instance_if_needed(
        orchestrator.state_manager,
        instance_id,
        strategy_name=strategy_name,
        prompt=prompt,
        base_branch=base_branch,
        branch_name=branch_name,
        container_name=container_name,
        metadata=metadata,
    )
    _attach_to_strategy(orchestrator.state_manager, strategy_execution_id, instance_id)

    if instance_id in manager.futures and manager.futures[instance_id].done():
        # Already terminal; no further action needed.
        return instance_id

    if instance_id in manager.futures and info.state == InstanceStatus.INTERRUPTED:
        logger.debug(
            "spawn_instance: iid=%s interrupted previously; keeping pending future",
            instance_id,
        )
        return instance_id

    if info.state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
        future = asyncio.Future()
        if not info.result:
            info.result = InstanceResult(
                success=(info.state == InstanceStatus.COMPLETED),
                branch_name=info.branch_name,
                has_changes=False,
                metrics={},
                session_id=info.session_id,
                status="success" if info.state == InstanceStatus.COMPLETED else "failed",
            )
        future.set_result(info.result)
        manager.futures[instance_id] = future
        _emit_terminal_if_missing(orchestrator, instance_id, info)
        return instance_id

    # Create future and enqueue for fresh execution
    manager.futures[instance_id] = asyncio.Future()
    await manager.enqueue(instance_id)
    orchestrator.event_bus.emit(
        "instance.queued",
        {
            "instance_id": instance_id,
            "strategy": strategy_name,
            "branch_name": branch_name,
        },
        instance_id=instance_id,
    )

    return instance_id
