"""Snapshot utilities for StateManager."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


async def save_snapshot(state_manager) -> None:
    state = state_manager.current_state
    if not state:
        return
    try:
        if state_manager.event_bus:
            state_manager.event_bus.flush_pending()
    except Exception:
        pass

    try:
        from pitaya.shared import InstanceStatus as _IS

        insts = list(state.instances.values())
        state.total_instances = len(insts)
        state.completed_instances = sum(1 for i in insts if i.state == _IS.COMPLETED)
        state.failed_instances = sum(1 for i in insts if i.state == _IS.FAILED)
        total_cost = 0.0
        total_tokens = 0
        for inst in insts:
            try:
                if inst.result and inst.result.metrics:
                    total_cost += float(
                        inst.result.metrics.get("total_cost", 0.0) or 0.0
                    )
                    total_tokens += int(inst.result.metrics.get("total_tokens", 0) or 0)
                    try:
                        inst.aggregated_cost = max(
                            inst.aggregated_cost,
                            float(inst.result.metrics.get("total_cost", 0.0) or 0.0),
                        )
                        inst.aggregated_tokens = max(
                            inst.aggregated_tokens,
                            int(inst.result.metrics.get("total_tokens", 0) or 0),
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        state.total_cost = total_cost
        state.total_tokens = total_tokens
    except Exception:
        pass

    run_dir = state_manager.state_dir / state.run_id
    await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)

    snapshot_path = run_dir / "state.json"
    temp_path = snapshot_path.with_suffix(".tmp")

    state_data = state.to_dict()
    await asyncio.to_thread(_write_json, temp_path, state_data)
    if temp_path.exists():
        await asyncio.to_thread(temp_path.rename, snapshot_path)
        state_manager._last_snapshot = datetime.now(timezone.utc)
    else:
        logger.warning("Temp file %s does not exist, skipping rename", temp_path)

    try:
        if state_manager.event_bus and getattr(
            state_manager.event_bus, "persist_path", None
        ):
            logs_run_dir = state_manager.event_bus.persist_path.parent
            logs_snapshot = logs_run_dir / "state.json"
            logs_tmp = logs_snapshot.with_suffix(".tmp")
            await asyncio.to_thread(_write_json, logs_tmp, state_data)
            if logs_tmp.exists():
                await asyncio.to_thread(logs_tmp.rename, logs_snapshot)
    except Exception as exc:
        logger.debug("Failed to duplicate state snapshot in logs directory: %s", exc)


async def maybe_snapshot(state_manager) -> None:
    if not state_manager.current_state:
        return
    elapsed = (
        datetime.now(timezone.utc) - state_manager._last_snapshot
    ).total_seconds()
    if elapsed >= state_manager.snapshot_interval:
        await save_snapshot(state_manager)


async def start_periodic_snapshots(state_manager) -> None:
    if state_manager._snapshot_task:
        return

    async def snapshot_loop():
        while True:
            try:
                await asyncio.sleep(state_manager.snapshot_interval)
                if state_manager.current_state:
                    await save_snapshot(state_manager)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in periodic snapshot: %s", exc)

    state_manager._snapshot_task = asyncio.create_task(snapshot_loop())
    logger.info(
        "Started periodic snapshots every %s seconds", state_manager.snapshot_interval
    )


async def stop_periodic_snapshots(state_manager) -> None:
    if state_manager._snapshot_task:
        state_manager._snapshot_task.cancel()
        try:
            await state_manager._snapshot_task
        except asyncio.CancelledError:
            pass
        state_manager._snapshot_task = None
        logger.info("Stopped periodic snapshots")


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
