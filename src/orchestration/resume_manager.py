"""Resume an interrupted run."""

from __future__ import annotations

import logging
from typing import List

from ..shared import InstanceResult
from .resume_helpers import (
    backfill_terminal,
    mark_run_completed,
    prepare_event_bus,
    reenter_strategies,
    reopen_strategies,
    schedule_instances,
)

logger = logging.getLogger(__name__)


async def resume_run(orchestrator, run_id: str) -> List[InstanceResult]:
    logger.info("Resuming run %s", run_id)
    if not orchestrator._initialized:
        await orchestrator.initialize()

    prepare_event_bus(orchestrator, run_id)
    saved_state = await orchestrator.state_manager.load_and_recover_state(run_id)
    if not saved_state:
        raise ValueError(f"No saved state found for run {run_id}")

    orchestrator.repo_path = saved_state.repo_path

    try:
        counts = {
            "queued": 0,
            "running": 0,
            "interrupted": 0,
            "completed": 0,
            "failed": 0,
        }
        for info in orchestrator.state_manager.current_state.instances.values():
            counts[info.state.value] = counts.get(info.state.value, 0) + 1
        logger.info(
            "resume_run: instances total=%s counts=%s",
            len(orchestrator.state_manager.current_state.instances),
            counts,
        )
    except Exception:
        pass

    backfill_terminal(orchestrator, run_id)
    reopen_strategies(orchestrator, run_id)
    await orchestrator.state_manager.start_periodic_snapshots()

    scheduled_ids = await schedule_instances(orchestrator, saved_state)
    if scheduled_ids:
        logger.info(
            "resume_run: enqueued %s instance(s) for resume", len(scheduled_ids)
        )

    reentry_results = await reenter_strategies(orchestrator, run_id)

    mark_run_completed(orchestrator)
    await orchestrator.state_manager.save_snapshot()

    final_results: List[InstanceResult] = []
    try:
        for strat in orchestrator.state_manager.current_state.strategies.values():
            if strat.results:
                final_results.extend(strat.results)
    except Exception:
        pass
    if not final_results:
        final_results = reentry_results

    await orchestrator.save_results(run_id, final_results)
    orchestrator.event_bus.emit(
        "run.completed",
        {
            "run_id": run_id,
            "resumed": True,
            "cannot_resume": 0,
            "total_results": len(final_results),
        },
    )
    return final_results
