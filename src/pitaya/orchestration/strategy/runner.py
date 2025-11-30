"""Strategy execution helpers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pitaya.exceptions import OrchestratorError, StrategyError
from pitaya.shared import InstanceResult
from pitaya.orchestration.strategy.context import StrategyContext
from pitaya.orchestration.strategy.utils import (
    detect_default_workspace_branches,
    emit_strategy_completed,
    emit_strategy_failed,
    emit_strategy_started,
    generate_run_id,
    prepare_event_bus,
    resolve_strategy,
)

logger = logging.getLogger(__name__)


async def _execute_strategy(
    orchestrator,
    *,
    strategy_class,
    strategy_id: str,
    effective_name: str,
    prompt: str,
    base_branch: str,
    run_id: str,
    strategy_config: Optional[Dict[str, Any]],
) -> List[InstanceResult]:
    strategy = strategy_class()
    strategy.set_config_overrides(strategy_config or {})
    strategy.create_config()

    ctx = StrategyContext(
        orchestrator=orchestrator,
        strategy_name=strategy.name,
        strategy_execution_id=strategy_id,
    )
    orchestrator.repo_path = orchestrator.state_manager.current_state.repo_path

    emit_strategy_started(
        orchestrator, strategy_id, strategy, effective_name, run_id, strategy_config
    )
    try:
        results = await strategy.execute(
            prompt=prompt, base_branch=base_branch, ctx=ctx
        )
    except Exception as exc:
        orchestrator.state_manager.update_strategy_state(
            strategy_id=strategy_id,
            state="failed",
        )
        emit_strategy_failed(orchestrator, strategy_id, run_id, str(exc))
        raise
    state_value = (
        "completed"
        if any(getattr(r, "success", False) for r in (results or []))
        else "failed"
    )
    orchestrator.state_manager.update_strategy_state(
        strategy_id=strategy_id,
        state=state_value,
        results=results,
    )
    emit_strategy_completed(orchestrator, strategy_id, results, run_id)
    return results


async def _run_multiple(
    orchestrator,
    *,
    strategy_class,
    effective_strategy_name: str,
    prompt: str,
    base_branch: str,
    run_id: str,
    strategy_config: Optional[Dict[str, Any]],
    runs: int,
) -> List[InstanceResult]:
    tasks: List[asyncio.Task] = []
    strategy_ids: List[str] = []
    for _ in range(runs):
        strategy_id = str(uuid.uuid4())
        strategy_ids.append(strategy_id)
        orchestrator.state_manager.register_strategy(
            strategy_id=strategy_id,
            strategy_name=effective_strategy_name,
            config=strategy_config or {},
        )
        tasks.append(
            asyncio.create_task(
                _execute_strategy(
                    orchestrator,
                    strategy_class=strategy_class,
                    strategy_id=strategy_id,
                    effective_name=effective_strategy_name,
                    prompt=prompt,
                    base_branch=base_branch,
                    run_id=run_id,
                    strategy_config=strategy_config,
                )
            )
        )
    aggregated: List[InstanceResult] = []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures: List[str] = []
    for strat_id, res in zip(strategy_ids, results):
        if isinstance(res, Exception):
            logger.error("Strategy execution failed: %s", res)
            orchestrator.state_manager.update_strategy_state(
                strategy_id=strat_id, state="failed"
            )
            emit_strategy_failed(orchestrator, strat_id, run_id, str(res))
            failures.append(strat_id)
            continue
        aggregated.extend(res)
    if failures:
        raise OrchestratorError(f"Strategies failed: {', '.join(failures)}")
    return aggregated


async def _run_single(
    orchestrator,
    *,
    strategy_class,
    effective_strategy_name: str,
    prompt: str,
    base_branch: str,
    run_id: str,
    strategy_config: Optional[Dict[str, Any]],
) -> List[InstanceResult]:
    strategy_id = str(uuid.uuid4())
    orchestrator.state_manager.register_strategy(
        strategy_id=strategy_id,
        strategy_name=effective_strategy_name,
        config=strategy_config or {},
    )
    return await _execute_strategy(
        orchestrator,
        strategy_class=strategy_class,
        strategy_id=strategy_id,
        effective_name=effective_strategy_name,
        prompt=prompt,
        base_branch=base_branch,
        run_id=run_id,
        strategy_config=strategy_config,
    )


async def run_strategy(
    orchestrator,
    strategy_name: str,
    prompt: str,
    repo_path: Path,
    base_branch: str = "main",
    strategy_config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    runs: int = 1,
) -> List[InstanceResult]:
    run_id = run_id or generate_run_id()
    prepare_event_bus(orchestrator, run_id)
    orchestrator.state_manager.event_bus = orchestrator.event_bus

    state = orchestrator.state_manager.initialize_run(
        run_id=run_id,
        prompt=prompt,
        repo_path=repo_path,
        base_branch=base_branch,
    )

    detect_default_workspace_branches(orchestrator, repo_path, base_branch)
    orchestrator._force_import = bool(
        (strategy_config or {}).get("force_import", False)
    )
    await orchestrator.state_manager.start_periodic_snapshots()

    orchestrator.event_bus.emit(
        "run.started",
        {
            "run_id": run_id,
            "strategy": strategy_name,
            "prompt": prompt,
            "repo_path": str(repo_path),
            "base_branch": base_branch,
        },
    )

    try:
        strategy_class, effective_strategy_name = resolve_strategy(strategy_name)
        if runs > 1:
            results = await _run_multiple(
                orchestrator,
                strategy_class=strategy_class,
                effective_strategy_name=effective_strategy_name,
                prompt=prompt,
                base_branch=base_branch,
                run_id=run_id,
                strategy_config=strategy_config,
                runs=runs,
            )
        else:
            results = await _run_single(
                orchestrator,
                strategy_class=strategy_class,
                effective_strategy_name=effective_strategy_name,
                prompt=prompt,
                base_branch=base_branch,
                run_id=run_id,
                strategy_config=strategy_config,
            )
        await orchestrator.save_results(run_id, results)
        return results
    except (StrategyError, OrchestratorError, asyncio.CancelledError) as exc:
        logger.exception("Strategy execution failed: %s", exc)
        orchestrator.event_bus.emit(
            "run.failed",
            {
                "run_id": run_id,
                "error": str(exc),
            },
        )
        raise
    finally:
        await orchestrator.state_manager.stop_periodic_snapshots()
        if state:
            state.completed_at = datetime.now(timezone.utc)
            await orchestrator.state_manager.save_snapshot()
            try:
                latest_results: List[InstanceResult] = []
                for _iid, _info in (state.instances or {}).items():
                    if getattr(_info, "result", None):
                        latest_results.append(_info.result)  # type: ignore[attr-defined]
                await orchestrator.save_results(run_id, latest_results)
            except Exception:
                pass
        orchestrator.event_bus.emit(
            "run.completed",
            {
                "run_id": run_id,
                "duration_seconds": (
                    (state.completed_at - state.started_at).total_seconds()
                    if state
                    else 0
                ),
                "total_cost": state.total_cost if state else 0,
                "total_tokens": state.total_tokens if state else 0,
            },
        )
