"""Strategy execution helpers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..exceptions import OrchestratorError, StrategyError
from ..shared import InstanceResult
from .event_bus import EventBus
from .strategies import AVAILABLE_STRATEGIES
from .strategies.loader import load_strategy
from .strategy_context import StrategyContext

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short8 = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{short8}"


def _prepare_event_bus(orchestrator, run_id: str) -> None:
    event_log_path = orchestrator.logs_dir / run_id / "events.jsonl"
    if not orchestrator.event_bus:
        orchestrator.event_bus = EventBus(
            max_events=orchestrator.event_buffer_size,
            persist_path=event_log_path,
            run_id=run_id,
        )
    else:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        if orchestrator.event_bus._persist_file:
            orchestrator.event_bus.close()
        orchestrator.event_bus.persist_path = event_log_path
        orchestrator.event_bus._open_persist_file()
    try:
        if orchestrator._pending_redaction_patterns:
            orchestrator.event_bus.set_custom_redaction_patterns(
                list(orchestrator._pending_redaction_patterns)
            )
    except Exception:
        pass


def _detect_default_workspace_branches(orchestrator, repo_path: Path, base_branch: str) -> None:
    if getattr(orchestrator, "default_workspace_include_branches", None) not in (None, []):
        return
    try:
        import subprocess as sp

        proc = sp.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        head_branch = (proc.stdout or "").strip()
        if head_branch and head_branch != "HEAD" and head_branch != base_branch:
            orchestrator.default_workspace_include_branches = [head_branch]
        else:
            orchestrator.default_workspace_include_branches = None
    except Exception:
        orchestrator.default_workspace_include_branches = None


def _resolve_strategy(strategy_name: str) -> Tuple[Any, str]:
    if strategy_name in AVAILABLE_STRATEGIES:
        return AVAILABLE_STRATEGIES[strategy_name], strategy_name
    strategy_class = load_strategy(strategy_name)
    try:
        effective_name = strategy_class().name
    except Exception:
        effective_name = strategy_name
    return strategy_class, effective_name


def _emit_strategy_started(
    orchestrator,
    strategy_id: str,
    strategy,
    effective_name: str,
    run_id: str,
    config: Optional[Dict[str, Any]],
) -> None:
    orchestrator.event_bus.emit(
        "strategy.started",
        {
            "strategy_id": strategy_id,
            "strategy_name": getattr(strategy, "name", effective_name),
            "config": config,
        },
    )
    orchestrator.event_bus.emit_canonical(
        type="strategy.started",
        run_id=run_id,
        strategy_execution_id=strategy_id,
        payload={
            "name": getattr(strategy, "name", effective_name),
            "params": getattr(strategy, "config", None) or {},
        },
    )


def _emit_strategy_completed(orchestrator, strategy_id: str, results: List[InstanceResult], run_id: str) -> None:
    orchestrator.event_bus.emit(
        "strategy.completed",
        {
            "strategy_id": strategy_id,
            "result_count": len(results),
            "branch_names": [r.branch_name for r in results if r.branch_name],
        },
    )
    any_success = any(getattr(r, "success", False) for r in (results or []))
    if not any_success and orchestrator._shutdown:
        status = "canceled"
        payload = {"status": status, "reason": "operator_interrupt"}
    else:
        status = "success" if any_success else "failed"
        payload = {"status": status}
        if status == "failed":
            payload["reason"] = "no_successful_tasks"
    orchestrator.event_bus.emit_canonical(
        type="strategy.completed",
        run_id=run_id,
        strategy_execution_id=strategy_id,
        payload=payload,
    )


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

    _emit_strategy_started(
        orchestrator, strategy_id, strategy, effective_name, run_id, strategy_config
    )
    results = await strategy.execute(
        prompt=prompt,
        base_branch=base_branch,
        ctx=ctx,
    )
    state_value = "completed"
    if results and all((not r.success) for r in results):
        state_value = "failed"
    orchestrator.state_manager.update_strategy_state(
        strategy_id=strategy_id,
        state=state_value,
        results=results,
    )
    _emit_strategy_completed(orchestrator, strategy_id, results, run_id)
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
    for run_idx in range(runs):
        strat = strategy_class()
        strat.set_config_overrides(strategy_config or {})
        strat.create_config()
        strategy_id = str(uuid.uuid4())
        orchestrator.state_manager.register_strategy(
            strategy_id=strategy_id,
            strategy_name=getattr(strat, "name", effective_strategy_name),
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
    for res in results:
        if isinstance(res, Exception):
            logger.error("Strategy execution failed: %s", res)
            continue
        aggregated.extend(res)
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
    strat = strategy_class()
    strat.set_config_overrides(strategy_config or {})
    strat.create_config()
    strategy_id = str(uuid.uuid4())
    orchestrator.state_manager.register_strategy(
        strategy_id=strategy_id,
        strategy_name=getattr(strat, "name", effective_strategy_name),
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
    run_id = run_id or _generate_run_id()
    _prepare_event_bus(orchestrator, run_id)
    orchestrator.state_manager.event_bus = orchestrator.event_bus

    state = orchestrator.state_manager.initialize_run(
        run_id=run_id,
        prompt=prompt,
        repo_path=repo_path,
        base_branch=base_branch,
    )

    _detect_default_workspace_branches(orchestrator, repo_path, base_branch)
    orchestrator._force_import = bool((strategy_config or {}).get("force_import", False))
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
        strategy_class, effective_strategy_name = _resolve_strategy(strategy_name)
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
