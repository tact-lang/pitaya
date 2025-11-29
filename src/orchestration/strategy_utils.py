"""Utility helpers shared by strategy runner logic."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .event_bus import EventBus
from .strategies import AVAILABLE_STRATEGIES
from .strategies.loader import load_strategy

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short8 = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{short8}"


def prepare_event_bus(orchestrator, run_id: str) -> None:
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


def detect_default_workspace_branches(orchestrator, repo_path: Path, base_branch: str) -> None:
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


def resolve_strategy(strategy_name: str) -> Tuple[Any, str]:
    if strategy_name in AVAILABLE_STRATEGIES:
        return AVAILABLE_STRATEGIES[strategy_name], strategy_name
    strategy_class = load_strategy(strategy_name)
    try:
        effective_name = strategy_class().name
    except Exception:
        effective_name = strategy_name
    return strategy_class, effective_name


def emit_strategy_started(
    orchestrator, strategy_id: str, strategy, effective_name: str, run_id: str, config: Optional[Dict[str, Any]]
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


def emit_strategy_completed(orchestrator, strategy_id: str, results, run_id: str) -> None:
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
