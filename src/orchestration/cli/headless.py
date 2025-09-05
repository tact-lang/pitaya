"""Headless execution paths (streaming, JSON, quiet)."""

from __future__ import annotations

import json
from typing import Any, Dict

from rich.console import Console

from ...orchestration import Orchestrator
from .strategy_config import get_strategy_config
from .results_display import display_detailed_results

__all__ = ["run_headless"]


def _subscribe_streaming(console: Console, orch: Orchestrator, args) -> None:
    def emit(ev: dict) -> None:
        et = ev.get("type", "")
        data = ev.get("data", {})
        if et.endswith("started"):
            what = data.get("strategy") or data.get("activity") or et
            console.print(f"▶ {what}")
        elif et.endswith("completed"):
            dur = data.get("metrics", {}).get("duration_seconds") or data.get(
                "duration_seconds"
            )
            cost = data.get("metrics", {}).get("total_cost")
            pieces = ["✅ completed"]
            if dur:
                pieces.append(f"time={dur:.1f}s")
            if cost is not None:
                pieces.append(f"cost=${float(cost):.2f}")
            console.print(" ".join(pieces))
        elif et.endswith("failed"):
            msg = (
                (data.get("message") or data.get("error") or "")
                .strip()
                .replace("\n", " ")
            )
            console.print(f"❌ failed {msg[:200]}")

    for t in (
        "task.started",
        "task.completed",
        "task.failed",
        "strategy.started",
        "strategy.completed",
    ):
        orch.subscribe(t, emit)


def _subscribe_json(orch: Orchestrator) -> None:
    def emit(ev: dict) -> None:
        print(json.dumps(ev, separators=(",", ":")))

    for t in (
        "task.scheduled",
        "task.started",
        "task.progress",
        "task.completed",
        "task.failed",
        "task.interrupted",
        "strategy.started",
        "strategy.completed",
    ):
        orch.subscribe(t, emit)


async def _execute(orch: Orchestrator, args, full_config: Dict[str, Any], run_id: str):
    if getattr(args, "resume", None):
        return await orch.resume_run(run_id)
    cfg = get_strategy_config(args, full_config)
    return await orch.run_strategy(
        strategy_name=args.strategy,
        prompt=args.prompt or "",
        repo_path=args.repo,
        base_branch=args.base_branch,
        runs=args.runs,
        strategy_config=cfg,
        run_id=run_id,
    )


def _exit_code(results) -> int:
    try:
        return 3 if any((not r.success) for r in results) else 0
    except Exception:
        return 0


async def run_headless(
    console: Console, orch: Orchestrator, args, full_config: Dict[str, Any], run_id: str
) -> int:
    mode = getattr(args, "output", "streaming")
    if mode == "streaming":
        _subscribe_streaming(console, orch, args)
    elif mode == "json":
        _subscribe_json(orch)
    # quiet: no subscriptions

    results = await _execute(orch, args, full_config, run_id)
    if mode != "quiet":
        # Try to get state (best-effort)
        state = getattr(
            getattr(orch, "state_manager", None), "get_current_state", lambda: None
        )()
        display_detailed_results(
            console, results, state.run_id if state else run_id, state
        )
    return _exit_code(results)
