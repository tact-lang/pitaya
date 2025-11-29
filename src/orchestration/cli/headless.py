"""Headless execution paths (streaming, JSON, quiet)."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from rich.console import Console

from ...orchestration import Orchestrator
from .headless_stream import subscribe_json, subscribe_streaming
from .results_display import display_detailed_results
from .strategy_config import get_strategy_config

__all__ = ["run_headless"]


async def _execute(
    orch: Orchestrator,
    args: argparse.Namespace,
    full_config: Dict[str, Any],
    run_id: str,
):
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
    """Return 3 only if any instance actually failed (not canceled)."""

    for result in results:
        status = getattr(result, "status", "")
        success = getattr(result, "success", False)
        if not success and status != "canceled":
            return 3
    return 0


async def run_headless(
    console: Console,
    orch: Orchestrator,
    args: argparse.Namespace,
    full_config: Dict[str, Any],
    run_id: str,
) -> int:
    """Execute orchestration without the TUI, supporting streaming and JSON."""

    mode = getattr(args, "output", "streaming")
    if mode == "streaming":
        subscribe_streaming(console, orch, args)
    elif mode == "json":
        subscribe_json(orch)
    # quiet: no subscriptions

    results = await _execute(orch, args, full_config, run_id)

    # Emit human summary only in streaming mode; JSON mode must remain pure NDJSON
    if mode == "streaming":
        state = None
        if hasattr(orch, "get_current_state"):
            state = orch.get_current_state()  # type: ignore[attr-defined]
        rid = getattr(state, "run_id", run_id)
        display_detailed_results(console, results, rid, state)
    return _exit_code(results)
