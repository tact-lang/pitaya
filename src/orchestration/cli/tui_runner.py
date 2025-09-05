"""TUI execution path: drive orchestrator and display."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from rich.console import Console

from ...tui.display import TUIDisplay
from ...orchestration import Orchestrator
from .strategy_config import get_strategy_config

__all__ = ["run_tui"]


async def _start_orchestrator(
    orch: Orchestrator, args, run_id: str, cfg: Dict[str, Any]
):
    if getattr(args, "resume", None):
        return await orch.resume_run(run_id)
    strat_cfg = get_strategy_config(args, cfg)
    return await orch.run_strategy(
        strategy_name=args.strategy,
        prompt=args.prompt or "",
        repo_path=args.repo,
        base_branch=args.base_branch,
        runs=args.runs,
        strategy_config=strat_cfg,
        run_id=run_id,
    )


def _display(console: Console, cfg: Dict[str, Any]) -> TUIDisplay:
    try:
        rr_ms = int(cfg.get("tui", {}).get("refresh_rate_ms", 100))
        rr = max(0.01, rr_ms / 1000.0)
    except Exception:
        rr = 0.1
    return TUIDisplay(console=console, refresh_rate=rr, state_poll_interval=3.0)


async def run_tui(
    console: Console, orch: Orchestrator, args, cfg: Dict[str, Any], run_id: str
) -> int:
    display = _display(console, cfg)
    events_file = args.logs_dir / run_id / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)

    orch_task = asyncio.create_task(_start_orchestrator(orch, args, run_id, cfg))
    tui_task = asyncio.create_task(
        display.run(orchestrator=orch, events_file=events_file, from_offset=0)
    )

    # Wait for orchestrator to finish
    try:
        results = await orch_task
    except Exception:
        # Stop TUI if orchestrator crashed
        await display.stop()
        return 1

    # Stop TUI cleanly now that orchestration is complete
    await display.stop()
    try:
        await tui_task
    except Exception:
        pass

    # Print a final summary to console for visibility after the TUI closes
    try:
        from .results_display import display_detailed_results

        state = orch.get_current_state() if hasattr(orch, "get_current_state") else None
        rid = getattr(state, "run_id", run_id)
        display_detailed_results(console, results, rid, state)
    except Exception:
        pass
    try:
        return 3 if any((not r.success) for r in results) else 0
    except Exception:
        return 0
