"""TUI execution path: drive orchestrator and display."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
import argparse

from rich.console import Console

from ...tui.display import TUIDisplay
from ...orchestration import Orchestrator
from .strategy_config import get_strategy_config

__all__ = ["run_tui"]


async def _start_orchestrator(
    orch: Orchestrator, args: argparse.Namespace, run_id: str, cfg: Dict[str, Any]
) -> Any:
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
    except (TypeError, ValueError, AttributeError):
        rr = 0.1
    return TUIDisplay(console=console, refresh_rate=rr, state_poll_interval=3.0)


async def run_tui(
    console: Console,
    orch: Orchestrator,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    run_id: str,
) -> int:
    display = _display(console, cfg)
    # Apply CLI display flags to TUI to match previous behavior
    if getattr(args, "display", None) and args.display != "auto":
        display.set_forced_display_mode(args.display)
    display.set_ids_full(getattr(args, "show_ids", "short") == "full")
    events_file = args.logs_dir / run_id / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)

    orch_task = asyncio.create_task(_start_orchestrator(orch, args, run_id, cfg))
    tui_task = asyncio.create_task(
        display.run(orchestrator=orch, events_file=events_file, from_offset=0)
    )

    try:
        # Wait until either task fails or the orchestrator completes
        done, pending = await asyncio.wait(
            {orch_task, tui_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # If the TUI task failed, stop everything and report a friendly error
        if tui_task in done and tui_task.exception() is not None:
            exc = tui_task.exception()
            # Cancel orchestrator if still running
            if not orch_task.done():
                orch_task.cancel()
                try:
                    await orch_task
                except Exception:
                    pass
            await display.stop()
            try:
                await tui_task
            except Exception:
                pass
            console.print(f"[red]TUI error:[/red] {exc}")
            return 1

        # Otherwise, ensure the orchestrator finished (success or error)
        if orch_task in done and orch_task.exception() is not None:
            exc = orch_task.exception()
            await display.stop()
            # Ensure the TUI task is fully torn down to avoid pending-task warnings
            try:
                await tui_task
            except Exception:
                pass
            # Let outer runner show a friendly message too, but give a hint here
            console.print(f"[red]Run failed:[/red] {exc}")
            return 1
        results = await orch_task

        # Stop TUI cleanly now that orchestration is complete
        await display.stop()
        try:
            await tui_task
        except asyncio.CancelledError:
            pass

        # Print a final summary to console for visibility after the TUI closes
        from .results_display import display_detailed_results

        state = orch.get_current_state() if hasattr(orch, "get_current_state") else None
        rid = getattr(state, "run_id", run_id)
        display_detailed_results(console, results, rid, state)
        # Exit 3 only if any instance actually failed (ignore canceled)
        for r in results:
            if (
                not getattr(r, "success", False)
                and getattr(r, "status", "") != "canceled"
            ):
                return 3
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Stop the TUI and ensure Live is torn down, then print hint and exit 2
        try:
            await display.stop()
            try:
                await tui_task
            except Exception:
                pass
        except Exception:
            pass
        # Print resume hint here (terminal restored), unless in JSON mode
        try:
            is_json = bool(
                getattr(args, "json", False) or getattr(args, "output", "") == "json"
            )
            if not is_json and run_id:
                console.print(
                    "\n[yellow]Interrupted — shutting down gracefully[/yellow]"
                )
                console.print(f"[blue]Resume:[/blue] pitaya --resume {run_id}")
        except Exception:
            pass
        return 2
    except Exception as e:
        # Generic failure: stop TUI first, then surface the error
        await display.stop()
        try:
            await tui_task
        except Exception:
            pass
        console.print(f"[red]Run failed:[/red] {e}")
        return 1
