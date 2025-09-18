"""Display results for headless mode in a structured way."""

from __future__ import annotations

from typing import Any, Iterable, List
from ...shared import InstanceResult, InstanceStatus

from rich.console import Console

__all__ = ["display_detailed_results"]


def _fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "-"
    if seconds >= 60:
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes}m {sec}s"
    return f"{seconds:.0f}s"


def _print_instance(console: Console, result) -> None:
    status = "✓" if getattr(result, "success", False) else "✗"
    duration = _fmt_duration(getattr(result, "duration_seconds", None))
    metrics = getattr(result, "metrics", None) or {}
    cost = f"${metrics.get('total_cost', 0):.2f}"
    tokens = metrics.get("total_tokens", 0)
    token_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)
    line = f"  {status} {result.branch_name or 'no-branch'}  {duration} • {cost} • {token_str} tokens"
    if getattr(result, "success", False):
        console.print(line, style="green" if status == "✓" else "red")
    else:
        etype = getattr(result, "error_type", None) or ""
        emsg = getattr(result, "error", "Unknown error") or "Unknown error"
        if etype:
            console.print(f"{line}  [red]Failed ({etype}): {emsg}[/red]")
        else:
            console.print(f"{line}  [red]Failed: {emsg}[/red]")
        # Surface log path if available for quick debugging
        lpath = getattr(result, "log_path", None)
        if lpath:
            console.print(f"    [dim]log:[/dim] {lpath}")

    md = getattr(result, "metadata", None) or {}
    pieces: list[str] = []
    for key in ("score", "complexity", "test_coverage"):
        if key in md:
            pieces.append(f"{key}={md[key]}")
    if pieces:
        console.print(f"    metadata: {', '.join(pieces)}")
    msg = getattr(result, "final_message", None)
    if msg:
        msg = (msg[:497] + "...") if len(msg) > 500 else msg
        console.print(f"    [dim]final_message:[/dim] {msg}")


def _summary(console: Console, results: Iterable, state: Any | None) -> None:
    # Normalize results to a list so we can iterate multiple times without
    # exhausting a generator passed by the caller.
    results_list = list(results)

    console.print("[bold]Summary:[/bold]")
    if state and hasattr(state, "strategies") and state.strategies:
        strat_states = [getattr(s, "state", "") for s in state.strategies.values()]
        strat_success = sum(1 for s in strat_states if s == "completed")
        strat_canceled = sum(1 for s in strat_states if s == "canceled")
        strat_failed = sum(1 for s in strat_states if s == "failed")
        console.print(
            f"  Strategies: {strat_success}/{len(strat_states)} completed; {strat_canceled} canceled; {strat_failed} failed"
        )
        if strat_canceled > 0 and getattr(state, "run_id", None):
            console.print("\n[blue]Run interrupted. To resume this run:[/blue]")
            console.print(f"  pitaya --resume {state.run_id}")

    # Prefer authoritative run-level aggregates from state when available.
    total_duration_seconds = None
    total_cost = None
    total_tokens = None
    success_count = None
    int_count = None
    failed_count = None
    total_count = None

    if state is not None:
        try:
            if getattr(state, "started_at", None) and getattr(
                state, "completed_at", None
            ):
                total_duration_seconds = max(
                    0.0,
                    (state.completed_at - state.started_at).total_seconds(),
                )
        except Exception:
            total_duration_seconds = None

        total_cost = getattr(state, "total_cost", None)
        total_tokens = getattr(state, "total_tokens", None)

        try:
            inst_infos = list(getattr(state, "instances", {}).values())
            if inst_infos:
                success_count = sum(
                    1 for i in inst_infos if i.state == InstanceStatus.COMPLETED
                )
                int_count = sum(
                    1 for i in inst_infos if i.state == InstanceStatus.INTERRUPTED
                )
                failed_count = sum(
                    1 for i in inst_infos if i.state == InstanceStatus.FAILED
                )
                total_count = len(inst_infos)
        except Exception:
            success_count = int_count = failed_count = total_count = None

    # Fall back to per-result aggregation when state data is absent.
    if total_duration_seconds is None:
        total_duration_seconds = sum(
            getattr(r, "duration_seconds", 0) or 0 for r in results_list
        )
    if total_cost is None:
        total_cost = sum(
            getattr(r, "metrics", {}).get("total_cost", 0) for r in results_list
        )
    if total_tokens is None:
        total_tokens = sum(
            getattr(r, "metrics", {}).get("total_tokens", 0) for r in results_list
        )
    if success_count is None:
        success_count = sum(1 for r in results_list if getattr(r, "success", False))
    if int_count is None:
        int_count = sum(
            1 for r in results_list if getattr(r, "status", "") == "canceled"
        )
    if failed_count is None:
        failed_count = sum(
            1
            for r in results_list
            if (
                not getattr(r, "success", False)
                and getattr(r, "status", "") != "canceled"
            )
        )
    if total_count is None:
        total_count = len(results_list)

    duration_str = _fmt_duration(total_duration_seconds)
    console.print(f"  Total Duration: {duration_str}")
    console.print(f"  Total Cost: ${total_cost:.2f}")
    console.print(f"  Total Tokens: {total_tokens:,}")
    console.print(
        f"  Instances: {success_count} succeeded, {int_count} canceled, {failed_count} failed (total {total_count})"
    )


def display_detailed_results(
    console: Console,
    results: List[InstanceResult],
    run_id: str,
    state: Any | None = None,
) -> None:
    """Display detailed results: header, grouped details, summary, and tips."""
    console.print(f"\n═══ Run Complete: {run_id} ═══\n")

    if state and hasattr(state, "strategies"):
        # Group by strategy if we have richer state; capture orphans too
        grouped: dict[str, list] = {}
        orphans: list = []
        for r in results:
            sid = getattr(r, "metadata", {}).get("strategy_execution_id")
            if sid:
                grouped.setdefault(sid, []).append(r)
            else:
                orphans.append(r)

        for strat_id, strat_info in state.strategies.items():
            console.print(f"[bold]Strategy: {strat_info.strategy_name}[/bold]")
            for key, value in (strat_info.config or {}).items():
                console.print(f"  {key}: {value}")
            console.print()
        console.print(f"[bold]Runs:[/bold] {len(state.strategies)}\n")
        console.print("[bold]Results by strategy:[/bold]\n")
        # When there is only one strategy, fall back to printing orphans under it
        single_strategy = len(state.strategies) == 1
        for idx, (sid, strat_info) in enumerate(state.strategies.items(), 1):
            console.print(f"[bold]Strategy #{idx} ({strat_info.strategy_name}):[/bold]")
            printed = False
            for r in grouped.get(sid, []):
                printed = True
                _print_instance(console, r)
            if single_strategy and not printed and orphans:
                for r in orphans:
                    _print_instance(console, r)
                printed = True
                orphans = []
            if not printed:
                console.print("  (no results)")
            console.print()
        # If any orphans remain (multi-strategy), render them in a separate section
        if orphans:
            console.print("[bold]Ungrouped Results:[/bold]")
            for r in orphans:
                _print_instance(console, r)
            console.print()
    else:
        for r in results:
            _print_instance(console, r)

    _summary(console, results, state)
