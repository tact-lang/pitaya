"""Display results for headless mode in a structured way."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List
from pitaya.shared import InstanceResult, InstanceStatus

from rich.console import Console

__all__ = ["display_detailed_results"]


@dataclass
class Totals:
    duration_seconds: float | None
    total_cost: float | None
    total_tokens: int | None
    success_count: int | None
    canceled_count: int | None
    failed_count: int | None
    total_count: int | None


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
    branch = result.branch_name or "no-branch"
    line = f"  {status} {branch}  {duration} • {cost} • {token_str} tokens"
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


def _print_strategy_status(console: Console, state: Any | None) -> None:
    if state and hasattr(state, "strategies") and state.strategies:
        strat_states = [getattr(s, "state", "") for s in state.strategies.values()]
        strat_success = sum(1 for s in strat_states if s == "completed")
        strat_canceled = sum(1 for s in strat_states if s == "canceled")
        strat_failed = sum(1 for s in strat_states if s == "failed")
        msg = (
            f"  Strategies: {strat_success}/{len(strat_states)} completed; "
            f"{strat_canceled} canceled; {strat_failed} failed"
        )
        console.print(msg)
        if strat_canceled > 0 and getattr(state, "run_id", None):
            console.print("\n[blue]Run interrupted. To resume this run:[/blue]")
            console.print(f"  pitaya --resume {state.run_id}")


def _totals_from_state(state: Any | None) -> Totals | None:
    if state is None:
        return None
    totals = Totals(None, None, None, None, None, None, None)
    try:
        if getattr(state, "started_at", None) and getattr(state, "completed_at", None):
            totals.duration_seconds = max(
                0.0, (state.completed_at - state.started_at).total_seconds()
            )
    except Exception:
        pass

    totals.total_cost = getattr(state, "total_cost", None)
    totals.total_tokens = getattr(state, "total_tokens", None)

    try:
        inst_infos = list(getattr(state, "instances", {}).values())
        if inst_infos:
            totals.success_count = sum(
                1 for i in inst_infos if i.state == InstanceStatus.COMPLETED
            )
            totals.canceled_count = sum(
                1 for i in inst_infos if i.state == InstanceStatus.INTERRUPTED
            )
            totals.failed_count = sum(
                1 for i in inst_infos if i.state == InstanceStatus.FAILED
            )
            totals.total_count = len(inst_infos)
    except Exception:
        pass

    if any(value is not None for value in totals.__dict__.values()):
        return totals
    return None


def _totals_from_results(results_list: list[Any]) -> Totals:
    return Totals(
        duration_seconds=sum(
            getattr(r, "duration_seconds", 0) or 0 for r in results_list
        ),
        total_cost=sum(
            getattr(r, "metrics", {}).get("total_cost", 0) for r in results_list
        ),
        total_tokens=sum(
            getattr(r, "metrics", {}).get("total_tokens", 0) for r in results_list
        ),
        success_count=sum(1 for r in results_list if getattr(r, "success", False)),
        canceled_count=sum(
            1 for r in results_list if getattr(r, "status", "") == "canceled"
        ),
        failed_count=sum(
            1
            for r in results_list
            if (
                not getattr(r, "success", False)
                and getattr(r, "status", "") != "canceled"
            )
        ),
        total_count=len(results_list),
    )


def _coalesce_totals(state_totals: Totals | None, fallback: Totals) -> Totals:
    if state_totals is None:
        return fallback
    return Totals(
        duration_seconds=(
            state_totals.duration_seconds
            if state_totals.duration_seconds is not None
            else fallback.duration_seconds
        ),
        total_cost=(
            state_totals.total_cost
            if state_totals.total_cost is not None
            else fallback.total_cost
        ),
        total_tokens=(
            state_totals.total_tokens
            if state_totals.total_tokens is not None
            else fallback.total_tokens
        ),
        success_count=(
            state_totals.success_count
            if state_totals.success_count is not None
            else fallback.success_count
        ),
        canceled_count=(
            state_totals.canceled_count
            if state_totals.canceled_count is not None
            else fallback.canceled_count
        ),
        failed_count=(
            state_totals.failed_count
            if state_totals.failed_count is not None
            else fallback.failed_count
        ),
        total_count=(
            state_totals.total_count
            if state_totals.total_count is not None
            else fallback.total_count
        ),
    )


def _print_totals(console: Console, totals: Totals) -> None:
    duration_str = _fmt_duration(totals.duration_seconds)
    console.print(f"  Total Duration: {duration_str}")
    console.print(f"  Total Cost: ${totals.total_cost:.2f}")
    console.print(f"  Total Tokens: {totals.total_tokens:,}")
    msg = (
        f"  Instances: {totals.success_count} succeeded, {totals.canceled_count} "
        f"canceled, {totals.failed_count} failed (total {totals.total_count})"
    )
    console.print(msg)


def _summary(console: Console, results: Iterable, state: Any | None) -> None:
    results_list = list(results)
    console.print("[bold]Summary:[/bold]")
    _print_strategy_status(console, state)
    fallback = _totals_from_results(results_list)
    totals = _coalesce_totals(_totals_from_state(state), fallback)
    _print_totals(console, totals)


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
