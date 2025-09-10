"""Headless execution paths (streaming, JSON, quiet)."""

from __future__ import annotations

import json
import argparse
from typing import Any, Dict, Callable
from functools import partial

from rich.console import Console

from ...orchestration import Orchestrator
from .strategy_config import get_strategy_config
from .results_display import display_detailed_results

__all__ = ["run_headless"]


def _make_glyph(no_emoji: bool) -> Callable[[str], str]:
    if no_emoji:
        return lambda _n: ""
    table = {"started": " ▶", "completed": " ✅", "failed": " ❌", "interrupted": " ⏸"}
    return lambda n: table.get(n, "")


def _make_prefix(show_full: bool) -> Callable[[dict], str]:
    import hashlib

    def _prefix(ev: dict) -> str:
        sid = ev.get("strategy_execution_id") or ev.get("data", {}).get(
            "strategy_execution_id", ""
        )
        key = ev.get("key") or ev.get("data", {}).get("key", "")
        k8 = (
            hashlib.sha256(f"{sid}|{key}".encode("utf-8", errors="ignore")).hexdigest()[
                :8
            ]
            if (sid or key)
            else "????????"
        )
        iid = ev.get("instance_id") or ev.get("data", {}).get("instance_id", "")
        inst = iid if show_full else (iid[:5] if iid else "?????")
        return f"[k{k8}][inst-{inst}] "

    return _prefix


def _handle_scheduled(
    console: Console,
    prefix: Callable[[dict], str],
    glyph: Callable[[str], str],
    verbose: bool,
    ev: dict,
) -> None:
    if verbose:
        console.print(f"{prefix(ev)}scheduled{glyph('started')}")


def _handle_started(
    console: Console,
    prefix: Callable[[dict], str],
    glyph: Callable[[str], str],
    ev: dict,
) -> None:
    data = ev.get("data", {})
    model = data.get("model")
    base = data.get("base_branch")
    parts = [f"{prefix(ev)}started{glyph('started')}"]
    if model:
        parts.append(f"model={model}")
    if base:
        parts.append(f"base={base}")
    console.print(" ".join(parts))


def _handle_progress(console: Console, prefix: Callable[[dict], str], ev: dict) -> None:
    data = ev.get("data", {})
    activity = data.get("activity") or data.get("tool") or ev.get("type")
    console.print(f"{prefix(ev)}{activity}")


def _handle_completed(
    console: Console,
    prefix: Callable[[dict], str],
    glyph: Callable[[str], str],
    ev: dict,
) -> None:
    data = ev.get("data", {})
    metrics = data.get("metrics", {})
    dur = metrics.get("duration_seconds") or data.get("duration_seconds")
    cost = metrics.get("total_cost")
    toks = metrics.get("total_tokens")
    artifact = data.get("artifact") or {}
    branch = artifact.get("branch_final") or artifact.get("branch_planned")
    parts = [f"{prefix(ev)}completed{glyph('completed')}"]
    try:
        if dur is not None:
            parts.append(f"time={float(dur):.1f}s")
    except (TypeError, ValueError):
        pass
    try:
        if cost is not None:
            parts.append(f"cost=${float(cost):.2f}")
    except (TypeError, ValueError):
        pass
    if toks is not None:
        parts.append(f"tok={toks}")
    if branch:
        parts.append(f"branch={branch}")
    console.print(" ".join(parts))


def _handle_failed(
    console: Console,
    prefix: Callable[[dict], str],
    glyph: Callable[[str], str],
    ev: dict,
) -> None:
    data = ev.get("data", {})
    etype = data.get("error_type") or "unknown"
    msg = (data.get("message") or data.get("error") or "").strip().replace("\n", " ")
    line = f"{prefix(ev)}failed{glyph('failed')} type={etype}"
    if msg:
        line += f' message="{msg[:200]}"'
    console.print(line)


def _subscribe_streaming(
    console: Console, orch: Orchestrator, args: argparse.Namespace
) -> None:
    glyph = _make_glyph(bool(getattr(args, "no_emoji", False)))
    prefix = _make_prefix(getattr(args, "show_ids", "short") == "full")
    verbose = bool(getattr(args, "verbose", False))

    handlers: dict[str, Callable[[dict], None]] = {
        "task.started": partial(_handle_started, console, prefix, glyph),
        "task.progress": partial(_handle_progress, console, prefix),
        "task.completed": partial(_handle_completed, console, prefix, glyph),
        "task.failed": partial(_handle_failed, console, prefix, glyph),
        "task.scheduled": partial(_handle_scheduled, console, prefix, glyph, verbose),
    }
    event_types = [
        "task.started",
        "task.completed",
        "task.failed",
        "strategy.started",
        "strategy.completed",
    ]
    if verbose:
        event_types.extend(["task.scheduled", "task.progress"])
    for t in event_types:
        orch.subscribe(t, handlers.get(t, lambda _e: None))


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
    for r in results:
        status = getattr(r, "status", "")
        success = getattr(r, "success", False)
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
    mode = getattr(args, "output", "streaming")
    if mode == "streaming":
        _subscribe_streaming(console, orch, args)
    elif mode == "json":
        _subscribe_json(orch)
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
