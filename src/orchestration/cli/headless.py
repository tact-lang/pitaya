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
    import hashlib

    no_emoji = bool(getattr(args, "no_emoji", False))
    show_full = getattr(args, "show_ids", "short") == "full"
    verbose = bool(getattr(args, "verbose", False))

    def _glyph(name: str) -> str:
        if no_emoji:
            return ""
        return {
            "started": " ▶",
            "completed": " ✅",
            "failed": " ❌",
            "interrupted": " ⏸",
        }.get(name, "")

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

    def emit(ev: dict) -> None:
        et = ev.get("type", "")
        data = ev.get("data", {})
        pre = _prefix(ev)
        if et == "task.scheduled":
            if verbose:
                console.print(f"{pre}scheduled{_glyph('started')}")
            return
        if et == "task.started":
            model = data.get("model")
            base = data.get("base_branch")
            line = f"{pre}started{_glyph('started')}"
            if model:
                line += f" model={model}"
            if base:
                line += f" base={base}"
            console.print(line)
        elif et == "task.progress":
            if verbose:
                activity = data.get("activity") or data.get("tool") or ev.get("type")
                console.print(f"{pre}{activity}")
        elif et == "task.completed":
            dur = data.get("metrics", {}).get("duration_seconds") or data.get(
                "duration_seconds"
            )
            cost = data.get("metrics", {}).get("total_cost")
            toks = data.get("metrics", {}).get("total_tokens")
            branch = (data.get("artifact") or {}).get("branch_final") or (
                data.get("artifact") or {}
            ).get("branch_planned")
            parts = [f"{pre}completed{_glyph('completed')}"]
            if dur is not None:
                try:
                    parts.append(f"time={float(dur):.1f}s")
                except Exception:
                    pass
            if cost is not None:
                try:
                    parts.append(f"cost=${float(cost):.2f}")
                except Exception:
                    pass
            if toks is not None:
                parts.append(f"tok={toks}")
            if branch:
                parts.append(f"branch={branch}")
            console.print(" ".join(parts))
        elif et == "task.failed":
            etype = data.get("error_type") or "unknown"
            msg = (
                (data.get("message") or data.get("error") or "")
                .strip()
                .replace("\n", " ")
            )
            line = f"{pre}failed{_glyph('failed')} type={etype}"
            if msg:
                line += f' message="{msg[:200]}"'
            console.print(line)

    types = [
        "task.started",
        "task.completed",
        "task.failed",
        "strategy.started",
        "strategy.completed",
    ]
    if verbose:
        types.extend(["task.scheduled", "task.progress"])
    for t in types:
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
