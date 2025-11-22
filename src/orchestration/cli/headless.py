"""Headless execution paths (streaming, JSON, quiet)."""

from __future__ import annotations

import json
import argparse
from functools import partial
from typing import Any, Dict, Callable

from rich.console import Console
from rich.text import Text

from ...orchestration import Orchestrator
from .strategy_config import get_strategy_config
from .results_display import display_detailed_results

__all__ = ["run_headless"]


def _make_prefix(show_full: bool) -> Callable[[dict], str]:
    def _payload(ev: dict) -> dict:
        return ev.get("data") or ev.get("payload") or {}

    def _prefix(ev: dict) -> str:
        data = _payload(ev)
        iid = (
            ev.get("instance_id")
            or data.get("instance_id")
            or data.get("data", {}).get("instance_id", "")
        )
        inst = iid if show_full else (iid[:5] if iid else "")
        sid = (
            ev.get("strategy_execution_id")
            or data.get("strategy_execution_id")
            or data.get("data", {}).get("strategy_execution_id", "")
        )
        parts = []
        if inst:
            parts.append(f"inst={inst}")
        if sid:
            parts.append(f"sid={sid[:6]}")
        return " ".join(parts) + (" " if parts else "")

    return _prefix


STATUS_STYLE = {
    "queued": "yellow",
    "started": "cyan",
    "progress": "bright_black",
    "assistant": "green",
    "tool": "magenta",
    "tool_result": "magenta",
    "completed": "green",
    "failed": "red",
}


def _render_line(
    kind: str,
    prefix: str,
    status: str,
    params: list[tuple[str, str]],
    extras: list[tuple[str, str]],
) -> Text:
    """Build a consistently-styled Rich Text line."""
    t = Text("", justify="left", no_wrap=True)
    if prefix:
        t.append(prefix.strip(), style="dim")
    if status:
        t.append(
            (" " if len(t) else "") + status, style=STATUS_STYLE.get(kind, "white")
        )
    for key, val in params:
        t.append(" ")
        t.append(f"{key}=", style="bright_black")
        t.append(val, style="bright_blue")
    for label, val in extras:
        t.append(" ")
        t.append(f"{label}=", style="bright_black")
        t.append(val, style="white")
    return t


def _handle_scheduled(
    console: Console,
    prefix: Callable[[dict], str],
    prompt_preview: str,
    ev: dict,
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    model = data.get("model")
    base = data.get("base_branch")
    params: list[tuple[str, str]] = []
    extras: list[tuple[str, str]] = []
    if model:
        params.append(("model", str(model)))
    if base:
        params.append(("base", str(base)))
    if prompt_preview:
        extras.append(("prompt", prompt_preview))
    console.print(
        _render_line(
            kind="queued",
            prefix=prefix(ev),
            status="queued",
            params=params,
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_started(
    console: Console,
    prefix: Callable[[dict], str],
    prompt_preview: str,
    ev: dict,
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    model = data.get("model")
    base = data.get("base_branch")
    params: list[tuple[str, str]] = []
    extras: list[tuple[str, str]] = []
    if model:
        params.append(("model", str(model)))
    if base:
        params.append(("base", str(base)))
    if prompt_preview:
        extras.append(("prompt", prompt_preview))
    console.print(
        _render_line(
            kind="started",
            prefix=prefix(ev),
            status="started",
            params=params,
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_progress(console: Console, prefix: Callable[[dict], str], ev: dict) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    activity = data.get("activity") or data.get("tool") or ev.get("type")
    console.print(
        _render_line(
            kind="progress",
            prefix=prefix(ev),
            status=str(activity),
            params=[],
            extras=[],
        ),
        markup=False,
        highlight=False,
    )


def _handle_agent_assistant(
    console: Console, prefix: Callable[[dict], str], ev: dict
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    text = str(data.get("content") or "").strip()
    if text:
        text = text.replace("\n", " ")
    extras = [("msg", text[:200])] if text else []
    console.print(
        _render_line(
            kind="assistant",
            prefix=prefix(ev),
            status="assistant",
            params=[],
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_agent_tool_use(
    console: Console, prefix: Callable[[dict], str], ev: dict
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    tool = data.get("tool") or data.get("action") or "tool"
    cmd = data.get("command") or data.get("input") or ""
    summary = cmd if isinstance(cmd, str) else str(cmd)
    summary = summary.replace("\n", " ") if isinstance(summary, str) else ""
    extras = [("cmd", summary[:160])] if summary else []
    console.print(
        _render_line(
            kind="tool",
            prefix=prefix(ev),
            status=f"tool:{tool}",
            params=[],
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_agent_tool_result(
    console: Console, prefix: Callable[[dict], str], ev: dict
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    cmd = data.get("command") or ""
    out = data.get("output") or ""
    status = "ok" if data.get("success", True) else "err"
    out_str = str(out).replace("\n", " ")
    extras = [("cmd", str(cmd)[:60])] if cmd else []
    if out_str:
        extras.append(("out", out_str[:160]))
    console.print(
        _render_line(
            kind="tool_result",
            prefix=prefix(ev),
            status=f"tool:{status}",
            params=[],
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_completed(
    console: Console,
    prefix: Callable[[dict], str],
    prompt_preview: str,
    ev: dict,
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    metrics = data.get("metrics", {}) or {}
    dur = metrics.get("duration_seconds") or data.get("duration_seconds")
    cost = metrics.get("total_cost")
    artifact = data.get("artifact") or {}
    branch = artifact.get("branch_final") or artifact.get("branch_planned")
    final_msg = (data.get("final_message") or "").strip().replace("\n", " ")
    parts = [f"{prefix(ev)}completed"]
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
    if branch:
        parts.append(f"branch={branch}")
    if prompt_preview:
        parts.append(f"prompt={prompt_preview}")
    if final_msg:
        parts.append(f"final={final_msg[:1000]}")
    console.print(
        _render_line(
            kind="completed",
            prefix=prefix(ev),
            status="completed",
            params=[
                ("branch", branch)
                for branch in [
                    artifact.get("branch_final"),
                    artifact.get("branch_planned"),
                ]
                if branch
            ],
            extras=[
                ("prompt", prompt_preview) if prompt_preview else None,
                ("final", final_msg[:1000]) if final_msg else None,
                ("time", f"{float(dur):.1f}s") if dur is not None else None,
                ("cost", f"${float(cost):.2f}") if cost is not None else None,
            ],
        ),
        markup=False,
        highlight=False,
    )


def _handle_failed(
    console: Console,
    prefix: Callable[[dict], str],
    glyph: Callable[[str], str],
    ev: dict,
) -> None:
    data = ev.get("data", {})
    etype = data.get("error_type") or "unknown"
    msg = (data.get("message") or data.get("error") or "").strip().replace("\n", " ")
    extras = []
    if msg:
        extras.append(("msg", msg[:200]))
    console.print(
        _render_line(
            kind="failed",
            prefix=prefix(ev),
            status=f"failed:{etype}",
            params=[],
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _subscribe_streaming(
    console: Console, orch: Orchestrator, args: argparse.Namespace
) -> None:
    prefix = _make_prefix(getattr(args, "show_ids", "short") == "full")
    verbose = bool(getattr(args, "verbose", False))
    prompt_preview = (args.prompt or "").strip().replace("\n", " ")[:1000]

    handlers: dict[str, Callable[[dict], None]] = {
        "task.started": partial(_handle_started, console, prefix, prompt_preview),
        "task.progress": partial(_handle_progress, console, prefix),
        "task.completed": partial(_handle_completed, console, prefix, prompt_preview),
        "task.failed": partial(_handle_failed, console, prefix),
        "task.scheduled": partial(_handle_scheduled, console, prefix, prompt_preview),
    }
    event_types = [
        "task.scheduled",
        "task.started",
        "task.completed",
        "task.failed",
        "strategy.started",
        "strategy.completed",
    ]
    if verbose:
        event_types.extend(
            [
                "task.progress",
                "instance.agent_assistant",
                "instance.agent_tool_use",
                "instance.agent_tool_result",
                "instance.agent_system",
            ]
        )
        handlers.update(
            {
                "instance.agent_assistant": partial(
                    _handle_agent_assistant, console, prefix
                ),
                "instance.agent_tool_use": partial(
                    _handle_agent_tool_use, console, prefix
                ),
                "instance.agent_tool_result": partial(
                    _handle_agent_tool_result, console, prefix
                ),
                "instance.agent_system": partial(_handle_progress, console, prefix),
            }
        )
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
