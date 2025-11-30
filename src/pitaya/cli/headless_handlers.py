"""Rendering and handler utilities for headless streaming output."""

from __future__ import annotations

from typing import Callable

from rich.console import Console
from rich.text import Text

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

__all__ = [
    "STATUS_STYLE",
    "build_streaming_handlers",
    "make_prefix",
]


def make_prefix(show_full: bool) -> Callable[[dict], str]:
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


def _render_line(
    kind: str,
    prefix: str,
    status: str,
    params: list[tuple[str, str]],
    extras: list[tuple[str, str] | None],
) -> Text:
    """Build a consistently-styled Rich Text line."""

    text = Text("", justify="left", no_wrap=True)
    if prefix:
        text.append(prefix.strip(), style="dim")
    if status:
        text.append(
            (" " if len(text) else "") + status, style=STATUS_STYLE.get(kind, "white")
        )
    for key, val in params:
        text.append(" ")
        text.append(f"{key}=", style="bright_black")
        text.append(val, style="bright_blue")
    for extra in extras:
        if not extra:
            continue
        label, val = extra
        text.append(" ")
        text.append(f"{label}=", style="bright_black")
        text.append(val, style="white")
    return text


def _handle_task_state(
    kind: str,
    status: str,
    console: Console,
    prefix: Callable[[dict], str],
    prompt_preview: str,
    ev: dict,
) -> None:
    data = ev.get("data") or ev.get("payload") or {}
    model = data.get("model")
    base = data.get("base_branch")
    params: list[tuple[str, str]] = []
    extras: list[tuple[str, str] | None] = []
    if model:
        params.append(("model", str(model)))
    if base:
        params.append(("base", str(base)))
    if prompt_preview:
        extras.append(("prompt", prompt_preview))
    console.print(
        _render_line(
            kind=kind,
            prefix=prefix(ev),
            status=status,
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
    text = str(data.get("content") or "").strip().replace("\n", " ")
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
    extras: list[tuple[str, str] | None] = [("cmd", str(cmd)[:60])] if cmd else []
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
    final_msg = (data.get("final_message") or "").strip().replace("\n", " ")
    params = [
        ("branch", branch)
        for branch in [artifact.get("branch_final"), artifact.get("branch_planned")]
        if branch
    ]
    extras: list[tuple[str, str] | None] = [
        ("prompt", prompt_preview) if prompt_preview else None,
        ("final", final_msg[:1000]) if final_msg else None,
    ]
    try:
        if dur is not None:
            extras.append(("time", f"{float(dur):.1f}s"))
    except (TypeError, ValueError):
        pass
    try:
        if cost is not None:
            extras.append(("cost", f"${float(cost):.2f}"))
    except (TypeError, ValueError):
        pass

    console.print(
        _render_line(
            kind="completed",
            prefix=prefix(ev),
            status="completed",
            params=params,
            extras=extras,
        ),
        markup=False,
        highlight=False,
    )


def _handle_failed(console: Console, prefix: Callable[[dict], str], ev: dict) -> None:
    data = ev.get("data", {})
    etype = data.get("error_type") or "unknown"
    msg = (data.get("message") or data.get("error") or "").strip().replace("\n", " ")
    extras = [("msg", msg[:200])] if msg else []
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


def build_streaming_handlers(
    console: Console, prefix: Callable[[dict], str], prompt_preview: str, verbose: bool
) -> tuple[dict[str, Callable[[dict], None]], list[str]]:
    handlers: dict[str, Callable[[dict], None]] = {
        "task.started": lambda ev: _handle_task_state(
            "started", "started", console, prefix, prompt_preview, ev
        ),
        "task.progress": lambda ev: _handle_progress(console, prefix, ev),
        "task.completed": lambda ev: _handle_completed(
            console, prefix, prompt_preview, ev
        ),
        "task.failed": lambda ev: _handle_failed(console, prefix, ev),
        "task.scheduled": lambda ev: _handle_task_state(
            "queued", "queued", console, prefix, prompt_preview, ev
        ),
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
                "instance.agent_assistant": lambda ev: _handle_agent_assistant(
                    console, prefix, ev
                ),
                "instance.agent_tool_use": lambda ev: _handle_agent_tool_use(
                    console, prefix, ev
                ),
                "instance.agent_tool_result": lambda ev: _handle_agent_tool_result(
                    console, prefix, ev
                ),
                "instance.agent_system": lambda ev: _handle_progress(
                    console, prefix, ev
                ),
            }
        )

    return handlers, event_types
