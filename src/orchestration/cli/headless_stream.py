"""Event subscriptions for headless streaming and JSON output."""

from __future__ import annotations

import argparse
import json

from rich.console import Console

from ...orchestration import Orchestrator
from .headless_handlers import build_streaming_handlers, make_prefix

__all__ = ["subscribe_streaming", "subscribe_json"]


def subscribe_streaming(
    console: Console, orch: Orchestrator, args: argparse.Namespace
) -> None:
    """Subscribe orchestrator events for human-friendly streaming output."""

    prefix = make_prefix(getattr(args, "show_ids", "short") == "full")
    verbose = bool(getattr(args, "verbose", False))
    prompt_preview = (args.prompt or "").strip().replace("\n", " ")[:1000]

    handlers, event_types = build_streaming_handlers(
        console, prefix, prompt_preview, verbose
    )
    for event_type in event_types:
        orch.subscribe(event_type, handlers.get(event_type, lambda _e: None))


def subscribe_json(orch: Orchestrator) -> None:
    """Subscribe orchestrator events and emit NDJSON to stdout."""

    def emit(ev: dict) -> None:
        print(json.dumps(ev, separators=(",", ":")))

    for event_type in (
        "task.scheduled",
        "task.started",
        "task.progress",
        "task.completed",
        "task.failed",
        "task.interrupted",
        "strategy.started",
        "strategy.completed",
    ):
        orch.subscribe(event_type, emit)
