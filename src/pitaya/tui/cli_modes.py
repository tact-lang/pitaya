"""Output mode helpers for the TUI CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

from .event_handler import AsyncEventStream, EventProcessor


async def stream_events_file(
    console: Console,
    event_processor: EventProcessor,
    events_file: Path,
    args,
) -> None:
    """Stream events as human-readable text."""
    event_stream = AsyncEventStream(event_processor)

    def stream_callback(event: Dict[str, Any]):
        if args.run_id and event.get("run_id") != args.run_id:
            return
        if args.instance_id and event.get("instance_id") != args.instance_id:
            return
        if args.event_types and event.get("type") not in args.event_types:
            return

        timestamp = event.get("ts") or event.get("timestamp", "")
        event_type = event.get("type", "unknown")
        instance_id = event.get("instance_id", "")

        if "failed" in event_type:
            style = "red"
        elif "completed" in event_type:
            style = "green"
        elif "started" in event_type:
            style = "yellow"
        else:
            style = "white"

        msg = f"[{timestamp}]"
        if instance_id:
            msg += f" [{instance_id[:8]}]"
        msg += f" {event_type}"

        data = event.get("data", {})
        if "error" in data:
            msg += f" - Error: {data['error']}"
        elif "activity" in data:
            msg += f" - {data['activity']}"
        elif "tool" in data:
            msg += f" - Tool: {data['tool']}"
        elif "strategy" in data:
            msg += f" - Strategy: {data['strategy']}"

        console.print(msg, style=style)

    original_process = event_processor.process_event
    event_processor.process_event = lambda e: (original_process(e), stream_callback(e))

    await event_stream.start(events_file, args.from_offset)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await event_stream.stop()


async def json_events_file(events_file: Path, args) -> None:
    """Output events as JSON."""
    with open(events_file, "rb") as f:
        if args.from_offset > 0:
            off = args.from_offset
            if off > 0:
                f.seek(off - 1)
                if f.read(1) != b"\n":
                    while True:
                        b = f.read(1)
                        if not b or b == b"\n":
                            break
            f.seek(max(0, args.from_offset))

        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="ignore").strip()
            if not s:
                continue

            try:
                event = json.loads(s)

                if args.run_id and event.get("run_id") != args.run_id:
                    continue
                if args.instance_id and event.get("instance_id") != args.instance_id:
                    continue
                if args.event_types and event.get("type") not in args.event_types:
                    continue

                print(json.dumps(event))

            except json.JSONDecodeError:
                continue


async def quiet_mode_file(
    console: Console, event_processor: EventProcessor, events_file: Path, args
) -> None:
    """Minimal output mode - just summary at end."""
    event_stream = AsyncEventStream(event_processor)
    await event_stream._read_existing_events(events_file, args.from_offset)

    state = event_processor.state
    if state.current_run:
        run = state.current_run
        console.print(f"Run: {run.run_id}")
        console.print(f"Total instances: {run.total_instances}")
        console.print(f"Completed: {run.completed_instances}")
        console.print(f"Failed: {run.failed_instances}")
        console.print(f"Success rate: {run.success_rate:.1f}%")
        console.print(f"Total cost: ${run.total_cost:.4f}")
        console.print(f"Total tokens: {run.total_tokens:,}")
        console.print(f"Duration: {format_duration(run.duration_seconds)}")
    else:
        console.print("No run data found")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


__all__ = [
    "stream_events_file",
    "json_events_file",
    "quiet_mode_file",
    "format_duration",
]
