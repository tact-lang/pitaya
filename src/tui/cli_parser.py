"""Argument parser construction for pitaya-tui."""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import __version__


def create_tui_parser() -> argparse.ArgumentParser:
    """Create argument parser for TUI with clear grouping and examples."""
    parser = argparse.ArgumentParser(
        prog="pitaya-tui",
        description="Monitor Pitaya runs from an events file (TUI or text)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Watch a run by ID\n"
            "  pitaya-tui --run-id run_20250114_123456\n\n"
            "  # Stream as text from a file\n"
            "  pitaya-tui --events-file logs/run_20250114_123456/events.jsonl --output streaming\n\n"
            "  # Emit JSON events (NDJSON, no TUI)\n"
            "  pitaya-tui --events-file events.jsonl --output json\n"
        ),
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    g_input = parser.add_argument_group("Input")
    g_input.add_argument(
        "--events-file", type=Path, help="Read events from file (offline mode)"
    )
    g_input.add_argument(
        "--from-offset", type=int, default=0, help="Start reading from byte offset"
    )
    g_input.add_argument("--run-id", help="Infer events file by run ID")

    g_display = parser.add_argument_group("Display")
    g_display.add_argument(
        "--output",
        choices=["tui", "streaming", "json", "quiet"],
        default="tui",
        help="Output mode (default: tui). JSON emits NDJSON only.",
    )
    g_display.add_argument(
        "--display-mode",
        choices=["auto", "detailed", "compact", "dense"],
        default="auto",
        help="Force display density for TUI",
    )
    g_display.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    g_filters = parser.add_argument_group("Filters")
    g_filters.add_argument("--instance-id", help="Filter by instance ID")
    g_filters.add_argument("--event-types", nargs="+", help="Filter by event type(s)")

    return parser


__all__ = ["create_tui_parser"]
