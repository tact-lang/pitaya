"""
Pitaya TUI - Interactive display for AI coding agent orchestration.

Provides real-time monitoring of parallel AI coding agent execution with
adaptive display modes based on instance count.
"""

__version__ = "0.2.0"

from .display import TUIDisplay
from .models import (
    TUIState,
    RunDisplay,
    StrategyDisplay,
    InstanceDisplay,
    InstanceStatus,
)
from .event_handler import EventProcessor, AsyncEventStream
from .adaptive import AdaptiveDisplay
from .cli import OrchestratorTUI, main

__all__ = [
    "TUIDisplay",
    "TUIState",
    "RunDisplay",
    "StrategyDisplay",
    "InstanceDisplay",
    "InstanceStatus",
    "EventProcessor",
    "AsyncEventStream",
    "AdaptiveDisplay",
    "OrchestratorTUI",
    "main",
]
