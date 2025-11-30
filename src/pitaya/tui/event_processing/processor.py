"""Aggregate EventProcessor composed from specialized handler mixins."""

from __future__ import annotations

from .agent import AgentEventHandlers
from .base import EventProcessorBase
from .instance_lifecycle import InstanceLifecycleHandlers
from .instance_phases import InstancePhaseHandlers
from .run_strategy import RunStrategyEventHandlers
from .task_lifecycle import TaskLifecycleHandlers
from .task_progress import TaskProgressHandlers


class EventProcessor(
    TaskLifecycleHandlers,
    TaskProgressHandlers,
    InstanceLifecycleHandlers,
    InstancePhaseHandlers,
    AgentEventHandlers,
    RunStrategyEventHandlers,
    EventProcessorBase,
):
    """Processes Pitaya events and updates TUI state."""


__all__ = ["EventProcessor"]
