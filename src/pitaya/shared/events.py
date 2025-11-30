"""Event data structures and constants shared across the system."""

from dataclasses import dataclass
from typing import Optional

from .type_aliases import EventData


@dataclass
class Event:
    """Base event structure for all Pitaya events."""

    type: str  # e.g., "instance.started", "strategy.completed"
    timestamp: str  # ISO format timestamp with timezone
    data: EventData  # Event-specific data
    instance_id: Optional[str] = None  # For instance-specific events

    def to_dict(self) -> EventData:
        """Convert event to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        if self.instance_id:
            result["instance_id"] = self.instance_id
        return result


class EventTypes:
    """Standard event types emitted by the Pitaya system."""

    # Instance Runner events
    INSTANCE_QUEUED = "instance.queued"
    INSTANCE_STARTED = "instance.started"
    INSTANCE_PROGRESS = "instance.progress"
    INSTANCE_COMPLETED = "instance.completed"
    INSTANCE_FAILED = "instance.failed"
    INSTANCE_PHASE_COMPLETED = "instance.phase_completed"
    INSTANCE_WORKSPACE_PREPARING = "instance.workspace_preparing"
    INSTANCE_WORKSPACE_PREPARED = "instance.workspace_prepared"
    INSTANCE_WORKSPACE_CLEANED = "instance.workspace_cleaned"
    INSTANCE_CONTAINER_CREATED = "instance.container_created"
    INSTANCE_CONTAINER_STARTED = "instance.container_started"
    INSTANCE_CONTAINER_STOPPED = "instance.container_stopped"
    INSTANCE_NO_CHANGES = "instance.no_changes"
    INSTANCE_RESULT_COLLECTION_STARTED = "instance.result_collection_started"
    # Generic assistant event names
    INSTANCE_AGENT_TOOL_USE = "instance.agent_tool_use"
    INSTANCE_AGENT_TOOL_RESULT = "instance.agent_tool_result"
    INSTANCE_AGENT_ASSISTANT = "instance.agent_assistant"
    INSTANCE_AGENT_RESULT = "instance.agent_result"
    INSTANCE_AGENT_COMPLETED = "instance.agent_completed"

    # Strategy events
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_INSTANCE_SPAWNED = "strategy.instance_spawned"
    STRATEGY_COMPLETED = "strategy.completed"
    STRATEGY_FAILED = "strategy.failed"

    # State management events
    STATE_RUN_INITIALIZED = "state.run_initialized"
    STATE_STRATEGY_REGISTERED = "state.strategy_registered"
    STATE_INSTANCE_REGISTERED = "state.instance_registered"
    STATE_INSTANCE_UPDATED = "state.instance_updated"
    STATE_SNAPSHOT_SAVED = "state.snapshot_saved"
    STATE_RESTORED = "state.restored"

    # Run lifecycle events
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELED = "run.canceled"


__all__ = ["Event", "EventTypes"]
