"""Instance status enumeration shared across layers."""

from enum import Enum


class InstanceStatus(Enum):
    """Possible states for an instance."""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


__all__ = ["InstanceStatus"]
