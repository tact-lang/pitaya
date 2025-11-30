"""Pitaya exception hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

__all__ = [
    "AggregateTaskFailed",
    "AgentError",
    "DockerError",
    "GitError",
    "KeyConflictDifferentFingerprint",
    "NoViableCandidates",
    "OrchestratorError",
    "StrategyError",
    "TaskFailed",
    "TimeoutError",
    "ValidationError",
]


class OrchestratorError(Exception):
    """Base class for Pitaya exceptions."""


class DockerError(OrchestratorError):
    """Raised when Docker operations fail."""


class GitError(OrchestratorError):
    """Raised when Git operations fail."""


class StrategyError(OrchestratorError):
    """Raised when strategy execution fails."""


class ValidationError(OrchestratorError):
    """Raised when input validation fails."""


class TimeoutError(OrchestratorError):
    """Raised when operations exceed time limits."""


class AgentError(OrchestratorError):
    """Raised when an agent tool operation fails."""


class KeyConflictDifferentFingerprint(OrchestratorError):
    """Raised when a durable key is reused with a different fingerprint."""


@dataclass(slots=True, frozen=True)
class TaskFailure:
    """Container for task failure metadata."""

    key: str
    error_type: str
    message: str
    result: Optional[Any]


class TaskFailed(OrchestratorError):
    """Raised when a single task fails while awaiting its result."""

    def __init__(
        self,
        key: str,
        error_type: str = "unknown",
        message: str = "",
        *,
        result: Optional[Any] = None,
    ) -> None:
        super().__init__(f"TaskFailed(key={key}, type={error_type}): {message}")
        self.failure = TaskFailure(
            key=key, error_type=error_type, message=message, result=result
        )


class AggregateTaskFailed(OrchestratorError):
    """Raised when multiple tasks fail and tolerance is disabled."""

    def __init__(self, keys: Iterable[str]) -> None:
        keys_list = list(keys)
        super().__init__(f"AggregateTaskFailed(keys={keys_list})")
        self.keys: List[str] = keys_list


class NoViableCandidates(OrchestratorError):
    """Raised when a strategy yields no valid candidates."""
