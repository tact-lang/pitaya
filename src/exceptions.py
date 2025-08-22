"""
Common exception hierarchy for the Pitaya system.

These exceptions are used across all layers (instance_runner, orchestration, tui)
to maintain consistent error handling as specified in the architecture.
"""


class OrchestratorError(Exception):
    """Base exception for all Pitaya errors."""

    pass


class DockerError(OrchestratorError):
    """Raised when Docker operations fail."""

    pass


class GitError(OrchestratorError):
    """Raised when Git operations fail."""

    pass


class StrategyError(OrchestratorError):
    """Raised when strategy execution fails."""

    pass


class ValidationError(OrchestratorError):
    """Raised when input validation fails."""

    pass


class TimeoutError(OrchestratorError):
    """Raised when operations exceed time limits."""

    pass


class AgentError(OrchestratorError):
    """Raised when agent tool operations fail."""

    pass


class KeyConflictDifferentFingerprint(OrchestratorError):
    """Raised when a durable key is reused with a different fingerprint."""

    pass


class TaskFailed(OrchestratorError):
    """Raised by ctx.wait(handle) when a single task fails."""

    def __init__(
        self, key: str, error_type: str = "unknown", message: str = ""
    ) -> None:
        super().__init__(f"TaskFailed(key={key}, type={error_type}): {message}")
        self.key = key
        self.error_type = error_type
        self.message = message


class AggregateTaskFailed(OrchestratorError):
    """Raised by ctx.wait_all(handles) when one or more tasks fail (and tolerance is False)."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__(f"AggregateTaskFailed(keys={keys})")
        self.keys = keys


class NoViableCandidates(OrchestratorError):
    """Raised by strategies when selection yields no valid candidates."""

    pass
