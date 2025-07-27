"""
Common exception hierarchy for the orchestrator system.

These exceptions are used across all layers (instance_runner, orchestration, tui)
to maintain consistent error handling as specified in the architecture.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""

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


class ClaudeError(OrchestratorError):
    """Raised when Claude Code operations fail."""

    pass
