"""
Instance Runner — executes AI coding agent instances in isolated Docker containers.

Used by Pitaya to orchestrate tools such as Claude Code and Codex CLI.
"""

__version__ = "0.2.0"

# Import exceptions from common module
from ..exceptions import (
    OrchestratorError,
    DockerError,
    GitError,
    AgentError,
    TimeoutError,
    ValidationError,
)

# Import from public API
from .api import run_instance

# Import shared types instead of local ones
from ..shared import (
    InstanceResult,
    ContainerLimits,
    AuthConfig,
    RetryConfig,
    RunnerPlugin,
    PluginCapabilities,
)

__all__ = [
    # Main function
    "run_instance",
    # Data classes
    "InstanceResult",
    "ContainerLimits",
    "AuthConfig",
    "RetryConfig",
    # Plugin system
    "RunnerPlugin",
    "PluginCapabilities",
    # Exceptions
    "OrchestratorError",
    "DockerError",
    "GitError",
    "AgentError",
    "TimeoutError",
    "ValidationError",
]
