"""
Instance Runner - Executes AI coding instances in isolated Docker containers.
"""

__version__ = "0.1.0"

# Import exceptions from common module
from ..exceptions import (
    OrchestratorError,
    DockerError,
    GitError,
    ClaudeError,
    TimeoutError,
    ValidationError,
)

# For backward compatibility, alias OrchestratorError as InstanceRunnerError
InstanceRunnerError = OrchestratorError

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
    "InstanceRunnerError",
    "DockerError",
    "GitError",
    "ClaudeError",
    "TimeoutError",
    "ValidationError",
]
