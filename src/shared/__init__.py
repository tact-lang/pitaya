"""
Shared types and interfaces used across layers.

This module contains data structures that need to be shared between layers
without creating cross-layer dependencies. By placing these in a separate
module, we maintain the architectural boundary that layers should not
import from each other directly.
"""

from .types import (
    InstanceResult,
    InstanceStatus,
    ContainerLimits,
    AuthConfig,
    RetryConfig,
    RunnerPlugin,
    PluginCapabilities,
)

__all__ = [
    "InstanceResult",
    "InstanceStatus",
    "ContainerLimits",
    "AuthConfig",
    "RetryConfig",
    "RunnerPlugin",
    "PluginCapabilities",
]
