"""
Shared types and interfaces used across layers.

This module contains data structures that need to be shared between layers
without creating cross-layer dependencies. By placing these in a separate
module, we maintain the architectural boundary that layers should not
import from each other directly.
"""

from pitaya.shared.results import InstanceResult
from pitaya.shared.status import InstanceStatus
from pitaya.shared.plugin import RunnerPlugin, PluginCapabilities
from pitaya.config.models import AuthConfig, ContainerLimits, RetryConfig

__all__ = [
    "InstanceResult",
    "InstanceStatus",
    "ContainerLimits",
    "AuthConfig",
    "RetryConfig",
    "RunnerPlugin",
    "PluginCapabilities",
]
