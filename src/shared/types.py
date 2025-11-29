"""Aggregated shared types and interfaces.

This module re-exports shared dataclasses, enums, and type aliases to keep
callers stable while allowing the underlying implementations to live in
smaller, focused modules.
"""

from .config import AuthConfig, ContainerLimits, RetryConfig
from .events import Event, EventTypes
from .plugin import PluginCapabilities, RunnerPlugin
from .results import InstanceResult
from .status import InstanceStatus
from .type_aliases import (
    AuthParams,
    Command,
    ContainerConfig,
    EnvironmentVars,
    ErrorPatterns,
    EventCallback,
    EventData,
    Metrics,
    ParserState,
)

__all__ = [
    "AuthConfig",
    "AuthParams",
    "Command",
    "ContainerConfig",
    "ContainerLimits",
    "EnvironmentVars",
    "ErrorPatterns",
    "Event",
    "EventCallback",
    "EventData",
    "EventTypes",
    "InstanceResult",
    "InstanceStatus",
    "Metrics",
    "ParserState",
    "PluginCapabilities",
    "RetryConfig",
    "RunnerPlugin",
]
