"""Aggregated shared types and interfaces."""

from pitaya.config.models import AuthConfig, ContainerLimits, RetryConfig
from pitaya.shared.events import Event, EventTypes
from pitaya.shared.plugin import PluginCapabilities, RunnerPlugin
from pitaya.shared.results import InstanceResult
from pitaya.shared.status import InstanceStatus
from pitaya.shared.type_aliases import (
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
