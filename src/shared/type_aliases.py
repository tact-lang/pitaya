"""Common type aliases shared across orchestrator layers."""

from typing import Any, Callable, Dict, List, Tuple

Metrics = Dict[str, Any]
ParserState = Dict[str, Any]
Command = List[str]
ContainerConfig = Dict[str, Any]
EnvironmentVars = Dict[str, str]
AuthParams = Dict[str, Any]
EventData = Dict[str, Any]
ErrorPatterns = Tuple[str, ...]

EventCallback = Callable[[EventData], None]

__all__ = [
    "AuthParams",
    "Command",
    "ContainerConfig",
    "EnvironmentVars",
    "ErrorPatterns",
    "EventCallback",
    "EventData",
    "Metrics",
    "ParserState",
]
