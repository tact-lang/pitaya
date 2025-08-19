"""
Orchestration component for coordinating multiple AI coding instances.

This component manages strategies, parallel execution, events, and state.
It provides the intelligence for running complex multi-instance workflows.
"""

from .orchestrator import Orchestrator
from .event_bus import EventBus
from .state import RunState, StateManager
from .strategies import (
    Strategy,
    StrategyConfig,
    AVAILABLE_STRATEGIES,
)
from ..exceptions import (
    OrchestratorError,
    DockerError,
    GitError,
    StrategyError,
    ValidationError,
    TimeoutError,
    AgentError,
)

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "EventBus",
    "RunState",
    "StateManager",
    "Strategy",
    "StrategyConfig",
    "AVAILABLE_STRATEGIES",
    # Exceptions
    "OrchestratorError",
    "DockerError",
    "GitError",
    "StrategyError",
    "ValidationError",
    "TimeoutError",
    "AgentError",
]
