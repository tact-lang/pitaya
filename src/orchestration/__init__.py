"""
Orchestration component for coordinating multiple AI coding agents.

Pitaya orchestrates agents such as Claude Code and Codex CLI with pluggable
and custom strategies. This component manages parallel execution, events, and
state, providing the intelligence for complex multi-instance workflows.
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

__version__ = "0.2.0"

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
