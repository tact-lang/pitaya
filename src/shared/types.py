"""
Shared types and data structures used across layers.

These types are extracted from individual layers to prevent cross-layer
imports while maintaining type safety and consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable


# From instance_runner/types.py


@dataclass
class ContainerLimits:
    """Resource limits for Docker containers."""

    cpu_count: int = 2
    memory_gb: int = 4
    memory_swap_gb: int = 4  # Total memory + swap


@dataclass
class AuthConfig:
    """Authentication configuration for AI tools."""

    oauth_token: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_attempts: int = 3
    initial_delay_seconds: float = 10.0  # Per spec: 10s, 60s, 360s
    max_delay_seconds: float = 360.0
    exponential_base: float = 6.0  # To get 10s -> 60s -> 360s progression
    # Pattern-based retry logic as per spec
    docker_error_patterns: tuple = (
        "connection refused",
        "no such host",
        "timeout",
        "daemon",  # broader match per spec
        "Cannot connect to the Docker daemon",
        "already in use",  # name conflict
        "409 Client Error",  # explicit status hint
        'Conflict ("Conflict.',  # docker's overlap wording
    )
    agent_error_patterns: tuple = (
        "rate limit",
        "API error",
        "connection reset",
        "overloaded_error",
        "429",  # Rate limit status code
    )
    general_error_patterns: tuple = (
        "ECONNREFUSED",
        "ETIMEDOUT",
        "ENETUNREACH",
        "Connection timed out",
    )


@dataclass
class InstanceResult:
    """Result from running a single instance."""

    success: bool
    branch_name: Optional[str] = None
    has_changes: bool = False
    final_message: Optional[str] = None
    session_id: Optional[str] = None
    container_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    # Additional fields from specification
    commit_statistics: Optional[Dict[str, Any]] = None  # count, lines added/deleted
    started_at: Optional[str] = None  # ISO timestamp
    completed_at: Optional[str] = None  # ISO timestamp
    retry_attempts: int = 0
    log_path: Optional[str] = None
    workspace_path: Optional[str] = None  # Until cleanup
    status: str = "unknown"  # success/failed/timeout/canceled
    # Artifact extras for orchestration mapping
    commit: Optional[str] = None
    duplicate_of_branch: Optional[str] = None
    dedupe_reason: Optional[str] = None
    # Strategy-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """Convenience: total cost from metrics (spec)."""
        # Metrics use 'total_cost' throughout the codebase
        return float(self.metrics.get("total_cost", self.metrics.get("cost", 0.0)))

    @property
    def tokens(self) -> int:
        """Convenience: total tokens from metrics (spec)."""
        return int(self.metrics.get("total_tokens", 0))

    @property
    def token_breakdown(self) -> Dict[str, int]:
        """Token breakdown dict: input/output/total."""
        return {
            "input": int(self.metrics.get("input_tokens", 0)),
            "output": int(self.metrics.get("output_tokens", 0)),
            "total": int(self.metrics.get("total_tokens", 0)),
        }


# From instance_runner/plugin_interface.py


@dataclass
class PluginCapabilities:
    """Capabilities supported by the plugin."""

    supports_resume: bool = False
    supports_cost: bool = False
    supports_streaming_events: bool = True
    supports_token_counts: bool = True
    supports_streaming: bool = True
    supports_cost_limits: bool = False
    requires_auth: bool = True
    auth_methods: Optional[List[str]] = None  # e.g., ["oauth", "api_key"]

    def __post_init__(self):
        if self.auth_methods is None:
            self.auth_methods = []


class RunnerPlugin(ABC):
    """
    Abstract interface for AI coding tool runners.

    Each plugin must implement these methods to integrate with Pitaya.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name (e.g., 'claude-code', 'github-copilot-workspace')."""
        pass

    @property
    @abstractmethod
    def docker_image(self) -> str:
        """Default Docker image for this plugin."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> PluginCapabilities:
        """Capabilities supported by this plugin."""
        pass

    @abstractmethod
    async def validate_environment(
        self, auth_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that the plugin can run in the current environment.

        Args:
            auth_config: Authentication configuration (OAuth token or API key)

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    async def prepare_environment(
        self,
        container: Any,  # Avoid importing docker types
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Prepare the container environment for this plugin.

        This method is called after container creation but before command execution.
        Use it to set up authentication, configuration files, etc.

        Args:
            container: The running Docker container
            auth_config: Authentication configuration (oauth_token, api_key, base_url)

        Returns:
            Environment variables to set in the container
        """
        pass

    @abstractmethod
    async def prepare_container(
        self,
        container_config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare container configuration specific to this plugin.

        Args:
            container_config: Base container configuration
            session_id: Session ID for resuming work

        Returns:
            Modified container configuration
        """
        pass

    @abstractmethod
    async def build_command(
        self,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Build the command to execute in the container.

        Args:
            prompt: The instruction for the AI tool
            model: Model to use (plugin-specific)
            session_id: Session ID for resuming
            **kwargs: Additional plugin-specific arguments

        Returns:
            Command array to execute
        """
        pass

    @abstractmethod
    async def parse_events(
        self,
        output_line: str,
        parser_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single line of output from the AI tool.

        Args:
            output_line: Single line of output
            parser_state: Mutable parser state for tracking across lines

        Returns:
            Parsed event or None if not a valid event
        """
        pass

    @abstractmethod
    async def extract_result(
        self,
        parser_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract the final result from parser state.

        Args:
            parser_state: Final parser state after all output processed

        Returns:
            Result dictionary with at least:
            - session_id: Optional[str]
            - final_message: Optional[str]
            - metrics: Dict[str, Any]
        """
        pass

    @abstractmethod
    async def execute(
        self,
        docker_manager: Any,
        container: Any,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the tool inside the given container and return result data.

        Args:
            docker_manager: Docker manager helper
            container: Running container to execute in
            prompt: Instruction for the tool
            model: Model selection
            session_id: Optional resume session id
            timeout_seconds: Max execution time
            event_callback: Callback for parsed events
            **kwargs: Additional plugin-specific args

        Returns:
            Result dict with session_id, final_message, metrics
        """
        pass

    @abstractmethod
    async def handle_error(
        self,
        error: Exception,
        parser_state: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        Handle plugin-specific errors.

        Args:
            error: The exception that occurred
            parser_state: Current parser state

        Returns:
            Tuple of (error_type, is_retryable)
        """
        pass


# From orchestration/state.py


class InstanceStatus(Enum):
    """Possible states for an instance."""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


# Event type definitions


@dataclass
class Event:
    """Base event structure for all Pitaya events."""

    type: str  # e.g., "instance.started", "strategy.completed"
    timestamp: str  # ISO format timestamp with timezone
    data: Dict[str, Any]  # Event-specific data
    instance_id: Optional[str] = None  # For instance-specific events

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        if self.instance_id:
            result["instance_id"] = self.instance_id
        return result


# Event type constants for type safety
class EventTypes:
    """Standard event types emitted by the Pitaya system."""

    # Instance Runner events
    INSTANCE_QUEUED = "instance.queued"
    INSTANCE_STARTED = "instance.started"
    INSTANCE_PROGRESS = "instance.progress"
    INSTANCE_COMPLETED = "instance.completed"
    INSTANCE_FAILED = "instance.failed"
    INSTANCE_PHASE_COMPLETED = "instance.phase_completed"
    INSTANCE_WORKSPACE_PREPARING = "instance.workspace_preparing"
    INSTANCE_WORKSPACE_PREPARED = "instance.workspace_prepared"
    INSTANCE_WORKSPACE_CLEANED = "instance.workspace_cleaned"
    INSTANCE_CONTAINER_CREATED = "instance.container_created"
    INSTANCE_CONTAINER_STARTED = "instance.container_started"
    INSTANCE_CONTAINER_STOPPED = "instance.container_stopped"
    INSTANCE_NO_CHANGES = "instance.no_changes"
    INSTANCE_RESULT_COLLECTION_STARTED = "instance.result_collection_started"
    # Generic assistant event names
    INSTANCE_AGENT_TOOL_USE = "instance.agent_tool_use"
    INSTANCE_AGENT_TOOL_RESULT = "instance.agent_tool_result"
    INSTANCE_AGENT_ASSISTANT = "instance.agent_assistant"
    INSTANCE_AGENT_RESULT = "instance.agent_result"
    INSTANCE_AGENT_COMPLETED = "instance.agent_completed"

    # Strategy events
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_INSTANCE_SPAWNED = "strategy.instance_spawned"
    STRATEGY_COMPLETED = "strategy.completed"
    STRATEGY_FAILED = "strategy.failed"

    # State management events
    STATE_RUN_INITIALIZED = "state.run_initialized"
    STATE_STRATEGY_REGISTERED = "state.strategy_registered"
    STATE_INSTANCE_REGISTERED = "state.instance_registered"
    STATE_INSTANCE_UPDATED = "state.instance_updated"
    STATE_SNAPSHOT_SAVED = "state.snapshot_saved"
    STATE_RESTORED = "state.restored"

    # Run lifecycle events
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELED = "run.canceled"
