"""Plugin capability and interface definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .type_aliases import (
    AuthParams,
    Command,
    ContainerConfig,
    EnvironmentVars,
    EventCallback,
    EventData,
    ParserState,
)


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

    def __post_init__(self) -> None:
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
        self, auth_config: Optional[AuthParams] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that the plugin can run in the current environment.

        Args:
            auth_config: Authentication configuration (OAuth token or API key)

        Returns:
            Tuple of (is_valid, error_message)
        """

    @abstractmethod
    async def prepare_environment(
        self,
        container: Any,  # Avoid importing docker types
        auth_config: Optional[AuthParams] = None,
    ) -> EnvironmentVars:
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

    @abstractmethod
    async def prepare_container(
        self,
        container_config: ContainerConfig,
        session_id: Optional[str] = None,
    ) -> ContainerConfig:
        """
        Prepare container configuration specific to this plugin.

        Args:
            container_config: Base container configuration
            session_id: Session ID for resuming work

        Returns:
            Modified container configuration
        """

    @abstractmethod
    async def build_command(
        self,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Command:
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

    @abstractmethod
    async def parse_events(
        self,
        output_line: str,
        parser_state: ParserState,
    ) -> Optional[EventData]:
        """
        Parse a single line of output from the AI tool.

        Args:
            output_line: Single line of output
            parser_state: Mutable parser state for tracking across lines

        Returns:
            Parsed event or None if not a valid event
        """

    @abstractmethod
    async def extract_result(
        self,
        parser_state: ParserState,
    ) -> EventData:
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

    @abstractmethod
    async def execute(
        self,
        docker_manager: Any,
        container: Any,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        event_callback: Optional[EventCallback] = None,
        **kwargs,
    ) -> EventData:
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

    @abstractmethod
    async def handle_error(
        self,
        error: Exception,
        parser_state: ParserState,
    ) -> Tuple[str, bool]:
        """
        Handle plugin-specific errors.

        Args:
            error: The exception that occurred
            parser_state: Current parser state

        Returns:
            Tuple of (error_type, is_retryable)
        """


__all__ = ["PluginCapabilities", "RunnerPlugin"]
