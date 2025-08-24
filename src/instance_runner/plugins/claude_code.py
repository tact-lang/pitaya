"""
Anthropic plugin implementation.

This plugin integrates the Anthropic coding agent with Pitaya's
runner interface.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..plugin_interface import RunnerPlugin, PluginCapabilities
from ..claude_parser import ClaudeOutputParser

if TYPE_CHECKING:
    from docker.models.containers import Container


logger = logging.getLogger(__name__)


class ClaudeCodePlugin(RunnerPlugin):
    """Plugin for running the Anthropic agent in containers."""

    def __init__(self):
        self._parser = ClaudeOutputParser()

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def docker_image(self) -> str:
        return "pitaya-agents:latest"

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            supports_resume=True,
            supports_cost=True,
            supports_streaming_events=True,
            supports_token_counts=True,
            supports_streaming=True,
            supports_cost_limits=False,  # Removed as per spec
            requires_auth=True,
            auth_methods=["oauth", "api_key"],
        )

    async def validate_environment(
        self, auth_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate plugin can run with required credentials."""
        # Check for authentication from auth_config first, then environment
        has_auth = False

        if auth_config:
            has_auth = bool(
                auth_config.get("oauth_token") or auth_config.get("api_key")
            )

        if not has_auth:
            # Fall back to environment variables
            has_oauth = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
            has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            has_auth = has_oauth or has_api_key

        if not has_auth:
            return (
                False,
                "No authentication found. Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY",
            )

        return True, None

    async def prepare_environment(
        self,
        container: "Container",
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Prepare environment variables for authentication."""
        env_vars = {}

        if auth_config:
            if auth_config.get("oauth_token"):
                env_vars["CLAUDE_CODE_OAUTH_TOKEN"] = auth_config["oauth_token"]
            if auth_config.get("api_key"):
                env_vars["ANTHROPIC_API_KEY"] = auth_config["api_key"]
            if auth_config.get("base_url"):
                env_vars["ANTHROPIC_BASE_URL"] = auth_config["base_url"]
        else:
            # Fall back to environment variables from host
            if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
                env_vars["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ[
                    "CLAUDE_CODE_OAUTH_TOKEN"
                ]
            if os.environ.get("ANTHROPIC_API_KEY"):
                env_vars["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
            if os.environ.get("ANTHROPIC_BASE_URL"):
                env_vars["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]

        return env_vars

    async def prepare_container(
        self,
        container_config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add plugin-specific container configuration (if needed)."""
        # Authentication is already handled by docker_manager.py based on auth_config
        # Session ID is also handled by docker_manager.py
        # This method is kept for future Claude-specific container needs

        # Note: /home/claude is mounted as a named volume by docker_manager.py
        # for session persistence

        return container_config

    async def build_command(
        self,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        operator_resume: bool = False,
        **kwargs,
    ) -> List[str]:
        """Build command for the Anthropic CLI."""
        command = [
            "claude",
            "--model",
            model,
            "--print",  # Non-interactive mode
            "--verbose",  # Required for stream-json
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",  # Skip permissions in sandbox
        ]

        if session_id:
            command.extend(["--resume", session_id])

        # Add prompt engineering options if provided
        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            command.extend(["--system-prompt", kwargs["system_prompt"]])

        if "append_system_prompt" in kwargs and kwargs["append_system_prompt"]:
            command.extend(["--append-system-prompt", kwargs["append_system_prompt"]])

        # Add passthrough CLI args, if any
        try:
            extra_args = kwargs.get("agent_cli_args") or []
            if isinstance(extra_args, (list, tuple)):
                command.extend([str(x) for x in extra_args if x is not None])
        except Exception:
            pass

        # Always pass the prompt when provided. On resume, Orchestrator supplies
        # a minimal continuation prompt (e.g., "Continue") so the agent keeps going.
        if prompt:
            command.append(prompt)
            try:
                if session_id and operator_resume:
                    logger.debug(
                        "claude-code: operator_resume=True; including continuation prompt with session_id=%s",
                        str(session_id)[:16],
                    )
            except Exception:
                pass

        return command

    async def parse_events(
        self,
        output_line: str,
        parser_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse agent stream-json output."""
        # Initialize parser in state if needed
        if "_parser" not in parser_state:
            parser_state["_parser"] = ClaudeOutputParser()

        parser = parser_state["_parser"]
        return parser.parse_line(output_line)

    async def extract_result(
        self,
        parser_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract final result from the parser."""
        if "_parser" not in parser_state:
            return {
                "session_id": None,
                "final_message": None,
                "metrics": {},
            }

        parser = parser_state["_parser"]
        summary = parser.get_summary()

        return {
            "session_id": summary.get("session_id"),
            "final_message": summary.get("final_message"),
            "metrics": summary.get("metrics", {}),
        }

    async def handle_error(
        self,
        error: Exception,
        parser_state: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """Determine error type and retryability."""
        error_str = str(error).lower()

        # Check for specific Claude errors
        if "rate limit" in error_str:
            return "rate_limit", True
        elif "authentication" in error_str or "unauthorized" in error_str:
            return "auth", False
        elif "timeout" in error_str:
            return "timeout", True
        elif "docker" in error_str:
            return "docker", True
        elif "session" in error_str and (
            "corrupt" in error_str or "invalid" in error_str
        ):
            return "session_corrupted", False
        else:
            # Generic agent error
            return "agent", True

    async def execute(
        self,
        docker_manager: Any,
        container: "Container",
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        event_callback: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the agent inside the container and return result data."""
        # Build command
        # Extract operator_resume flag (avoid passing duplicate kwarg)
        op_resume = bool(kwargs.pop("operator_resume", False))
        command = await self.build_command(
            prompt=prompt,
            model=model,
            session_id=session_id,
            operator_resume=op_resume,
            **kwargs,
        )

        # Execute and parse via docker manager
        # Optional raw stream log path (tee everything to file)
        stream_log_path = kwargs.get("stream_log_path")

        result_data = await docker_manager.execute_command(
            container=container,
            command=command,
            plugin=self,
            event_callback=(
                (lambda ev: event_callback(ev)) if callable(event_callback) else None
            ),
            timeout_seconds=timeout_seconds,
            max_turns=kwargs.get("max_turns"),
            stream_log_path=stream_log_path,
        )

        return result_data
