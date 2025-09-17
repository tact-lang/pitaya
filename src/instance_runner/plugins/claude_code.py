"""Anthropic Claude Code plugin implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..claude_parser import ClaudeOutputParser
from ..plugin_interface import PluginCapabilities, RunnerPlugin

if TYPE_CHECKING:
    from docker.models.containers import Container

EventCallback = Callable[[Dict[str, Any]], None]
ParserState = Dict[str, Any]

logger = logging.getLogger(__name__)

__all__ = ["ClaudeCodePlugin"]

_ENV_OAUTH = "CLAUDE_CODE_OAUTH_TOKEN"
_ENV_API_KEY = "ANTHROPIC_API_KEY"
_ENV_BASE_URL = "ANTHROPIC_BASE_URL"
_DEFAULT_IMAGE = "pitaya-agents:latest"
_CLI_STREAM_FORMAT = "stream-json"


class ClaudeCodePlugin(RunnerPlugin):
    """Runner plugin that wraps the Anthropic Claude Code CLI."""

    def __init__(self) -> None:
        self._parser = ClaudeOutputParser()

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def docker_image(self) -> str:
        return _DEFAULT_IMAGE

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
        """Ensure required authentication is available."""
        if auth_config and (
            auth_config.get("oauth_token") or auth_config.get("api_key")
        ):
            return True, None

        if os.environ.get(_ENV_OAUTH) or os.environ.get(_ENV_API_KEY):
            return True, None

        return (
            False,
            "No authentication found. Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY",
        )

    async def prepare_environment(
        self,
        container: "Container",
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Prepare environment variables for authentication."""
        return _collect_auth_env(auth_config)

    async def prepare_container(
        self,
        container_config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add plugin-specific container configuration (if needed)."""
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
        command: List[str] = [
            "claude",
            "--model",
            model,
            "--print",
            "--verbose",
            "--output-format",
            _CLI_STREAM_FORMAT,
            "--dangerously-skip-permissions",
        ]

        if session_id:
            command.extend(["--resume", session_id])

        # Add prompt engineering options if provided
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            command.extend(["--system-prompt", str(system_prompt)])

        append_system_prompt = kwargs.get("append_system_prompt")
        if append_system_prompt:
            command.extend(["--append-system-prompt", str(append_system_prompt)])

        extra_args = kwargs.get("agent_cli_args")
        if isinstance(extra_args, (list, tuple)):
            command.extend([str(arg) for arg in extra_args if arg is not None])

        # Always pass the prompt when provided. On resume, Orchestrator supplies
        # a minimal continuation prompt (e.g., "Continue") so the agent keeps going.
        if prompt:
            command.append(prompt)
            if session_id and operator_resume:
                logger.debug(
                    "claude-code: operator_resume=True; including continuation prompt (session_id=%s)",
                    str(session_id)[:16],
                )

        return command

    async def parse_events(
        self,
        output_line: str,
        parser_state: ParserState,
    ) -> Optional[Dict[str, Any]]:
        """Parse agent stream-json output."""
        # Initialize parser in state if needed
        if "_parser" not in parser_state:
            parser_state["_parser"] = ClaudeOutputParser()

        parser: ClaudeOutputParser = parser_state["_parser"]
        return parser.parse_line(output_line)

    async def extract_result(
        self,
        parser_state: ParserState,
    ) -> Dict[str, Any]:
        """Extract final result from the parser."""
        if "_parser" not in parser_state:
            return {
                "session_id": None,
                "final_message": None,
                "metrics": {},
            }

        parser: ClaudeOutputParser = parser_state["_parser"]
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

        if "rate limit" in error_str:
            return "rate_limit", True
        if "authentication" in error_str or "unauthorized" in error_str:
            return "auth", False
        if "timeout" in error_str:
            return "timeout", True
        if "docker" in error_str:
            return "docker", True
        if "session" in error_str and (
            "corrupt" in error_str or "invalid" in error_str
        ):
            return "session_corrupted", False
        return "agent", True

    async def execute(
        self,
        docker_manager: Any,
        container: "Container",
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        event_callback: Optional[EventCallback] = None,
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

        callback = event_callback if callable(event_callback) else None
        result_data = await docker_manager.execute_command(
            container=container,
            command=command,
            plugin=self,
            event_callback=callback,
            timeout_seconds=timeout_seconds,
            max_turns=kwargs.get("max_turns"),
            stream_log_path=stream_log_path,
        )

        return result_data


def _collect_auth_env(auth_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env: Dict[str, str] = {}

    if auth_config:
        if auth_config.get("oauth_token"):
            env[_ENV_OAUTH] = str(auth_config["oauth_token"])
        if auth_config.get("api_key"):
            env[_ENV_API_KEY] = str(auth_config["api_key"])
        if auth_config.get("base_url"):
            env[_ENV_BASE_URL] = str(auth_config["base_url"])
        return env

    if _ENV_OAUTH in os.environ:
        env[_ENV_OAUTH] = os.environ[_ENV_OAUTH]
    if _ENV_API_KEY in os.environ:
        env[_ENV_API_KEY] = os.environ[_ENV_API_KEY]
    if _ENV_BASE_URL in os.environ:
        env[_ENV_BASE_URL] = os.environ[_ENV_BASE_URL]
    return env
