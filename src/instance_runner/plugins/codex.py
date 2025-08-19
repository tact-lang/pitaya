"""
Codex CLI plugin implementation.

Integrates Codex (open-source Codex CLI) with the orchestrator RunnerPlugin
interface, mirroring the generic agent behavior for event streaming and metrics.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Callable

from ..plugin_interface import RunnerPlugin, PluginCapabilities
from ..codex_parser import CodexOutputParser

if TYPE_CHECKING:
    from docker.models.containers import Container


logger = logging.getLogger(__name__)


class CodexPlugin(RunnerPlugin):
    """Plugin for running Codex CLI in containers."""

    def __init__(self) -> None:
        self._parser = CodexOutputParser()

    @property
    def name(self) -> str:  # pragma: no cover - simple property
        return "codex"

    @property
    def docker_image(self) -> str:  # pragma: no cover - simple property
        return "codex-cli:latest"

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            supports_resume=True,  # via rollout/reattach when available
            supports_cost=False,
            supports_streaming_events=True,
            supports_token_counts=True,
            supports_streaming=True,
            supports_cost_limits=False,
            requires_auth=False,  # allow OSS mode; OpenAI auth optional
            auth_methods=["api_key", "oauth", "oss"],
        )

    async def validate_environment(
        self, auth_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate Codex can run (auth optional)."""
        # Codex supports multiple providers. For OpenAI, accept OPENAI_API_KEY via
        # auth_config or environment; otherwise, proceed without hard requirement.
        try:
            if auth_config and auth_config.get("api_key"):
                return True, None
            if os.environ.get("OPENAI_API_KEY"):
                return True, None
        except Exception:
            pass
        # No API key is still valid (OSS modes); runner will rely on container tooling
        return True, None

    async def prepare_environment(
        self,
        container: Optional["Container"],
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Prepare Codex environment variables for the container."""
        env: Dict[str, str] = {}

        # Prefer explicit auth_config over host env
        if auth_config:
            if auth_config.get("api_key"):
                env["OPENAI_API_KEY"] = str(auth_config["api_key"])
            if auth_config.get("base_url"):
                env["OPENAI_BASE_URL"] = str(auth_config["base_url"])
        # Host env fallback
        if "OPENAI_API_KEY" in os.environ:
            env.setdefault("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
        if "OPENAI_BASE_URL" in os.environ:
            env.setdefault("OPENAI_BASE_URL", os.environ["OPENAI_BASE_URL"])

        # Fallback to .env file for OPENAI_* if not present in process environment
        if "OPENAI_API_KEY" not in env or not env.get("OPENAI_API_KEY"):
            try:
                from dotenv import dotenv_values  # type: ignore
                values = dotenv_values()
                if values and values.get("OPENAI_API_KEY"):
                    env["OPENAI_API_KEY"] = str(values.get("OPENAI_API_KEY"))
                if values and values.get("OPENAI_BASE_URL") and "OPENAI_BASE_URL" not in env:
                    env["OPENAI_BASE_URL"] = str(values.get("OPENAI_BASE_URL"))
            except Exception:
                pass

        # Proxy passthrough will be handled in DockerManager based on network_egress
        return env

    async def prepare_container(
        self,
        container_config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Codex-specific container adjustments (none required)."""
        # Nothing special: DockerManager already enforces isolation and mounts.
        return container_config

    async def build_command(
        self,
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Build the Codex CLI command line.

        We use container isolation as the primary sandbox and stream JSON.
        """
        # Base command
        cmd: List[str] = [
            "codex",
            "exec",
            "--json",
            "-C",
            "/workspace",
        ]

        # Prefer running without interactive approvals and repo checks inside our containerized sandbox
        cmd += ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]

        # Model passthrough (if provided)
        if model:
            cmd += ["-m", model]

        # Resume/reattach: experimental; accept session_id when provided
        # (Codex may expect a rollout path; we pass session id to be safe if supported)
        if session_id:
            # Keep a conservative flag name to avoid hard dependency
            # Callers can augment through kwargs if a specific flag is required later
            cmd += ["--resume", session_id]

        # System prompt injection not standardized for Codex; forward when present
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            cmd += ["--system-prompt", str(system_prompt)]

        # Final positional: the prompt
        if prompt:
            cmd.append(prompt)

        return cmd

    async def parse_events(
        self,
        output_line: str,
        parser_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse Codex JSONL into runner events using CodexOutputParser."""
        if "_parser" not in parser_state:
            parser_state["_parser"] = CodexOutputParser()
        parser: CodexOutputParser = parser_state["_parser"]
        return parser.parse_line(output_line)

    async def extract_result(
        self,
        parser_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "_parser" not in parser_state:
            return {"session_id": None, "final_message": None, "metrics": {}}
        parser: CodexOutputParser = parser_state["_parser"]
        summary = parser.get_summary()
        return {
            "session_id": summary.get("session_id"),
            "final_message": summary.get("final_message"),
            "metrics": summary.get("metrics", {}),
        }

    async def execute(
        self,
        docker_manager: Any,
        container: "Container",
        prompt: str,
        model: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 3600,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute Codex CLI inside container and collect the result."""
        command = await self.build_command(
            prompt=prompt, model=model, session_id=session_id, **kwargs
        )
        result = await docker_manager.execute_command(
            container=container,
            command=command,
            plugin=self,
            event_callback=(
                (lambda ev: event_callback(ev)) if callable(event_callback) else None
            ),
            timeout_seconds=timeout_seconds,
            max_turns=kwargs.get("max_turns"),
        )
        return result

    async def handle_error(
        self,
        error: Exception,
        parser_state: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """Classify Codex failures into runner error taxonomy."""
        s = str(error).lower()
        if any(k in s for k in ("timeout", "timed out", "deadline")):
            return "timeout", True
        if any(k in s for k in ("unauthorized", "forbidden", "invalid api key")):
            return "auth", False
        if any(k in s for k in ("dns", "econnrefused", "enetwork", "network")):
            return "network", True
        if any(k in s for k in ("docker", "container", "exec")):
            return "docker", True
        # Generic Codex error; retryable by default
        return "codex", True
