"""Open-source Codex CLI plugin implementation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from pitaya.runner.parsing.codex_parser import CodexOutputParser
from pitaya.shared.plugin import PluginCapabilities, RunnerPlugin
from pitaya.exceptions import AgentError
from .codex_env import (
    ENV_API_KEY,
    collect_codex_env,
    select_provider_base_url,
    select_provider_env_key,
    select_provider_name,
)

if TYPE_CHECKING:
    from docker.models.containers import Container

EventCallback = Callable[[Dict[str, Any]], None]
ParserState = Dict[str, Any]

logger = logging.getLogger(__name__)

__all__ = ["CodexPlugin"]

_DEFAULT_IMAGE = "pitaya-agents:latest"
_BASE_COMMAND = ["codex", "exec", "--json", "-C", "/workspace"]
_SANDBOX_FLAGS = [
    "--skip-git-repo-check",
    "--sandbox",
    "danger-full-access",
]


class CodexPlugin(RunnerPlugin):
    """Runner plugin for Codex CLI."""

    def __init__(self) -> None:
        self._parser = CodexOutputParser()

    @property
    def name(self) -> str:  # pragma: no cover - simple property
        return "codex"

    @property
    def docker_image(self) -> str:  # pragma: no cover - simple property
        return _DEFAULT_IMAGE

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            supports_resume=True,  # via rollout/reattach when available
            supports_cost=False,
            supports_streaming_events=True,
            supports_token_counts=True,
            supports_streaming=True,
            supports_cost_limits=False,
            requires_auth=True,
            auth_methods=["api_key"],
        )

    async def validate_environment(
        self, auth_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Ensure a Codex API key is available before running."""
        if auth_config and auth_config.get("api_key"):
            return True, None
        if select_provider_env_key():
            return True, None
        return (
            False,
            (
                "Missing Codex API key. Provide --api-key, set runner.api_key, "
                "or export an environment variable such as CODEX_API_KEY or OPENAI_API_KEY."
            ),
        )

    async def prepare_environment(
        self,
        container: Optional["Container"],
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Prepare Codex environment variables for the container."""
        return collect_codex_env(auth_config)

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
        cmd: List[str] = list(_BASE_COMMAND)
        cmd += _SANDBOX_FLAGS
        self._append_model_flags(cmd, model)
        self._apply_provider_overrides(cmd, model, kwargs)
        cmd += ["-c", "features.web_search_request=true"]
        self._append_session_and_prompt(cmd, session_id, prompt)
        self._append_extra_args(cmd, kwargs.get("agent_cli_args"))
        return cmd

    async def parse_events(
        self,
        output_line: str,
        parser_state: ParserState,
    ) -> Optional[Dict[str, Any]]:
        """Parse Codex JSONL into runner events using CodexOutputParser."""
        if "_parser" not in parser_state:
            parser_state["_parser"] = CodexOutputParser()
        parser: CodexOutputParser = parser_state["_parser"]
        return parser.parse_line(output_line)

    async def extract_result(
        self,
        parser_state: ParserState,
    ) -> Dict[str, Any]:
        if "_parser" not in parser_state:
            return {"session_id": None, "final_message": None, "metrics": {}}
        parser: CodexOutputParser = parser_state["_parser"]
        summary = parser.get_summary()
        error_message = summary.get("error")
        if error_message:
            raise AgentError(str(error_message))
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
        event_callback: Optional[EventCallback] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute Codex CLI inside container and collect the result."""
        command = await self.build_command(
            prompt=prompt, model=model, session_id=session_id, **kwargs
        )
        # Optional raw stream log path (tee everything to file)
        stream_log_path = kwargs.get("stream_log_path")

        callback = event_callback if callable(event_callback) else None
        result = await docker_manager.execute_command(
            container=container,
            command=command,
            plugin=self,
            event_callback=callback,
            timeout_seconds=timeout_seconds,
            max_turns=kwargs.get("max_turns"),
            stream_log_path=stream_log_path,
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
        if any(
            k in s
            for k in (
                "unauthorized",
                "forbidden",
                "invalid api key",
                "must be verified",
            )
        ):
            return "auth", False
        if any(k in s for k in ("dns", "econnrefused", "enetwork", "network")):
            return "network", True
        if any(k in s for k in ("docker", "container", "exec")):
            return "docker", True
        # Generic Codex error; retryable by default
        return "codex", True

    def _append_model_flags(self, cmd: List[str], model: str) -> None:
        if model:
            cmd += ["-m", model]

    def _apply_provider_overrides(
        self, cmd: List[str], model: str, kwargs: Dict[str, Any]
    ) -> None:
        provider_env_key = kwargs.get("provider_env_key") or select_provider_env_key()
        provider_base_url = kwargs.get("provider_base_url") or select_provider_base_url(
            provider_env_key
        )
        provider_name = kwargs.get("provider_name") or select_provider_name(
            provider_env_key
        )
        if provider_env_key and (
            provider_env_key != ENV_API_KEY or provider_base_url or provider_name
        ):
            provider_label = provider_name or "pitaya_env"
            cmd += ["-c", f"model_provider={provider_label}"]
            provider_display = provider_name or "PitayaProvider"
            provider_parts = [f'name="{provider_display}"']
            if provider_base_url:
                provider_parts.append(f'base_url="{provider_base_url}"')
            provider_parts.append(f'env_key="{provider_env_key}"')
            cmd += [
                "-c",
                (
                    f"model_providers.{provider_label}="
                    "{" + ", ".join(provider_parts) + "}"
                ),
            ]
            if model:
                cmd += ["-c", f'model="{model}"']
        elif model:
            cmd += ["-c", f'model="{model}"']

    def _append_session_and_prompt(
        self, cmd: List[str], session_id: Optional[str], prompt: str
    ) -> None:
        if session_id:
            cmd.extend(["resume", session_id])
        if prompt:
            cmd.append(prompt)
        elif not session_id:
            logger.debug("codex: no prompt provided and no session_id to resume")

    def _append_extra_args(self, cmd: List[str], extra_args: Any) -> None:
        if isinstance(extra_args, (list, tuple)):
            cmd += [str(arg) for arg in extra_args if arg is not None]


def _collect_codex_env(auth_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    return collect_codex_env(auth_config)


def _select_provider_env_key() -> Optional[str]:
    return select_provider_env_key()
