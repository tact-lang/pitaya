"""Open-source Codex CLI plugin implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..codex_parser import CodexOutputParser
from ..plugin_interface import PluginCapabilities, RunnerPlugin
from ...exceptions import AgentError

if TYPE_CHECKING:
    from docker.models.containers import Container

EventCallback = Callable[[Dict[str, Any]], None]
ParserState = Dict[str, Any]

logger = logging.getLogger(__name__)

__all__ = ["CodexPlugin"]

_ENV_API_KEY = "OPENAI_API_KEY"
_ENV_BASE_URL = "OPENAI_BASE_URL"
_ENV_PROVIDER_OVERRIDE = "CODEX_ENV_KEY"
_ENV_BASE_OVERRIDE = "CODEX_BASE_URL"
_ENV_PROVIDER_NAME_OVERRIDE = "CODEX_MODEL_PROVIDER"

_PROVIDER_ENV_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"),
    ("GROQ_API_KEY", "GROQ_BASE_URL"),
    ("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    ("GEMINI_API_KEY", "GEMINI_BASE_URL"),
    ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    ("OLLAMA_API_KEY", "OLLAMA_BASE_URL"),
    ("ARCEEAI_API_KEY", "ARCEEAI_BASE_URL"),
)
_DEFAULT_IMAGE = "pitaya-agents:latest"
_BASE_COMMAND = ["codex", "exec", "--json", "-C", "/workspace"]
_SANDBOX_FLAGS = ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]


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
            requires_auth=False,  # allow OSS mode; OpenAI auth optional
            auth_methods=["api_key", "oss"],  # reflect actual supported auth paths
        )

    async def validate_environment(
        self, auth_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Ensure a Codex API key is available before running."""
        if auth_config and auth_config.get("api_key"):
            return True, None
        if _select_provider_env_key():
            return True, None
        return (
            False,
            (
                "Missing Codex API key. Provide --api-key, set runner.api_key, "
                "or export an environment variable like OPENAI_API_KEY."
            ),
        )

    async def prepare_environment(
        self,
        container: Optional["Container"],
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Prepare Codex environment variables for the container."""
        return _collect_codex_env(auth_config)

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

        # Model passthrough (if provided)
        if model:
            cmd += ["-m", model]

        provider_env_key = kwargs.get("provider_env_key") or _select_provider_env_key()
        provider_base_url = kwargs.get(
            "provider_base_url"
        ) or _select_provider_base_url(provider_env_key)

        provider_name = kwargs.get("provider_name") or _select_provider_name(
            provider_env_key
        )

        if provider_env_key and (
            provider_env_key != _ENV_API_KEY or provider_base_url or provider_name
        ):
            # Select a custom provider key and map model + provider config
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

        # Agent CLI passthrough args: insert before prompt
        extra_args = kwargs.get("agent_cli_args")
        if isinstance(extra_args, (list, tuple)):
            cmd += [str(arg) for arg in extra_args if arg is not None]

        # Final positional: the prompt
        if prompt:
            cmd.append(prompt)
        elif session_id:
            logger.debug(
                "codex: omitted prompt (resume flow) session_id=%s",
                str(session_id)[:16],
            )

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


def _collect_codex_env(auth_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env: Dict[str, str] = {}

    if auth_config:
        if auth_config.get("api_key"):
            env[_ENV_API_KEY] = str(auth_config["api_key"])
        if auth_config.get("base_url"):
            env[_ENV_BASE_URL] = str(auth_config["base_url"])

    for env_key, base_key in _PROVIDER_ENV_CANDIDATES:
        if env_key in os.environ:
            env.setdefault(env_key, os.environ[env_key])
        if base_key in os.environ:
            env.setdefault(base_key, os.environ[base_key])

    if _ENV_PROVIDER_OVERRIDE in os.environ:
        override_env = os.environ[_ENV_PROVIDER_OVERRIDE]
        if override_env and override_env in os.environ:
            env.setdefault(override_env, os.environ[override_env])
    if _ENV_BASE_OVERRIDE in os.environ:
        env.setdefault(_ENV_BASE_URL, os.environ[_ENV_BASE_OVERRIDE])

    return env


def _select_provider_env_key() -> Optional[str]:
    override = os.environ.get(_ENV_PROVIDER_OVERRIDE)
    if override:
        return override
    if _ENV_API_KEY in os.environ:
        return _ENV_API_KEY
    for env_key, _ in _PROVIDER_ENV_CANDIDATES:
        if env_key in os.environ:
            return env_key
    return None


def _select_provider_base_url(env_key: Optional[str]) -> Optional[str]:
    if os.environ.get(_ENV_BASE_OVERRIDE):
        return os.environ[_ENV_BASE_OVERRIDE]
    if env_key:
        for candidate_key, base_key in _PROVIDER_ENV_CANDIDATES:
            if candidate_key == env_key and base_key in os.environ:
                return os.environ[base_key]
    if _ENV_BASE_URL in os.environ:
        return os.environ[_ENV_BASE_URL]
    return None


def _select_provider_name(env_key: Optional[str]) -> Optional[str]:
    if os.environ.get(_ENV_PROVIDER_NAME_OVERRIDE):
        return os.environ[_ENV_PROVIDER_NAME_OVERRIDE]
    if not env_key:
        return None
    if env_key == _ENV_API_KEY:
        return None
    prefix = env_key.lower().replace("_api_key", "")
    if prefix:
        return prefix
    return None
