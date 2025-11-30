"""Execute the plugin (agent) inside the prepared container."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from pitaya.shared.plugin import RunnerPlugin
from .runner_params import RunnerParams


class AgentRunner:
    """Runs the agent plugin and returns session/metrics data."""

    def __init__(
        self,
        *,
        plugin: RunnerPlugin,
        params: RunnerParams,
        emit_event: Callable[[str, Dict[str, Any]], None],
        resolved_model_id: str,
        session_id: Optional[str],
        log_path: Optional[str],
        container,
        docker_manager,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> None:
        self.plugin = plugin
        self.params = params
        self.emit_event = emit_event
        self.resolved_model_id = resolved_model_id
        self.session_id = session_id
        self.log_path = log_path
        self.container = container
        self.docker_manager = docker_manager
        self.env_vars = env_vars or {}

    async def run(self) -> Dict[str, Any]:
        self.emit_event(
            "instance.agent_starting",
            {
                "model": self.params.model,
                "model_id": self.resolved_model_id,
                "session_id": self.session_id,
                "operator_resume": bool(self.params.operator_resume),
            },
        )

        attempt_prompt = (
            "Continue"
            if (self.params.operator_resume and self.session_id)
            else self.params.prompt
        )
        codex_provider_kwargs: Dict[str, str] = {}
        try:
            if getattr(self.plugin, "name", "") == "codex":
                provider_url = None
                if self.params.auth_config and getattr(
                    self.params.auth_config, "base_url", None
                ):
                    provider_url = self.params.auth_config.base_url
                if not provider_url:
                    provider_url = self.env_vars.get("OPENAI_BASE_URL")
                if provider_url:
                    codex_provider_kwargs["provider_base_url"] = provider_url
                    codex_provider_kwargs["provider_env_key"] = "OPENAI_API_KEY"
        except Exception:
            pass

        return await self.plugin.execute(
            docker_manager=self.docker_manager,
            container=self.container,
            prompt=attempt_prompt,
            model=self.resolved_model_id,
            session_id=self.session_id,
            timeout_seconds=self.params.timeout_seconds,
            event_callback=lambda event: self.emit_event(
                f"instance.agent_{event['type']}", event
            ),
            system_prompt=self.params.system_prompt,
            append_system_prompt=self.params.append_system_prompt,
            operator_resume=self.params.operator_resume,
            max_turns=self.params.max_turns,
            stream_log_path=self.log_path,
            agent_cli_args=(self.params.agent_cli_args or []),
            **codex_provider_kwargs,
        )
