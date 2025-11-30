"""Retry-aware coordinator for running instances."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from pitaya.runner.attempt.executor import AttemptExecutor
from pitaya.shared.plugin import RunnerPlugin
from pitaya.runner.plugins import AVAILABLE_PLUGINS
from pitaya.runner.runner_params import RunnerParams
from pitaya.runner.runner_utils import is_retryable_error
from pitaya.exceptions import ValidationError
from pitaya.shared import ContainerLimits, InstanceResult, RetryConfig

logger = logging.getLogger(__name__)


class RunnerCoordinator:
    """Orchestrates retries and delegates single attempts."""

    def __init__(self, params: RunnerParams) -> None:
        self.plugin: RunnerPlugin = self._load_plugin(params.plugin_name)
        docker_image = params.docker_image or getattr(self.plugin, "docker_image", "")
        if not docker_image:
            raise ValidationError(
                "Docker image must be provided by the plugin or caller."
            )
        self.params = replace(params, docker_image=docker_image)
        self.resolved_model_id = params.model

    async def run(self) -> InstanceResult:
        await self._validate_plugin_environment()
        attempt = 0
        delay = self.params.retry_config.initial_delay_seconds
        last_result: InstanceResult | None = None
        reuse_container = self.params.reuse_container
        session_id = self.params.session_id

        while attempt < self.params.retry_config.max_attempts:
            attempt += 1
            current_session = (
                last_result.session_id
                if last_result and last_result.session_id
                else session_id
            )

            attempt_runner = AttemptExecutor(
                params=self.params,
                plugin=self.plugin,
                resolved_model_id=self.resolved_model_id,
                attempt_number=attempt,
                total_attempts=self.params.retry_config.max_attempts,
                reuse_container=reuse_container,
                session_id=current_session,
            )
            result = await attempt_runner.run()

            if result.success:
                return result

            retryable = is_retryable_error(
                result.error or "", result.error_type or "", self.params.retry_config
            )
            retryable, reuse_container, session_id, delay = (
                self._maybe_force_retry_on_resume(
                    retryable, result, reuse_container, session_id, delay
                )
            )

            if not retryable or attempt >= self.params.retry_config.max_attempts:
                return result

            last_result = result
            logger.info(
                "Retrying instance %s after %s error (attempt %s/%s)",
                self.params.instance_id,
                result.error_type,
                attempt + 1,
                self.params.retry_config.max_attempts,
            )
            await asyncio.sleep(delay)
            delay = min(
                delay * self.params.retry_config.exponential_base,
                self.params.retry_config.max_delay_seconds,
            )

        return last_result  # pragma: no cover - loop ensures return earlier

    def _maybe_force_retry_on_resume(
        self,
        retryable: bool,
        result: InstanceResult,
        reuse_container: bool,
        session_id: str | None,
        delay: float,
    ) -> tuple[bool, bool, str | None, float]:
        """Handle the resume-specific docker-exit scenario."""
        try:
            err = (result.error or "").lower()
            if (
                not retryable
                and self.params.operator_resume
                and (result.error_type or "").lower() == "docker"
                and ("exited with code" in err or "exit code" in err)
            ):
                retryable = True
                self._emit_retry_hint()
                reuse_container = False
                session_id = None
                delay = min(delay, 1.0)
        except Exception:
            pass
        return retryable, reuse_container, session_id, delay

    def _emit_retry_hint(self) -> None:
        if not self.params.event_callback:
            return
        self.params.event_callback(
            {
                "type": "instance.retrying",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": self.params.instance_id,
                "data": {"reason": "resume_exec_exit", "action": "fresh_container"},
            }
        )

    def _load_plugin(self, plugin_name: str) -> RunnerPlugin:
        if plugin_name not in AVAILABLE_PLUGINS:
            raise ValidationError(
                f"Unknown plugin: {plugin_name}. Available: {list(AVAILABLE_PLUGINS.keys())}"
            )
        plugin_class = AVAILABLE_PLUGINS[plugin_name]
        return plugin_class()

    async def _validate_plugin_environment(self) -> None:
        auth_dict = None
        if self.params.auth_config:
            auth_dict = {
                "oauth_token": self.params.auth_config.oauth_token,
                "api_key": self.params.auth_config.api_key,
                "base_url": self.params.auth_config.base_url,
            }
        is_valid, error_msg = await self.plugin.validate_environment(auth_dict)
        if not is_valid:
            raise ValidationError(
                f"Plugin {self.plugin.name} validation failed: {error_msg}"
            )


def build_runner_params(
    *,
    prompt,
    repo_path,
    base_branch,
    branch_name,
    run_id,
    strategy_execution_id,
    instance_id,
    container_name,
    model,
    session_id,
    operator_resume,
    session_group_key,
    event_callback,
    startup_semaphore,
    timeout_seconds,
    container_limits,
    auth_config,
    reuse_container,
    finalize,
    docker_image,
    retry_config,
    plugin_name,
    system_prompt,
    append_system_prompt,
    import_policy,
    import_conflict_policy,
    skip_empty_import,
    network_egress,
    max_turns,
    allow_overwrite_protected_refs,
    allow_global_session_volume,
    agent_cli_args,
    force_commit,
    workspace_include_branches,
    task_key,
    logs_dir,
) -> RunnerParams:
    """Create RunnerParams with defaults for missing identifiers."""
    instance = instance_id or str(uuid.uuid4())
    session_group = session_group_key or task_key or instance
    limits = container_limits or ContainerLimits()
    retry = retry_config or RetryConfig()
    params = {
        "prompt": prompt,
        "repo_path": repo_path,
        "base_branch": base_branch,
        "branch_name": branch_name,
        "run_id": run_id,
        "strategy_execution_id": strategy_execution_id,
        "instance_id": instance,
        "container_name": container_name,
        "model": model,
        "session_id": session_id,
        "operator_resume": operator_resume,
        "session_group_key": session_group,
        "event_callback": event_callback,
        "startup_semaphore": startup_semaphore,
        "timeout_seconds": timeout_seconds,
        "container_limits": limits,
        "auth_config": auth_config,
        "reuse_container": reuse_container,
        "finalize": finalize,
        "docker_image": docker_image or "",
        "retry_config": retry,
        "plugin_name": plugin_name,
        "system_prompt": system_prompt,
        "append_system_prompt": append_system_prompt,
        "import_policy": import_policy,
        "import_conflict_policy": import_conflict_policy,
        "skip_empty_import": skip_empty_import,
        "network_egress": network_egress,
        "max_turns": max_turns,
        "allow_overwrite_protected_refs": allow_overwrite_protected_refs,
        "allow_global_session_volume": allow_global_session_volume,
        "agent_cli_args": agent_cli_args,
        "force_commit": force_commit,
        "workspace_include_branches": workspace_include_branches,
        "task_key": task_key,
        "logs_dir": logs_dir,
    }
    return RunnerParams(**params)
