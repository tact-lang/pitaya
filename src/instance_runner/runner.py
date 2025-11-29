"""Instance Runner entrypoint wrapper."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import ValidationError
from .plugin_interface import RunnerPlugin
from .plugins import AVAILABLE_PLUGINS
from .run_attempt import _run_instance_attempt, _is_retryable_error
from ..shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig

logger = logging.getLogger(__name__)


async def run_instance(
    prompt: str,
    repo_path: Path,
    base_branch: str = "main",
    branch_name: Optional[str] = None,
    run_id: Optional[str] = None,
    strategy_execution_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    task_key: Optional[str] = None,
    container_name: Optional[str] = None,
    model: str = "sonnet",
    session_id: Optional[str] = None,
    operator_resume: bool = False,
    session_group_key: Optional[str] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    startup_semaphore: Optional[asyncio.Semaphore] = None,
    timeout_seconds: int = 3600,
    container_limits: Optional[ContainerLimits] = None,
    auth_config: Optional[AuthConfig] = None,
    reuse_container: bool = True,
    finalize: bool = True,
    docker_image: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    plugin_name: str = "claude-code",
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    import_policy: str = "auto",
    import_conflict_policy: str = "fail",
    skip_empty_import: bool = True,
    network_egress: Optional[str] = None,
    max_turns: Optional[int] = None,
    allow_overwrite_protected_refs: bool = False,
    allow_global_session_volume: bool = False,
    agent_cli_args: Optional[list[str]] = None,
    force_commit: bool = False,
    workspace_include_branches: Optional[list[str]] = None,
) -> InstanceResult:
    """Execute a single AI coding instance in Docker."""

    logger.info(f"run_instance called with prompt: {prompt[:50]}...")
    instance_id = instance_id or str(uuid.uuid4())
    container_limits = container_limits or ContainerLimits()
    retry_config = retry_config or RetryConfig()
    if not session_group_key:
        session_group_key = task_key or instance_id

    if plugin_name not in AVAILABLE_PLUGINS:
        raise ValidationError(
            f"Unknown plugin: {plugin_name}. Available: {list(AVAILABLE_PLUGINS.keys())}"
        )

    plugin_class = AVAILABLE_PLUGINS[plugin_name]
    plugin: RunnerPlugin = plugin_class()

    if docker_image is None:
        docker_image = plugin.docker_image
    resolved_model_id = model

    auth_dict = None
    if auth_config:
        auth_dict = {
            "oauth_token": auth_config.oauth_token,
            "api_key": auth_config.api_key,
            "base_url": auth_config.base_url,
        }

    is_valid, error_msg = await plugin.validate_environment(auth_dict)
    if not is_valid:
        raise ValidationError(f"Plugin {plugin_name} validation failed: {error_msg}")

    attempt = 0
    delay = retry_config.initial_delay_seconds
    last_result: Optional[InstanceResult] = None
    force_import = False

    while attempt < retry_config.max_attempts:
        attempt += 1
        current_session_id = (
            last_result.session_id
            if last_result and last_result.session_id
            else session_id
        )

        try:
            result = await _run_instance_attempt(
                prompt=prompt,
                repo_path=repo_path,
                base_branch=base_branch,
                branch_name=branch_name,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                instance_id=instance_id,
                container_name=container_name,
                model=model,
                resolved_model_id=resolved_model_id,
                session_id=current_session_id,
                operator_resume=operator_resume,
                session_group_key=session_group_key,
                event_callback=event_callback,
                startup_semaphore=startup_semaphore,
                timeout_seconds=timeout_seconds,
                container_limits=container_limits,
                auth_config=auth_config,
                reuse_container=reuse_container,
                finalize=finalize,
                docker_image=docker_image,
                plugin=plugin,
                attempt_number=attempt,
                total_attempts=retry_config.max_attempts,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                retry_config=retry_config,
                force_import=force_import,
                import_policy=import_policy,
                import_conflict_policy=import_conflict_policy,
                skip_empty_import=skip_empty_import,
                task_key=task_key,
                network_egress=network_egress,
                max_turns=max_turns,
                allow_overwrite_protected_refs=allow_overwrite_protected_refs,
                allow_global_session_volume=allow_global_session_volume,
                agent_cli_args=agent_cli_args,
                force_commit=force_commit,
                workspace_include_branches=workspace_include_branches,
            )
            if result.success:
                return result

            last_result = result
            if attempt >= retry_config.max_attempts or not _is_retryable_error(
                result.error or "", result.error_type or "", retry_config
            ):
                return result

            await asyncio.sleep(delay)
            delay = min(
                delay * retry_config.exponential_base, retry_config.max_delay_seconds
            )

        except Exception as exc:
            logger.error(f"Instance attempt {attempt} failed: {exc}")
            last_result = InstanceResult(
                success=False,
                error=str(exc),
                error_type="unknown",
                status="failed",
            )
            if attempt >= retry_config.max_attempts:
                return last_result
            await asyncio.sleep(delay)
            delay = min(
                delay * retry_config.exponential_base, retry_config.max_delay_seconds
            )

    return last_result


__all__ = ["run_instance"]
