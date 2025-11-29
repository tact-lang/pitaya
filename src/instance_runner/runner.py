"""Public entry point for running an instance with retry support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .runner_coordinator import RunnerCoordinator, build_runner_params
from ..shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig


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
    startup_semaphore: Optional["asyncio.Semaphore"] = None,
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
    """Execute a single AI coding instance in Docker (retry-aware)."""
    params = build_runner_params(
        prompt=prompt,
        repo_path=repo_path,
        base_branch=base_branch,
        branch_name=branch_name or "",
        run_id=run_id,
        strategy_execution_id=strategy_execution_id,
        instance_id=instance_id,
        container_name=container_name or "",
        model=model,
        session_id=session_id,
        operator_resume=operator_resume,
        session_group_key=session_group_key,
        event_callback=event_callback,
        startup_semaphore=startup_semaphore,
        timeout_seconds=timeout_seconds,
        container_limits=container_limits or ContainerLimits(),
        auth_config=auth_config,
        reuse_container=reuse_container,
        finalize=finalize,
        docker_image=docker_image,
        retry_config=retry_config or RetryConfig(),
        plugin_name=plugin_name,
        system_prompt=system_prompt,
        append_system_prompt=append_system_prompt,
        import_policy=import_policy,
        import_conflict_policy=import_conflict_policy,
        skip_empty_import=skip_empty_import,
        network_egress=network_egress,
        max_turns=max_turns,
        allow_overwrite_protected_refs=allow_overwrite_protected_refs,
        allow_global_session_volume=allow_global_session_volume,
        agent_cli_args=agent_cli_args,
        force_commit=force_commit,
        workspace_include_branches=workspace_include_branches,
        task_key=task_key,
    )
    coordinator = RunnerCoordinator(params)
    return await coordinator.run()
