"""Public entry point for running an instance with retry support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .runner_coordinator import RunnerCoordinator, build_runner_params
from .runner_params import RunnerParams
from pitaya.shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig


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
    logs_dir: Path = Path(".pitaya/logs"),
) -> InstanceResult:
    """Execute a single AI coding instance in Docker (retry-aware)."""
    params = _build_params_from_args(locals())
    coordinator = RunnerCoordinator(params)
    return await coordinator.run()


def _build_params_from_args(args: Dict[str, Any]) -> RunnerParams:
    """Translate run_instance kwargs into RunnerParams with defaults."""
    return build_runner_params(
        prompt=args["prompt"],
        repo_path=args["repo_path"],
        base_branch=args["base_branch"],
        branch_name=args["branch_name"] or "",
        run_id=args["run_id"],
        strategy_execution_id=args["strategy_execution_id"],
        instance_id=args["instance_id"],
        container_name=args["container_name"] or "",
        model=args["model"],
        session_id=args["session_id"],
        operator_resume=args["operator_resume"],
        session_group_key=args["session_group_key"],
        event_callback=args["event_callback"],
        startup_semaphore=args["startup_semaphore"],
        timeout_seconds=args["timeout_seconds"],
        container_limits=args["container_limits"] or ContainerLimits(),
        auth_config=args["auth_config"],
        reuse_container=args["reuse_container"],
        finalize=args["finalize"],
        docker_image=args["docker_image"],
        retry_config=args["retry_config"] or RetryConfig(),
        plugin_name=args["plugin_name"],
        system_prompt=args["system_prompt"],
        append_system_prompt=args["append_system_prompt"],
        import_policy=args["import_policy"],
        import_conflict_policy=args["import_conflict_policy"],
        skip_empty_import=args["skip_empty_import"],
        network_egress=args["network_egress"],
        max_turns=args["max_turns"],
        allow_overwrite_protected_refs=args["allow_overwrite_protected_refs"],
        allow_global_session_volume=args["allow_global_session_volume"],
        agent_cli_args=args["agent_cli_args"],
        force_commit=args["force_commit"],
        workspace_include_branches=args["workspace_include_branches"],
        task_key=args["task_key"],
        logs_dir=args["logs_dir"],
    )
