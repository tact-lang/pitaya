"""
Public API for the Instance Runner component.

This module provides the official interface for the Instance Runner that other
components should use. Direct imports from instance_runner implementation modules
are prohibited to maintain clean architectural boundaries.
"""

from pathlib import Path
import asyncio
from typing import Any, Callable, Dict, Optional

from ..shared import (
    AuthConfig,
    ContainerLimits,
    InstanceResult,
    RetryConfig,
)

# Import the actual implementation
from .runner import run_instance as _run_instance


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
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    startup_semaphore: Optional[asyncio.Semaphore] = None,
    timeout_seconds: int = 3600,
    container_limits: Optional[ContainerLimits] = None,
    auth_config: Optional[AuthConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    reuse_container: bool = True,
    finalize: bool = True,
    plugin_name: str = "claude-code",
    docker_image: Optional[str] = None,
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    # Import controls per spec
    import_policy: str = "auto",
    import_conflict_policy: str = "fail",
    skip_empty_import: bool = True,
    session_group_key: Optional[str] = None,
    network_egress: Optional[str] = None,
    max_turns: Optional[int] = None,
    allow_overwrite_protected_refs: bool = False,
    allow_global_session_volume: bool = False,
    agent_cli_args: Optional[list[str]] = None,
    force_commit: bool = False,
) -> InstanceResult:
    """
    Execute a single AI coding instance in an isolated environment.

    This is the primary entry point for running AI coding tasks. Each instance
    runs in its own Docker container with a dedicated git workspace, ensuring
    complete isolation between parallel executions.

    Args:
        prompt: The instruction for the agent tool to execute
        repo_path: Path to the host repository to work on
        base_branch: Starting branch to base work on (default: "main")
        branch_name: Target branch name for the result (provided by orchestration)
        run_id: Identifier for the overall run (provided by orchestration)
        strategy_execution_id: Identifier for the strategy execution
        instance_id: Unique identifier for this instance (auto-generated if not provided)
        container_name: Full container name to use (provided by orchestration)
        model: Model to use (default: "sonnet")
        session_id: Resume a previous agent session
        event_callback: Function to call with real-time events
        timeout_seconds: Maximum execution time (default: 3600)
        container_limits: CPU and memory restrictions
        auth_config: OAuth token or API key configuration
        retry_config: Retry behavior configuration
        reuse_container: Whether to reuse existing container (default: True)
        finalize: Whether to import branch and cleanup (default: True)
        plugin_name: Agent tool plugin name (default: "claude-code")
        docker_image: Override Docker image to use
        system_prompt: System prompt to prepend to instructions
        append_system_prompt: Additional system prompt to append
        import_policy: "auto|never|always"
        import_conflict_policy: "fail|overwrite|suffix"
        skip_empty_import: Whether to skip import when no changes

    Returns:
        InstanceResult containing execution outcome, metrics, and branch information

    Raises:
        OrchestratorError: Base exception for all instance runner errors
        DockerError: Docker daemon not available or container issues
        GitError: Repository access or git operation failures
        AgentError: Agent tool execution failures
        ValidationError: Invalid parameters
    """
    return await _run_instance(
        prompt=prompt,
        repo_path=repo_path,
        base_branch=base_branch,
        branch_name=branch_name,
        run_id=run_id,
        strategy_execution_id=strategy_execution_id,
        instance_id=instance_id,
        task_key=task_key,
        container_name=container_name,
        model=model,
        session_id=session_id,
        operator_resume=operator_resume,
        session_group_key=session_group_key,
        event_callback=event_callback,
        startup_semaphore=startup_semaphore,
        timeout_seconds=timeout_seconds,
        container_limits=container_limits,
        auth_config=auth_config,
        retry_config=retry_config,
        reuse_container=reuse_container,
        finalize=finalize,
        plugin_name=plugin_name,
        docker_image=docker_image,
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
    )


# Re-export only what's part of the public API
__all__ = ["run_instance"]
