"""
Public API for the Instance Runner component.

This module provides the official interface for the Instance Runner that other
components should use. Direct imports from instance_runner implementation modules
are prohibited to maintain clean architectural boundaries.
"""

from pathlib import Path
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
    container_name: Optional[str] = None,
    model: str = "sonnet",
    session_id: Optional[str] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    timeout_seconds: int = 3600,
    container_limits: Optional[ContainerLimits] = None,
    auth_config: Optional[AuthConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    reuse_container: bool = True,
    finalize: bool = True,
) -> InstanceResult:
    """
    Execute a single AI coding instance in an isolated environment.

    This is the primary entry point for running AI coding tasks. Each instance
    runs in its own Docker container with a dedicated git workspace, ensuring
    complete isolation between parallel executions.

    Args:
        prompt: The instruction for Claude Code to execute
        repo_path: Path to the host repository to work on
        base_branch: Starting branch to base work on (default: "main")
        branch_name: Target branch name for the result (provided by orchestration)
        run_id: Identifier for the overall run (provided by orchestration)
        strategy_execution_id: Identifier for the strategy execution
        instance_id: Unique identifier for this instance (auto-generated if not provided)
        container_name: Full container name to use (provided by orchestration)
        model: Claude model to use (default: "sonnet")
        session_id: Resume a previous Claude session
        event_callback: Function to call with real-time events
        timeout_seconds: Maximum execution time (default: 3600)
        container_limits: CPU and memory restrictions
        auth_config: OAuth token or API key configuration
        retry_config: Retry behavior configuration
        reuse_container: Whether to reuse existing container (default: True)
        finalize: Whether to import branch and cleanup (default: True)

    Returns:
        InstanceResult containing execution outcome, metrics, and branch information

    Raises:
        InstanceRunnerError: Base exception for all instance runner errors
        DockerError: Docker daemon not available or container issues
        GitError: Repository access or git operation failures
        ClaudeError: AI tool execution failures
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
        container_name=container_name,
        model=model,
        session_id=session_id,
        event_callback=event_callback,
        timeout_seconds=timeout_seconds,
        container_limits=container_limits,
        auth_config=auth_config,
        retry_config=retry_config,
        reuse_container=reuse_container,
        finalize=finalize,
    )


# Re-export only what's part of the public API
__all__ = ["run_instance"]
