"""Structured parameters for running a single instance attempt."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..shared import AuthConfig, ContainerLimits, RetryConfig


@dataclass(frozen=True, slots=True)
class RunnerParams:
    """Immutable configuration for an instance run."""

    prompt: str
    repo_path: Path
    base_branch: str
    branch_name: str
    run_id: Optional[str]
    strategy_execution_id: Optional[str]
    instance_id: str
    container_name: str
    model: str
    session_id: Optional[str]
    operator_resume: bool
    session_group_key: Optional[str]
    event_callback: Optional[Callable[[Dict[str, Any]], None]]
    startup_semaphore: Optional[asyncio.Semaphore]
    timeout_seconds: int
    container_limits: ContainerLimits
    auth_config: Optional[AuthConfig]
    reuse_container: bool
    finalize: bool
    docker_image: str
    retry_config: RetryConfig
    plugin_name: str
    system_prompt: Optional[str]
    append_system_prompt: Optional[str]
    import_policy: str
    import_conflict_policy: str
    skip_empty_import: bool
    network_egress: Optional[str]
    max_turns: Optional[int]
    allow_overwrite_protected_refs: bool
    allow_global_session_volume: bool
    agent_cli_args: Optional[list[str]]
    force_commit: bool
    workspace_include_branches: Optional[list[str]]
    task_key: Optional[str]
