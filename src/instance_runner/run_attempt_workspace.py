"""Workspace preparation phase for instance attempts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .git_operations import GitOperations

logger = logging.getLogger(__name__)


async def prepare_workspace_phase(
    git_ops: GitOperations,
    repo_path: Path,
    base_branch: str,
    instance_id: str,
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
    container_name: str,
    operator_resume: bool,
    session_id: Optional[str],
    workspace_include_branches: Optional[list[str]],
    event_callback,
    startup_semaphore,
):
    logger.info(f"Preparing isolated workspace for branch {base_branch}")
    if startup_semaphore is not None:
        try:
            if event_callback:
                event_callback("instance.startup_waiting", {"base_branch": base_branch})
        except Exception:
            pass
        async with startup_semaphore:
            if event_callback:
                event_callback(
                    "instance.workspace_preparing", {"base_branch": base_branch}
                )
            logger.info("Calling git_ops.prepare_workspace (startup slot acquired)...")
            _reuse_ws = bool(operator_resume and session_id)
            return await git_ops.prepare_workspace(
                repo_path=repo_path,
                base_branch=base_branch,
                instance_id=instance_id,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                container_name=container_name,
                reuse_if_exists=_reuse_ws,
                include_branches=workspace_include_branches,
            )

    if event_callback:
        try:
            event_callback("instance.workspace_preparing", {"base_branch": base_branch})
        except Exception:
            pass
    logger.info("Calling git_ops.prepare_workspace...")
    _reuse_ws = bool(operator_resume and session_id)
    return await git_ops.prepare_workspace(
        repo_path=repo_path,
        base_branch=base_branch,
        instance_id=instance_id,
        run_id=run_id,
        strategy_execution_id=strategy_execution_id,
        container_name=container_name,
        reuse_if_exists=_reuse_ws,
        include_branches=workspace_include_branches,
    )


__all__ = ["prepare_workspace_phase"]
