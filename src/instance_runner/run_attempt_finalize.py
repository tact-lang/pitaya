"""Finalize success path for run attempts."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .run_attempt_results import build_success_result


async def finalize_success(
    emit_event,
    docker_manager,
    git_ops,
    container,
    workspace_dir,
    allow_global_session_volume: bool,
    finalize: bool,
    branch_name: str,
    has_changes: bool,
    final_message: str,
    agent_session_id: Optional[str],
    metrics: Dict[str, Any],
    commit_statistics,
    started_at: str,
    start_time: float,
    run_id: Optional[str],
    instance_id: str,
    total_attempts: int,
    attempt_number: int,
    model: str,
    resolved_model_id: Optional[str],
    log_path: Optional[str],
):
    if finalize and container:
        try:
            await docker_manager.cleanup_container(
                container, remove_home_volume=not allow_global_session_volume
            )
            emit_event(
                "instance.container_removed",
                {
                    "container_name": getattr(container, "name", None),
                    "container_id": getattr(container, "id", "")[:12],
                },
            )
        except Exception:
            pass

    if finalize and workspace_dir:
        try:
            await git_ops.cleanup_workspace(workspace_dir)
            emit_event(
                "instance.workspace_cleaned", {"workspace_dir": str(workspace_dir)}
            )
        except Exception:
            pass

    emit_event(
        "instance.completed",
        {
            "branch_name": branch_name,
            "has_changes": has_changes,
            "duration_seconds": time.time() - start_time,
            "final_message": final_message,
            "metrics": metrics,
        },
    )
    emit_event("instance.phase_completed", {"phase": "completed"})

    return build_success_result(
        branch_name,
        has_changes,
        final_message,
        agent_session_id,
        container,
        metrics,
        commit_statistics,
        started_at,
        start_time,
        run_id,
        instance_id,
        workspace_dir,
        total_attempts,
        attempt_number,
        model,
        resolved_model_id,
        log_path,
    )


__all__ = ["finalize_success"]
