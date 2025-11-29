"""Result construction helpers for instance attempts."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

from ..shared import InstanceResult


def build_success_result(
    branch_name: str,
    has_changes: bool,
    final_message: str,
    agent_session_id: Optional[str],
    container,
    metrics: Dict,
    commit_statistics,
    started_at: str,
    start_time: float,
    run_id: Optional[str],
    instance_id: str,
    workspace_dir,
    total_attempts: int,
    attempt_number: int,
    model: str,
    resolved_model_id: Optional[str],
    log_path: Optional[str],
) -> InstanceResult:
    completed_at = datetime.now(timezone.utc).isoformat()
    duration_seconds = time.time() - start_time
    return InstanceResult(
        success=True,
        branch_name=branch_name,
        has_changes=has_changes,
        final_message=final_message,
        session_id=agent_session_id,
        container_name=getattr(container, "name", None),
        metrics=metrics,
        duration_seconds=duration_seconds,
        commit_statistics=commit_statistics,
        started_at=started_at,
        completed_at=completed_at,
        log_path=log_path,
        workspace_path=str(workspace_dir) if workspace_dir else None,
        status="completed",
        metadata={
            "attempt": attempt_number,
            "total_attempts": total_attempts,
            "model": model,
            "resolved_model_id": resolved_model_id,
        },
    )


def build_failure_result(
    error_str: str,
    error_type: str,
    attempt_number: int,
    start_time: float,
    started_at: str,
    agent_session_id,
    container,
    metrics: Dict,
    log_path: Optional[str],
    workspace_dir,
) -> InstanceResult:
    result = InstanceResult(
        success=False,
        branch_name=None,
        has_changes=False,
        final_message=error_str,
        session_id=agent_session_id,
        container_name=getattr(container, "name", None) if container else None,
        metrics=metrics,
        error=error_str,
        error_type=error_type,
        duration_seconds=time.time() - start_time,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        log_path=log_path,
        workspace_path=str(workspace_dir) if workspace_dir else None,
        status="failed",
    )
    result.retry_attempts = attempt_number
    return result


__all__ = ["build_success_result", "build_failure_result"]
