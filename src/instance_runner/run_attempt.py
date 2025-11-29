"""Attempt orchestration with retry logic (delegated phases)."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from . import AgentError, DockerError, GitError, TimeoutError, ValidationError
from .docker_manager import DockerManager
from .git_operations import GitOperations
from .plugin_interface import RunnerPlugin
from .run_attempt_container import prepare_container_phase
from .run_attempt_execute import execute_agent_phase
from .run_attempt_finalize import finalize_success
from .run_attempt_events import make_emit_event
from .run_attempt_results import build_failure_result
from .run_attempt_validation import validate_inputs
from .run_attempt_workspace import prepare_workspace_phase
from .run_attempt_retry import is_retryable_error
from ..shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig

logger = logging.getLogger(__name__)


async def _run_instance_attempt(
    prompt: str,
    repo_path: Path,
    base_branch: str,
    branch_name: str,
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
    instance_id: str,
    container_name: str,
    model: str,
    resolved_model_id: Optional[str],
    session_id: Optional[str],
    operator_resume: bool,
    session_group_key: Optional[str],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
    startup_semaphore: Optional[asyncio.Semaphore],
    timeout_seconds: int,
    container_limits: ContainerLimits,
    auth_config: Optional[AuthConfig],
    reuse_container: bool,
    finalize: bool,
    docker_image: str,
    plugin: RunnerPlugin,
    attempt_number: int,
    total_attempts: int,
    system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    retry_config: RetryConfig,
    force_import: bool,
    import_policy: str,
    import_conflict_policy: str,
    skip_empty_import: bool,
    task_key: Optional[str] = None,
    network_egress: Optional[str] = None,
    max_turns: Optional[int] = None,
    allow_overwrite_protected_refs: bool = False,
    allow_global_session_volume: bool = False,
    agent_cli_args: Optional[list[str]] = None,
    force_commit: bool = False,
    workspace_include_branches: Optional[list[str]] = None,
) -> InstanceResult:
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = None
    container = None
    agent_session_id = None
    docker_manager: Optional[DockerManager] = None
    metrics: Dict[str, Any] = {}
    final_message_ref: Dict[str, Any] = {"value": ""}
    final_message = ""
    log_path = (
        f"./logs/{run_id}/instance_{instance_id}.log"
        if run_id and instance_id
        else None
    )
    emit_event = make_emit_event(
        event_callback, instance_id, final_message_ref, metrics
    )
    try:
        logger.info(f"Starting instance {instance_id} with prompt: {prompt[:100]}...")
        emit_event(
            "instance.started",
            {
                "prompt": prompt,
                "model": model,
                "attempt": attempt_number,
                "total_attempts": total_attempts,
                "is_retry": attempt_number > 1,
                "session_id": session_id,
                "log_path": log_path,
            },
        )
        validate_inputs(repo_path, container_name, branch_name)
        docker_manager = DockerManager()
        git_ops = GitOperations()
        await docker_manager.initialize()
        if not await docker_manager.validate_environment():
            raise DockerError("Docker daemon is not accessible")
        emit_event("instance.phase_completed", {"phase": "validation"})
        workspace_dir = await prepare_workspace_phase(
            git_ops,
            repo_path,
            base_branch,
            instance_id,
            run_id,
            strategy_execution_id,
            container_name,
            operator_resume,
            session_id,
            workspace_include_branches,
            emit_event,
            startup_semaphore,
        )
        emit_event("instance.workspace_prepared", {"workspace_dir": str(workspace_dir)})
        emit_event("instance.phase_completed", {"phase": "workspace_preparation"})
        container = await prepare_container_phase(
            docker_manager,
            plugin,
            container_name,
            workspace_dir,
            container_limits,
            {
                "run_id": run_id,
                "strategy_execution_id": strategy_execution_id,
                "docker_image": docker_image,
            },
            instance_id,
            session_id,
            auth_config,
            reuse_container,
            session_group_key,
            import_policy,
            network_egress,
            emit_event,
            task_key,
            getattr(plugin, "name", None),
            resolved_model_id,
            allow_global_session_volume,
        )
        emit_event("instance.phase_completed", {"phase": "container_creation"})
        result_data = await execute_agent_phase(
            docker_manager,
            plugin,
            container,
            prompt,
            model,
            resolved_model_id,
            session_id,
            system_prompt,
            append_system_prompt,
            agent_cli_args,
            max_turns,
            timeout_seconds,
            emit_event,
            run_id,
            instance_id,
        )
        agent_session_id = result_data.get("session_id")
        final_message = result_data.get("final_message", "")
        metrics = result_data.get("metrics", {}) or {}
        emit_event(
            "instance.agent_completed",
            {
                "session_id": agent_session_id,
                "final_message": final_message,
                "metrics": metrics,
            },
        )
        emit_event("instance.phase_completed", {"phase": "agent_execution"})
        final_message = final_message_ref.get("value", final_message)
        git_ops_result = await git_ops.import_branch(
            repo_path=repo_path,
            workspace_dir=workspace_dir,
            branch_name=branch_name,
            import_policy=import_policy,
            import_conflict_policy=import_conflict_policy,
            skip_empty_import=skip_empty_import,
            task_key=task_key,
            run_id=run_id,
            strategy_execution_id=strategy_execution_id,
            allow_overwrite_protected_refs=allow_overwrite_protected_refs,
        )
        branch_name = git_ops_result.get("target_branch") or branch_name
        has_changes = git_ops_result.get("has_changes") == "true"
        commit_statistics = git_ops_result.get("commit_statistics")
        emit_event("instance.phase_completed", {"phase": "import"})
        return await finalize_success(
            emit_event,
            docker_manager,
            git_ops,
            container,
            workspace_dir,
            allow_global_session_volume,
            finalize,
            branch_name,
            has_changes,
            final_message,
            agent_session_id,
            metrics,
            commit_statistics,
            started_at,
            start_time,
            run_id,
            instance_id,
            total_attempts,
            attempt_number,
            model,
            resolved_model_id,
            log_path,
        )
    except (AgentError, DockerError, GitError, ValidationError, TimeoutError) as exc:
        error_type = (
            "docker"
            if isinstance(exc, DockerError)
            else (
                "git"
                if isinstance(exc, GitError)
                else (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else (
                        "agent"
                        if isinstance(exc, AgentError)
                        else (
                            "validation"
                            if isinstance(exc, ValidationError)
                            else "unknown"
                        )
                    )
                )
            )
        )
        error_str = str(exc)
        failure_message = final_message_ref.get("value") or final_message or error_str
        emit_event(
            "instance.failed",
            {
                "error": error_str,
                "error_type": error_type,
                "attempt": attempt_number,
                "total_attempts": total_attempts,
                "is_retryable": is_retryable_error(error_str, error_type, retry_config),
            },
        )
        return build_failure_result(
            failure_message,
            error_type,
            attempt_number,
            start_time,
            started_at,
            agent_session_id,
            container,
            metrics,
            log_path,
            workspace_dir,
        )
    except Exception as exc:
        failure_message = final_message_ref.get("value") or str(exc)
        emit_event(
            "instance.failed",
            {
                "error": str(exc),
                "error_type": "unknown",
                "attempt": attempt_number,
                "total_attempts": total_attempts,
                "is_retryable": False,
            },
        )
        return build_failure_result(
            failure_message,
            "unknown",
            attempt_number,
            start_time,
            started_at,
            agent_session_id,
            container,
            metrics,
            log_path,
            workspace_dir,
        )
    finally:
        if docker_manager and container and not finalize:
            try:
                await docker_manager.cleanup_container(container)
            except Exception:
                pass
        if docker_manager:
            try:
                docker_manager.close()
            except Exception:
                pass


__all__ = ["_run_instance_attempt", "is_retryable_error"]
_is_retryable_error = is_retryable_error
