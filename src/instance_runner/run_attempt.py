"""Attempt execution helpers for the instance runner."""

from __future__ import annotations

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
from ..shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig

logger = logging.getLogger(__name__)


def _is_retryable_error(
    error_str: str, error_type: str, retry_config: RetryConfig
) -> bool:
    """Check if an error is retryable based on pattern matching."""

    error_lower = error_str.lower()

    if error_type == "timeout":
        return True

    if error_type == "docker":
        for pattern in retry_config.docker_error_patterns:
            if pattern.lower() in error_lower:
                return True

    if error_type in ("agent",):
        for pattern in retry_config.agent_error_patterns:
            if pattern.lower() in error_lower:
                return True

    for pattern in retry_config.general_error_patterns:
        if pattern.lower() in error_lower:
            return True

    return False


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
    """Single attempt at running an instance (internal helper for retry logic)."""

    # The body is kept identical to the original implementation for semantic parity.
    # fmt: off
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = None
    container = None
    agent_session_id = None

    if run_id and instance_id:
        log_path = f"./logs/{run_id}/instance_{instance_id}.log"
    else:
        log_path = None

    def emit_event(event_type: str, data: Dict[str, Any]) -> None:
        if not event_callback:
            return
        try:
            if event_type.endswith("agent_result"):
                fm = data.get("final_message")
                if isinstance(fm, str) and fm:
                    nonlocal final_message
                    final_message = fm
                m = data.get("metrics") or {}
                if isinstance(m, dict) and m:
                    nonlocal metrics
                    metrics.update(m)
        except Exception:
            pass
        event_callback(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": instance_id,
                "data": data,
            }
        )

    docker_manager: Optional[DockerManager] = None
    final_message: str = ""
    metrics: Dict[str, Any] = {}
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

        if not repo_path.exists():
            raise ValidationError(f"Repository path does not exist: {repo_path}")

        if not repo_path.is_dir():
            raise ValidationError(f"Repository path is not a directory: {repo_path}")

        if not (repo_path / ".git").exists():
            raise ValidationError(f"Not a git repository: {repo_path}")

        if not container_name:
            raise ValidationError("Container name must be provided by orchestration")

        if not branch_name:
            raise ValidationError("Branch name must be provided by orchestration")

        docker_manager = DockerManager()
        git_ops = GitOperations()

        try:
            await docker_manager.initialize()

            logger.info("Validating Docker environment...")
            if not await docker_manager.validate_environment():
                raise DockerError("Docker daemon is not accessible")
            logger.info("Docker validation successful")

            emit_event("instance.phase_completed", {"phase": "validation"})

            logger.info(f"Preparing isolated workspace for branch {base_branch}")
            if startup_semaphore is not None:
                try:
                    emit_event(
                        "instance.startup_waiting",
                        {"base_branch": base_branch},
                    )
                except Exception:
                    pass
                async with startup_semaphore:
                    emit_event(
                        "instance.workspace_preparing", {"base_branch": base_branch}
                    )
                    logger.info(
                        "Calling git_ops.prepare_workspace (startup slot acquired)..."
                    )
                    _reuse_ws = bool(operator_resume and session_id)
                    workspace_dir = await git_ops.prepare_workspace(
                        repo_path=repo_path,
                        base_branch=base_branch,
                        instance_id=instance_id,
                        run_id=run_id,
                        strategy_execution_id=strategy_execution_id,
                        container_name=container_name,
                        reuse_if_exists=_reuse_ws,
                        include_branches=workspace_include_branches,
                    )
            else:
                emit_event("instance.workspace_preparing", {"base_branch": base_branch})
                logger.info("Calling git_ops.prepare_workspace...")
                _reuse_ws = bool(operator_resume and session_id)
                workspace_dir = await git_ops.prepare_workspace(
                    repo_path=repo_path,
                    base_branch=base_branch,
                    instance_id=instance_id,
                    run_id=run_id,
                    strategy_execution_id=strategy_execution_id,
                    container_name=container_name,
                    reuse_if_exists=_reuse_ws,
                    include_branches=workspace_include_branches,
                )
            logger.info(f"Workspace prepared at: {workspace_dir}")

            emit_event(
                "instance.workspace_prepared", {"workspace_dir": str(workspace_dir)}
            )
            emit_event("instance.phase_completed", {"phase": "workspace_preparation"})

            logger.info(f"Creating container {container_name}")
            emit_event(
                "instance.container_creating",
                {
                    "container_name": container_name,
                    "workspace_dir": str(workspace_dir),
                    "image": docker_image,
                },
            )

            emit_event(
                "instance.container_env_preparing", {"container_name": container_name}
            )
            from dataclasses import asdict

            env_vars = await plugin.prepare_environment(
                None, asdict(auth_config) if auth_config else None
            )
            try:
                emit_event(
                    "instance.container_env_prepared",
                    {
                        "container_name": container_name,
                        "env_keys": len(env_vars or {}),
                    },
                )
            except Exception:
                pass

            container = await docker_manager.create_container(
                container_name=container_name,
                workspace_dir=workspace_dir,
                cpu_count=container_limits.cpu_count,
                memory_gb=container_limits.memory_gb,
                memory_swap_gb=container_limits.memory_swap_gb,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                instance_id=instance_id,
                image=docker_image,
                session_id=session_id,
                auth_config=auth_config,
                reuse_container=reuse_container,
                extra_env=env_vars,
                plugin=plugin,
                session_group_key=session_group_key,
                import_policy=import_policy,
                network_egress=network_egress or "online",
                event_callback=event_callback,
                task_key=task_key,
                plugin_name=getattr(plugin, "name", None),
                resolved_model_id=resolved_model_id,
                allow_global_session_volume=allow_global_session_volume,
            )

            emit_event(
                "instance.container_created",
                {
                    "container_name": getattr(container, "name", container_name),
                    "container_id": getattr(container, "id", "")[:12],
                },
            )
            emit_event("instance.phase_completed", {"phase": "container_creation"})

            command = await plugin.build_command(
                prompt,
                model=resolved_model_id or model,
                session_id=session_id,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                agent_cli_args=agent_cli_args,
                max_turns=max_turns,
            )

            emit_event(
                "instance.agent_starting",
                {
                    "container_name": container_name,
                    "command": command,
                    "model": model,
                    "resolved_model_id": resolved_model_id,
                    "system_prompt_provided": bool(system_prompt),
                    "append_system_prompt_provided": bool(append_system_prompt),
                    "max_turns": max_turns,
                    "force_commit": force_commit,
                },
            )

            stream_log_path = None
            if run_id and instance_id:
                stream_log_path = f"./logs/{run_id}/runner_{instance_id}.log"

            result_data = await docker_manager.execute_command(
                container,
                command,
                plugin,
                event_callback=event_callback,
                timeout_seconds=timeout_seconds,
                max_turns=max_turns,
                stream_log_path=stream_log_path,
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

            git_ops_result = await git_ops.import_branch(
                repo_path=repo_path,
                workspace_dir=workspace_dir,
                branch_name=branch_name,
                base_branch=base_branch,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                instance_id=instance_id,
                container_name=container_name,
                force_import=force_import,
                import_policy=import_policy,
                import_conflict_policy=import_conflict_policy,
                skip_empty_import=skip_empty_import,
                allow_overwrite_protected_refs=allow_overwrite_protected_refs,
                force_commit=force_commit,
            )

            branch_name = git_ops_result.branch_name
            has_changes = git_ops_result.has_changes
            commit_statistics = git_ops_result.commit_statistics

            emit_event("instance.phase_completed", {"phase": "import"})

            if finalize and container:
                try:
                    await docker_manager.cleanup_container(
                        container, remove_home_volume=not allow_global_session_volume
                    )
                    emit_event(
                        "instance.container_removed",
                        {
                            "container_name": getattr(container, "name", container_name),
                            "container_id": getattr(container, "id", "")[:12],
                        },
                    )
                except Exception:
                    pass

            if finalize and workspace_dir:
                try:
                    await git_ops.cleanup_workspace(workspace_dir)
                    emit_event(
                        "instance.workspace_cleaned",
                        {"workspace_dir": str(workspace_dir)},
                    )
                except Exception:
                    pass

            completed_at = datetime.now(timezone.utc).isoformat()
            duration_seconds = time.time() - start_time
            emit_event(
                "instance.completed",
                {
                    "branch_name": branch_name,
                    "has_changes": has_changes,
                    "duration_seconds": duration_seconds,
                    "final_message": final_message,
                    "metrics": metrics,
                },
            )
            emit_event("instance.phase_completed", {"phase": "completed"})

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
                    "plugin": getattr(plugin, "name", "unknown"),
                    "model": model,
                    "resolved_model_id": resolved_model_id,
                },
            )

        except (AgentError, DockerError, GitError, ValidationError, TimeoutError) as exc:
            error_type = "unknown"
            if isinstance(exc, DockerError):
                error_type = "docker"
            elif isinstance(exc, GitError):
                error_type = "git"
            elif isinstance(exc, TimeoutError):
                error_type = "timeout"
            elif isinstance(exc, AgentError):
                error_type = "agent"
            elif isinstance(exc, ValidationError):
                error_type = "validation"

            error_str = str(exc)
            is_retryable = _is_retryable_error(error_str, error_type, retry_config)

            emit_event(
                "instance.failed",
                {
                    "error": error_str,
                    "error_type": error_type,
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                    "is_retryable": is_retryable,
                },
            )

            if container and finalize:
                try:
                    await docker_manager.cleanup_container(container)
                except Exception:
                    pass
            if workspace_dir and finalize:
                try:
                    await git_ops.cleanup_workspace(workspace_dir)
                except Exception:
                    pass

            result = InstanceResult(
                success=False,
                branch_name=None,
                has_changes=False,
                final_message=final_message or error_str,
                session_id=agent_session_id,
                container_name=getattr(container, "name", None),
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

        except Exception as exc:
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
            result = InstanceResult(
                success=False,
                branch_name=None,
                has_changes=False,
                final_message=str(exc),
                session_id=agent_session_id,
                container_name=getattr(container, "name", None),
                metrics=metrics,
                error=str(exc),
                error_type="unknown",
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                log_path=log_path,
                workspace_path=str(workspace_dir) if workspace_dir else None,
                status="failed",
            )
            result.retry_attempts = attempt_number
            return result
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
    # fmt: on


__all__ = ["_run_instance_attempt", "_is_retryable_error"]
