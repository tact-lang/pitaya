"""Single-instance execution helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from ..exceptions import DockerError, GitError, OrchestratorError
from ..instance_runner import run_instance
from ..shared import ContainerLimits, InstanceResult, InstanceStatus
from .instance_events import (
    build_event_callback,
    heartbeat_monitor,
    map_error_type,
    strategy_execution_id,
    truncate_final_message,
)

logger = logging.getLogger(__name__)


def _emit_instance_started(orchestrator, info, instance_id: str, task_key: Optional[str]) -> None:
    orchestrator.event_bus.emit(
        "instance.started",
        {
            "strategy": info.strategy_name,
            "prompt": info.prompt,
            "model": info.metadata.get("model", "sonnet"),
            "branch_name": info.branch_name,
        },
        instance_id=instance_id,
    )
    if not task_key:
        return
    orchestrator.event_bus.emit_canonical(
        type="task.started",
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=strategy_execution_id(orchestrator, instance_id),
        key=task_key,
        payload={
            "key": task_key,
            "instance_id": instance_id,
            "container_name": info.container_name,
            "model": info.metadata.get("model", "sonnet"),
        },
    )


def _emit_task_completed(orchestrator, info, instance_id: str, result, task_key: str) -> None:
    artifact = {
        "type": "branch",
        "branch_planned": info.branch_name,
        "branch_final": result.branch_name,
        "base": info.base_branch,
        "commit": getattr(result, "commit", None) or "",
        "has_changes": result.has_changes,
        "duplicate_of_branch": getattr(result, "duplicate_of_branch", None),
        "dedupe_reason": getattr(result, "dedupe_reason", None),
    }
    run_logs = orchestrator.logs_dir / orchestrator.state_manager.current_state.run_id
    final_message, truncated_flag, rel_path = truncate_final_message(
        result.final_message or "", 65536, run_logs, instance_id
    )
    orchestrator.event_bus.emit_canonical(
        type="task.completed",
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=strategy_execution_id(orchestrator, instance_id),
        key=task_key,
        payload={
            "key": task_key,
            "instance_id": instance_id,
            "artifact": artifact,
            "metrics": result.metrics or {},
            "final_message": final_message,
            "final_message_truncated": truncated_flag,
            "final_message_path": rel_path,
        },
    )


def _emit_task_failed(orchestrator, info, instance_id: str, result, task_key: str) -> None:
    orchestrator.event_bus.emit_canonical(
        type="task.failed",
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=strategy_execution_id(orchestrator, instance_id),
        key=task_key,
        payload={
            "key": task_key,
            "instance_id": instance_id,
            "error_type": map_error_type(result, info),
            "message": result.error or "",
            "network_egress": (info.metadata or {}).get("network_egress"),
        },
    )


def _prepare_run_kwargs(orchestrator, info, strategy_exec_id, task_key, startup_pool):
    eff_cpu = max(
        1,
        int((info.metadata or {}).get("container_cpu", orchestrator.container_limits.cpu_count)),
    )
    eff_mem = max(
        1,
        int((info.metadata or {}).get("container_memory_gb", orchestrator.container_limits.memory_gb)),
    )
    return dict(
        prompt=info.prompt,
        repo_path=orchestrator.state_manager.current_state.repo_path,
        base_branch=info.base_branch,
        branch_name=info.branch_name,
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=strategy_exec_id,
        instance_id=info.instance_id,
        task_key=task_key,
        container_name=info.container_name,
        model=info.metadata.get("model", "sonnet"),
        session_id=(info.session_id or (info.metadata or {}).get("resume_session_id")),
        operator_resume=bool((info.metadata or {}).get("operator_resume", False)),
        event_callback=None,  # placeholder set by caller
        startup_semaphore=startup_pool,
        timeout_seconds=orchestrator.runner_timeout_seconds,
        container_limits=ContainerLimits(cpu_count=eff_cpu, memory_gb=eff_mem, memory_swap_gb=eff_mem),
        auth_config=orchestrator.auth_config,
        retry_config=orchestrator.retry_config,
        plugin_name=(info.metadata or {}).get("plugin_name", "claude-code"),
        docker_image=(info.metadata or {}).get("docker_image") or orchestrator.default_docker_image,
        import_policy=(info.metadata or {}).get("import_policy", "auto"),
        import_conflict_policy=(info.metadata or {}).get("import_conflict_policy", "fail"),
        skip_empty_import=bool((info.metadata or {}).get("skip_empty_import", True)),
        network_egress=(info.metadata or {}).get("network_egress"),
        max_turns=(info.metadata or {}).get("max_turns"),
        reuse_container=bool((info.metadata or {}).get("reuse_container", True)),
        allow_overwrite_protected_refs=orchestrator.allow_overwrite_protected_refs,
        allow_global_session_volume=orchestrator.allow_global_session_volume,
        agent_cli_args=(info.metadata or {}).get("agent_cli_args"),
        force_commit=orchestrator.force_commit,
        workspace_include_branches=(info.metadata or {}).get("workspace_include_branches"),
    )


def _apply_result_metadata(result, info, strategy_exec_id) -> None:
    if not (result and hasattr(result, "metadata")):
        return
    result.metadata = result.metadata or {}
    result.metadata.update(info.metadata or {})
    if strategy_exec_id:
        result.metadata["strategy_execution_id"] = strategy_exec_id


def _derive_state(result) -> InstanceStatus:
    if result.success:
        return InstanceStatus.COMPLETED
    if getattr(result, "status", None) == "canceled" or getattr(result, "error_type", None) == "canceled":
        return InstanceStatus.INTERRUPTED
    return InstanceStatus.FAILED


def _set_future(futures, instance_id: str, result: InstanceResult) -> None:
    futures[instance_id].set_result(result)


def _interrupt_instance(orchestrator, instance_id: str, futures) -> None:
    interrupt_result = InstanceResult(
        success=False,
        error="canceled",
        error_type="canceled",
        status="canceled",
    )
    orchestrator.state_manager.update_instance_state(
        instance_id=instance_id,
        state=InstanceStatus.INTERRUPTED,
        result=interrupt_result,
    )
    _set_future(futures, instance_id, interrupt_result)


def _fail_instance(orchestrator, instance_id: str, futures, exc: Exception) -> None:
    error_result = InstanceResult(
        success=False,
        error=str(exc),
        error_type="orchestration",
    )
    orchestrator.state_manager.update_instance_state(
        instance_id=instance_id,
        state=InstanceStatus.FAILED,
        result=error_result,
    )
    _set_future(futures, instance_id, error_result)


def _unexpected_failure(orchestrator, instance_id: str, futures, exc: Exception) -> None:
    error_result = InstanceResult(
        success=False,
        error=str(exc),
        error_type="unexpected",
    )
    orchestrator.state_manager.update_instance_state(
        instance_id=instance_id,
        state=InstanceStatus.FAILED,
        result=error_result,
    )
    _set_future(futures, instance_id, error_result)


async def execute_instance(
    *,
    orchestrator,
    instance_id: str,
    startup_pool: asyncio.Semaphore,
    futures: Dict[str, asyncio.Future],
) -> None:
    """Execute a single instance through the runner."""
    info = orchestrator.state_manager.current_state.instances.get(instance_id)
    if not info:
        logger.error("Instance %s not found in state", instance_id)
        return

    orchestrator.state_manager.update_instance_state(
        instance_id=instance_id,
        state=InstanceStatus.RUNNING,
    )
    task_key = (info.metadata or {}).get("key")
    _emit_instance_started(orchestrator, info, instance_id, task_key)

    event_callback = build_event_callback(orchestrator, instance_id, task_key)
    heartbeat_task: Optional[asyncio.Task] = None
    if orchestrator.event_bus:
        heartbeat_task = asyncio.create_task(heartbeat_monitor(orchestrator, instance_id))

    strategy_exec_id = strategy_execution_id(orchestrator, instance_id)

    try:
        kwargs = _prepare_run_kwargs(orchestrator, info, strategy_exec_id, task_key, startup_pool)
        kwargs["event_callback"] = event_callback
        result = await run_instance(**kwargs)

        _apply_result_metadata(result, info, strategy_exec_id)

        new_state = _derive_state(result)
        orchestrator.state_manager.update_instance_state(
            instance_id=instance_id,
            state=new_state,
            result=result,
        )

        if task_key:
            if result.success:
                _emit_task_completed(orchestrator, info, instance_id, result, task_key)
            elif new_state != InstanceStatus.INTERRUPTED:
                _emit_task_failed(orchestrator, info, instance_id, result, task_key)

        _set_future(futures, instance_id, result)

    except asyncio.CancelledError:
        _interrupt_instance(orchestrator, instance_id, futures)
    except (DockerError, GitError, OrchestratorError) as exc:
        if getattr(orchestrator, "_shutdown", False):
            _interrupt_instance(orchestrator, instance_id, futures)
        else:
            logger.exception("Instance %s execution failed: %s", instance_id, exc)
            _fail_instance(orchestrator, instance_id, futures, exc)
    except Exception as exc:  # noqa: BLE001
        if getattr(orchestrator, "_shutdown", False):
            _interrupt_instance(orchestrator, instance_id, futures)
        else:
            logger.exception("Instance %s crashed with unexpected error: %s", instance_id, exc)
            _unexpected_failure(orchestrator, instance_id, futures, exc)
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
