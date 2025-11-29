"""Single-instance execution helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from ..exceptions import DockerError, GitError, OrchestratorError
from ..instance_runner import run_instance
from ..shared import ContainerLimits, InstanceResult, InstanceStatus

logger = logging.getLogger(__name__)


def _strategy_execution_id(orchestrator, instance_id: str) -> Optional[str]:
    for sid, strat in orchestrator.state_manager.current_state.strategies.items():
        if instance_id in strat.instance_ids:
            return sid
    return None


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
        strategy_execution_id=_strategy_execution_id(orchestrator, instance_id),
        key=task_key,
        payload={
            "key": task_key,
            "instance_id": instance_id,
            "container_name": info.container_name,
            "model": info.metadata.get("model", "sonnet"),
        },
    )


def _progress_payload(event: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    et = str(event.get("type", ""))
    data = event.get("data", {}) or {}
    phase = None
    activity = None
    tool = None
    if et == "instance.workspace_preparing":
        phase, activity = "workspace_preparing", "Preparing workspace..."
    elif et in {
        "instance.container_creating",
        "instance.container_create_call",
        "instance.container_create_attempt",
        "instance.container_image_check",
    }:
        phase, activity = "container_creating", "Creating container..."
    elif et == "instance.container_env_preparing":
        phase, activity = "container_env_preparing", "Preparing container env..."
    elif et == "instance.startup_waiting":
        phase, activity = "startup_waiting", "Waiting for startup slot..."
    elif et == "instance.container_env_prepared":
        phase, activity = "container_env_prepared", "Container env ready"
    elif et == "instance.container_created":
        phase, activity = "container_created", "Container created"
    elif et == "instance.container_adopted":
        phase, activity = "container_created", "Container adopted"
    elif et == "instance.agent_starting":
        phase, activity = "agent_starting", "Starting Agent..."
    elif et == "instance.result_collection_started":
        phase, activity = "result_collection", "Collecting results..."
    elif et == "instance.branch_imported":
        phase, activity = "branch_imported", f"Imported branch {data.get('branch_name','')}"
    elif et == "instance.no_changes":
        phase, activity = "no_changes", "No changes"
    elif et == "instance.workspace_cleaned":
        phase, activity = "cleanup", "Workspace cleaned"
    elif et == "instance.agent_tool_use":
        phase, tool = "tool_use", (data.get("tool") or data.get("data", {}).get("tool"))
        activity = f"Using {tool}" if tool else "Tool use"
    elif et == "instance.agent_assistant":
        phase, activity = "assistant", "Agent is thinking..."
    elif et == "instance.agent_system":
        phase, activity = "system", "Agent connected"
    extras: Dict[str, Any] = {}
    usage_payload = data.get("usage")
    if isinstance(usage_payload, dict):
        extras["usage"] = usage_payload
    message_id_val = data.get("message_id")
    if isinstance(message_id_val, str):
        extras["message_id"] = message_id_val
    if et == "instance.agent_turn_complete":
        phase = "assistant"
        activity = None
        tm = data.get("turn_metrics", {}) if isinstance(data, dict) else {}
        if isinstance(tm, dict):
            usage = {
                "tokens": int(tm.get("tokens", 0) or 0),
                "total_tokens": int(tm.get("total_tokens", 0) or 0),
            }
            if "input_tokens" in tm:
                usage["input_tokens"] = int(tm.get("input_tokens", 0) or 0)
            if "output_tokens" in tm:
                usage["output_tokens"] = int(tm.get("output_tokens", 0) or 0)
            extras.setdefault("usage", usage)

    payload: Dict[str, Any] = {}
    if activity:
        payload["activity"] = activity
    if tool:
        payload["tool"] = tool
    payload.update(extras)
    return phase, payload


def _build_event_callback(orchestrator, instance_id: str, task_key: Optional[str]):
    def _callback(event: Dict[str, Any]) -> None:
        data = event.get("data", {})
        session_id = data.get("session_id")
        if session_id:
            try:
                orchestrator.state_manager.update_instance_session_id(
                    instance_id, session_id
                )
            except Exception:
                pass
        try:
            orchestrator.event_bus.emit(
                event_type=event.get("type", "instance.event"),
                data=data,
                instance_id=instance_id,
            )
        except Exception:
            pass

        if not task_key:
            return

        try:
            phase, payload = _progress_payload(event)
            if not phase:
                return
            orchestrator.event_bus.emit_canonical(
                type="task.progress",
                run_id=orchestrator.state_manager.current_state.run_id,
                strategy_execution_id=_strategy_execution_id(orchestrator, instance_id),
                key=task_key,
                payload={
                    "key": task_key,
                    "instance_id": instance_id,
                    "phase": phase,
                    **payload,
                },
            )
        except Exception:
            pass

    return _callback


def _truncate_final_message(
    full_msg: str, max_bytes: int, run_logs_dir, instance_id: str
) -> tuple[str, bool, str]:
    msg_bytes = full_msg.encode("utf-8", errors="ignore")
    if max_bytes <= 0 or len(msg_bytes) <= max_bytes:
        return full_msg, False, ""
    truncated = msg_bytes[:max_bytes].decode("utf-8", errors="ignore")
    try:
        dest_dir = run_logs_dir / "final_messages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{instance_id}.txt"
        with open(dest_dir / fname, "w", encoding="utf-8", errors="ignore") as fh:
            fh.write(full_msg)
        return truncated, True, f"final_messages/{fname}"
    except Exception:
        return truncated, True, ""


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
    final_message, truncated_flag, rel_path = _truncate_final_message(
        result.final_message or "", 65536, run_logs, instance_id
    )
    orchestrator.event_bus.emit_canonical(
        type="task.completed",
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=_strategy_execution_id(orchestrator, instance_id),
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


def _map_error_type(result, info) -> str:
    mapping = {
        "docker": "docker",
        "git": "git",
        "timeout": "timeout",
        "auth": "auth",
        "session_corrupted": "session_corrupted",
        "claude": "api",
        "orchestration": "unknown",
        "validation": "unknown",
        "system": "unknown",
        "unexpected": "unknown",
    }
    etype = (result.error_type or "unknown").lower()
    mapped = mapping.get(
        etype,
        (
            etype
            if etype
            in {
                "docker",
                "api",
                "network",
                "git",
                "timeout",
                "session_corrupted",
                "auth",
                "unknown",
            }
            else "unknown"
        ),
    )
    try:
        if (info.metadata or {}).get("network_egress") == "offline" and mapped != "canceled":
            mapped = "network"
    except Exception:
        pass
    return mapped


def _emit_task_failed(orchestrator, info, instance_id: str, result, task_key: str) -> None:
    orchestrator.event_bus.emit_canonical(
        type="task.failed",
        run_id=orchestrator.state_manager.current_state.run_id,
        strategy_execution_id=_strategy_execution_id(orchestrator, instance_id),
        key=task_key,
        payload={
            "key": task_key,
            "instance_id": instance_id,
            "error_type": _map_error_type(result, info),
            "message": result.error or "",
            "network_egress": (info.metadata or {}).get("network_egress"),
        },
    )


async def heartbeat_monitor(orchestrator, instance_id: str, interval: float = 2.0) -> None:
    """Emit periodic debug logs about last observed event."""
    try:
        while True:
            last_type = None
            last_ts = None
            if orchestrator.event_bus:
                try:
                    for ev in reversed(orchestrator.event_bus.events):
                        if ev.get("instance_id") == instance_id:
                            last_type = ev.get("type")
                            last_ts = ev.get("timestamp")
                            break
                except Exception:
                    pass
            if last_type and last_ts:
                logger.debug(
                    "Heartbeat: instance %s last_event=%s at %s",
                    instance_id,
                    last_type,
                    last_ts,
                )
            else:
                logger.debug("Heartbeat: instance %s awaiting first event", instance_id)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        return


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

    event_callback = _build_event_callback(orchestrator, instance_id, task_key)
    heartbeat_task: Optional[asyncio.Task] = None
    if orchestrator.event_bus:
        heartbeat_task = asyncio.create_task(heartbeat_monitor(orchestrator, instance_id))

    strategy_execution_id = _strategy_execution_id(orchestrator, instance_id)
    eff_cpu = max(
        1,
        int((info.metadata or {}).get("container_cpu", orchestrator.container_limits.cpu_count)),
    )
    eff_mem = max(
        1,
        int((info.metadata or {}).get("container_memory_gb", orchestrator.container_limits.memory_gb)),
    )

    try:
        result = await run_instance(
            prompt=info.prompt,
            repo_path=orchestrator.state_manager.current_state.repo_path,
            base_branch=info.base_branch,
            branch_name=info.branch_name,
            run_id=orchestrator.state_manager.current_state.run_id,
            strategy_execution_id=strategy_execution_id,
            instance_id=instance_id,
            task_key=task_key,
            container_name=info.container_name,
            model=info.metadata.get("model", "sonnet"),
            session_id=(info.session_id or (info.metadata or {}).get("resume_session_id")),
            operator_resume=bool((info.metadata or {}).get("operator_resume", False)),
            event_callback=event_callback,
            startup_semaphore=startup_pool,
            timeout_seconds=orchestrator.runner_timeout_seconds,
            container_limits=ContainerLimits(
                cpu_count=eff_cpu, memory_gb=eff_mem, memory_swap_gb=eff_mem
            ),
            auth_config=orchestrator.auth_config,
            retry_config=orchestrator.retry_config,
            plugin_name=(info.metadata or {}).get("plugin_name", "claude-code"),
            docker_image=(info.metadata or {}).get("docker_image")
            or orchestrator.default_docker_image,
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

        if result and hasattr(result, "metadata"):
            result.metadata = result.metadata or {}
            result.metadata.update(info.metadata or {})
            if strategy_execution_id:
                result.metadata["strategy_execution_id"] = strategy_execution_id

        new_state = (
            InstanceStatus.COMPLETED
            if result.success
            else (
                InstanceStatus.INTERRUPTED
                if (
                    getattr(result, "status", None) == "canceled"
                    or getattr(result, "error_type", None) == "canceled"
                )
                else InstanceStatus.FAILED
            )
        )
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

        futures[instance_id].set_result(result)

    except asyncio.CancelledError:
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
        futures[instance_id].set_result(interrupt_result)
    except (DockerError, GitError, OrchestratorError) as exc:
        if getattr(orchestrator, "_shutdown", False):
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
            futures[instance_id].set_result(interrupt_result)
        else:
            logger.exception("Instance %s execution failed: %s", instance_id, exc)
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
            futures[instance_id].set_result(error_result)
    except Exception as exc:  # noqa: BLE001
        if getattr(orchestrator, "_shutdown", False):
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
            futures[instance_id].set_result(interrupt_result)
        else:
            logger.exception("Instance %s crashed with unexpected error: %s", instance_id, exc)
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
            futures[instance_id].set_result(error_result)
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
