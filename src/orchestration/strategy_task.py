"""Durable task scheduling logic for StrategyContext."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from .strategy_handles import Handle


def schedule_task(
    ctx,
    task: Dict[str, Any],
    *,
    key: str,
    policy: Optional[Dict[str, Any]],
) -> Tuple[Handle, Optional[Exception]]:
    return asyncio.get_event_loop().run_until_complete(
        _schedule_async(ctx, task, key=key, policy=policy)
    )


async def _schedule_async(
    ctx, task: Dict[str, Any], *, key: str, policy: Optional[Dict[str, Any]]
):
    orch = ctx._orchestrator
    _emit_debug(
        orch, {"op": "run_start", "key": key, "task_keys": sorted(list(task.keys()))}
    )

    fingerprint, encoded_to_use = _fingerprint_task(task, orch, key)
    _register_task(orch, key, fingerprint, encoded_to_use)

    prompt = task.get("prompt", "")
    base_branch = task.get("base_branch", "main")
    model = task.get("model") or getattr(orch, "default_model_alias", "sonnet")
    session_group_key = task.get("session_group_key") or key
    _metadata = _build_metadata(task, orch, key, fingerprint, model, session_group_key)

    max_attempts, backoff = _parse_policy(policy)

    attempt = 0
    delay = backoff["base_s"]
    instance_id = None
    last_exc: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            instance_id = await orch.spawn_instance(
                prompt=prompt,
                repo_path=orch.repo_path,
                base_branch=base_branch,
                strategy_name=ctx._strategy_name,
                strategy_execution_id=ctx._strategy_execution_id,
                instance_index=ctx._instance_counter,
                metadata=_metadata,
                key=key,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if not _should_retry(exc, attempt, max_attempts, backoff):
                raise
            _emit_debug(
                orch,
                {
                    "op": "run_schedule_retry",
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "error_type": _classify_error(exc),
                    "delay_s": _next_delay(delay, backoff),
                    "message": str(exc)[:500],
                },
            )
            await _sleep(delay, backoff)
            delay = delay * backoff["factor"] if backoff["factor"] > 0 else delay

    if instance_id is None and last_exc is not None:
        raise last_exc

    _emit_debug(orch, {"op": "run_spawned", "key": key, "iid": instance_id})
    _emit_scheduled_canonical(
        orch, ctx, key, instance_id, base_branch, model, fingerprint
    )
    _emit_debug(orch, {"op": "run_enqueued", "key": key, "instance_id": instance_id})
    return Handle(key=key, instance_id=instance_id, scheduled_at=time.monotonic())


def _fingerprint_task(task: Dict[str, Any], orch, key: str) -> Tuple[str, str]:
    _task_cpu = task.get("container_cpu")
    _task_mem = task.get("container_memory") or task.get("container_memory_gb")
    _default_model = getattr(orch, "default_model_alias", "sonnet")
    canonical = {
        "schema_version": "1",
        "prompt": task.get("prompt", ""),
        "base_branch": task.get("base_branch", "main"),
        "model": task.get("model", _default_model),
        "import_policy": task.get("import_policy", "auto"),
        "import_conflict_policy": task.get("import_conflict_policy", "fail"),
        "skip_empty_import": bool(task.get("skip_empty_import", True)),
        "session_group_key": task.get("session_group_key"),
        "resume_session_id": task.get("resume_session_id"),
        "plugin_name": task.get(
            "plugin_name", getattr(orch, "default_plugin_name", "claude-code")
        ),
        "system_prompt": task.get("system_prompt"),
        "append_system_prompt": task.get("append_system_prompt"),
        "runner": {
            "network_egress": task.get("network_egress", "online"),
            "max_turns": task.get("max_turns"),
            "container_cpu": (
                _task_cpu
                if _task_cpu is not None
                else getattr(getattr(orch, "container_limits", None), "cpu_count", None)
            ),
            "container_memory": _container_mem_value(_task_mem, orch),
        },
    }

    def _drop_nulls(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_drop_nulls(v) for v in obj if v is not None]
        return obj

    canonical = _drop_nulls(canonical)
    encoded = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    stored = None
    try:
        if orch and orch.state_manager and orch.state_manager.current_state:
            stored = orch.state_manager.current_state.tasks.get(key, {}).get("input")
    except Exception:
        stored = None
    encoded_to_use = stored or encoded
    fingerprint = hashlib.sha256(encoded_to_use.encode("utf-8")).hexdigest()
    return fingerprint, encoded_to_use


def _container_mem_value(_task_mem, orch) -> Any:
    if isinstance(_task_mem, (int, float)):
        return f"{int(_task_mem)}g"
    if isinstance(_task_mem, str):
        return _task_mem
    mem_gb = getattr(getattr(orch, "container_limits", None), "memory_gb", None)
    return f"{mem_gb}g" if mem_gb is not None else None


def _register_task(orch, key: str, fingerprint: str, encoded: str) -> None:
    try:
        orch.state_manager.register_task(key, fingerprint, encoded)
    except Exception as exc:
        _emit_debug(
            orch,
            {
                "op": "key_conflict",
                "key": key,
                "fingerprint": fingerprint,
                "error": str(exc),
            },
        )
        raise


def _build_metadata(
    task: Dict[str, Any],
    orch,
    key: str,
    fingerprint: str,
    model: str,
    session_group_key: str,
) -> Dict[str, Any]:
    _task_cpu = task.get("container_cpu")
    _task_mem = task.get("container_memory") or task.get("container_memory_gb")
    _default_egress = getattr(orch, "default_network_egress", "online")
    metadata = {
        "model": model,
        "key": key,
        "fingerprint": fingerprint,
        "session_group_key": session_group_key,
        "import_policy": task.get("import_policy", "auto"),
        "import_conflict_policy": task.get("import_conflict_policy", "fail"),
        "skip_empty_import": bool(task.get("skip_empty_import", True)),
        "resume_session_id": task.get("resume_session_id"),
        "network_egress": task.get("network_egress", _default_egress),
        "max_turns": task.get("max_turns"),
        "plugin_name": task.get(
            "plugin_name", getattr(orch, "default_plugin_name", "claude-code")
        ),
    }
    if isinstance(_task_cpu, (int, float)):
        metadata["container_cpu"] = int(_task_cpu)
    if isinstance(_task_mem, (int, float)):
        metadata["container_memory_gb"] = int(_task_mem)
    if (
        "workspace_include_branches" in task
        and task.get("workspace_include_branches") is not None
    ):
        metadata["workspace_include_branches"] = list(
            task.get("workspace_include_branches", [])
        )
    try:
        if not metadata.get("workspace_include_branches"):
            default_inc = getattr(orch, "default_workspace_include_branches", None)
            if default_inc:
                metadata["workspace_include_branches"] = list(default_inc)
    except Exception:
        pass
    try:
        _args = getattr(orch, "default_agent_cli_args", [])
        if _args and "agent_cli_args" not in metadata:
            metadata["agent_cli_args"] = list(_args)
    except Exception:
        pass
    try:
        if getattr(orch, "_force_import", False):
            task.setdefault("import_conflict_policy", "overwrite")
    except Exception:
        pass
    return metadata


def _parse_policy(policy: Optional[Dict[str, Any]]) -> tuple[int, Dict[str, float]]:
    max_attempts = 1
    base_s = 0.0
    factor = 1.0
    max_s = 0.0
    if isinstance(policy, dict):
        try:
            max_attempts = int(policy.get("max_attempts", 1))
        except Exception:
            max_attempts = 1
        bo = policy.get("backoff", {}) or {}
        base_s = _to_float(bo.get("base_s", 0.0), 0.0)
        factor = _to_float(bo.get("factor", 1.0), 1.0)
        max_s = _to_float(bo.get("max_s", 0.0), 0.0)
    return max_attempts, {"base_s": base_s, "factor": factor, "max_s": max_s}


def _to_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _classify_error(exc: Exception) -> str:
    try:
        from ..exceptions import (
            DockerError,
            GitError,
            TimeoutError,
            ValidationError,
            OrchestratorError,
        )

        if isinstance(exc, DockerError):
            return "docker"
        if isinstance(exc, GitError):
            return "git"
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, ValidationError):
            return "validation"
        if isinstance(exc, OrchestratorError):
            return "unknown"
        emsg = str(exc).lower()
        if any(tok in emsg for tok in ("network", "econnrefused", "timeout")):
            return "network"
    except Exception:
        pass
    return "unknown"


def _should_retry(
    exc: Exception, attempt: int, max_attempts: int, backoff: Dict[str, float]
) -> bool:
    _ = _classify_error(exc)
    return attempt < max_attempts


def _next_delay(current: float, backoff: Dict[str, float]) -> float:
    if backoff["max_s"] > 0 and current > backoff["max_s"]:
        return backoff["max_s"]
    return current


async def _sleep(delay: float, backoff: Dict[str, float]) -> None:
    if delay <= 0:
        return
    sleep_for = delay
    if backoff["max_s"] > 0 and sleep_for > backoff["max_s"]:
        sleep_for = backoff["max_s"]
    await asyncio.sleep(sleep_for)


def _emit_debug(orch, payload: Dict[str, Any]) -> None:
    try:
        if getattr(orch, "event_bus", None):
            orch.event_bus.emit("strategy.debug", payload)
    except Exception:
        pass


def _emit_scheduled_canonical(
    orch,
    ctx,
    key: str,
    instance_id: str,
    base_branch: str,
    model: str,
    fingerprint: str,
) -> None:
    if not getattr(orch, "event_bus", None):
        return
    info_map = orch.state_manager.current_state.instances
    info = info_map.get(instance_id)
    orch.event_bus.emit_canonical(
        type="task.scheduled",
        run_id=orch.state_manager.current_state.run_id,
        strategy_execution_id=ctx._strategy_execution_id,
        key=key,
        payload={
            "key": key,
            "instance_id": instance_id,
            "container_name": info.container_name if info else "",
            "branch_name": info.branch_name if info else "",
            "model": model,
            "base_branch": base_branch,
            "task_fingerprint_hash": fingerprint,
        },
    )
