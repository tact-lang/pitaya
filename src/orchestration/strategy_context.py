"""
Strategy Context abstraction for instance spawning and coordination.

This module provides the StrategyContext that strategies use to spawn instances
and coordinate execution, isolating them from orchestrator implementation details.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import hashlib
import json
import time

from ..shared import InstanceResult

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


class InstanceHandle:
    """
    Handle for tracking a spawned instance.

    Provides access to instance results and status without exposing
    orchestrator internals.
    """

    def __init__(self, instance_id: str, orchestrator: "Orchestrator"):
        self.instance_id = instance_id
        self._orchestrator = orchestrator

    async def result(self) -> InstanceResult:
        """Wait for the instance to complete and return its result."""
        results = await self._orchestrator.wait_for_instances([self.instance_id])
        return results[self.instance_id]


class Handle:
    """Durable task handle."""

    def __init__(self, key: str, instance_id: str, scheduled_at: float):
        self.key = key
        self.instance_id = instance_id
        self.scheduled_at = scheduled_at

    # Not awaitable directly; use context.wait(handle)


class StrategyContext:
    """
    Context object providing strategy execution capabilities.

    This abstraction isolates strategies from orchestrator implementation
    details while providing access to instance spawning and coordination.
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        strategy_name: str,
        strategy_execution_id: str,
    ):
        self._orchestrator = orchestrator
        self._strategy_name = strategy_name
        self._strategy_execution_id = strategy_execution_id
        self._instance_counter = 0
        self._rng_seq: List[float] = []
        self._rng_index: int = 0

    # Deterministic utilities per spec
    def key(self, *parts: Any) -> str:
        return "/".join(str(p) for p in parts)

    def now(self) -> float:
        return time.time()

    def rand(self) -> float:
        """Deterministic pseudo-random in [0,1) with canonical event emission.

        Uses a stable hash of (strategy_execution_id, seq) to derive a float in [0,1).
        Emits strategy.rand canonical event with {seq, value}.
        """
        self._rng_index += 1
        h = hashlib.sha256(
            f"{self._strategy_execution_id}:{self._rng_index}".encode("utf-8")
        ).hexdigest()
        # Take 8 hex bytes -> int -> normalize to [0,1)
        v_int = int(h[:8], 16)
        v = (v_int % 10_000_000) / 10_000_000.0
        self._rng_seq.append(v)
        try:
            if getattr(self._orchestrator, "event_bus", None) and getattr(
                self._orchestrator, "state_manager", None
            ):
                run_id = self._orchestrator.state_manager.current_state.run_id
                # Emit canonical event
                self._orchestrator.event_bus.emit_canonical(
                    type="strategy.rand",
                    run_id=run_id,
                    strategy_execution_id=self._strategy_execution_id,
                    payload={"seq": self._rng_index, "value": v},
                )
                # Mirror latest seq/value into state snapshot for fast resume
                try:
                    self._orchestrator.state_manager.update_strategy_rand(
                        self._strategy_execution_id, self._rng_index, v
                    )
                except Exception:
                    pass
        except Exception:
            pass
        return v

    async def spawn_instance(
        self,
        prompt: str,
        base_branch: str,
        model: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InstanceHandle:
        """
        Spawn a new instance with the given parameters.

        Args:
            prompt: Instruction for the AI agent
            base_branch: Starting branch for the instance
            model: AI model to use (default: "sonnet")
            metadata: Strategy-specific metadata to attach

        Returns:
            Handle for tracking the spawned instance
        """
        self._instance_counter += 1

        # Resolve model default from orchestrator when not explicitly provided
        if not model:
            model = getattr(self._orchestrator, "default_model_alias", "sonnet")

        # Add model to metadata
        if metadata is None:
            metadata = {}
        metadata["model"] = model

        # Include default agent CLI passthrough args when configured
        try:
            _args = getattr(self._orchestrator, "default_agent_cli_args", [])
            if _args and "agent_cli_args" not in metadata:
                metadata["agent_cli_args"] = list(_args)
        except Exception:
            pass

        instance_id = await self._orchestrator.spawn_instance(
            prompt=prompt,
            repo_path=self._orchestrator.repo_path,  # Context knows the repo
            base_branch=base_branch,
            strategy_name=self._strategy_name,
            strategy_execution_id=self._strategy_execution_id,
            instance_index=self._instance_counter,
            metadata=metadata,
        )

        return InstanceHandle(instance_id, self._orchestrator)

    # Durable task API
    async def run(
        self,
        task: Dict[str, Any],
        *,
        key: str,
        policy: Optional[Dict[str, Any]] = None,
    ) -> Handle:
        """Schedule a durable task and return a handle.

        Orchestration-level retry policy (optional):
        - Applies only to scheduling/reattach failures (pre-runner). Once the runner executes,
          the runner-level retry handles transient execution errors.
        - Schema example:
          {
            "max_attempts": 3,
            "backoff": {"type": "exponential", "base_s": 10, "factor": 6, "max_s": 360},
            "retry_on": ["docker", "api", "network", "git", "timeout", "unknown"]
          }
        """
        try:
            if getattr(self._orchestrator, "event_bus", None):
                self._orchestrator.event_bus.emit(
                    "strategy.debug",
                    {
                        "op": "run_start",
                        "key": key,
                        "task_keys": sorted(list(task.keys())),
                    },
                )
        except Exception:
            pass
        # Compute canonical fingerprint (JCS-like sorted JSON) with nulls dropped
        # Allow per-task CPU/memory overrides for admission and container limits
        _task_cpu = task.get("container_cpu")
        _task_mem = task.get("container_memory") or task.get("container_memory_gb")
        # Resolve defaults (model defaults to orchestrator.default_model_alias)
        _default_model = getattr(self._orchestrator, "default_model_alias", "sonnet")
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
                "plugin_name",
                getattr(self._orchestrator, "default_plugin_name", "claude-code"),
            ),
            "system_prompt": task.get("system_prompt"),
            "append_system_prompt": task.get("append_system_prompt"),
            "runner": {
                "network_egress": task.get("network_egress", "online"),
                "max_turns": task.get("max_turns"),
                # Include container defaults for fingerprint stability
                "container_cpu": (
                    _task_cpu
                    if _task_cpu is not None
                    else getattr(
                        getattr(self._orchestrator, "container_limits", None),
                        "cpu_count",
                        None,
                    )
                ),
                "container_memory": (
                    f"{int(_task_mem)}g"
                    if isinstance(_task_mem, (int, float))
                    else (
                        _task_mem
                        if isinstance(_task_mem, str)
                        else (
                            f"{getattr(getattr(self._orchestrator, 'container_limits', None), 'memory_gb', 0)}g"
                            if getattr(
                                getattr(self._orchestrator, "container_limits", None),
                                "memory_gb",
                                None,
                            )
                            is not None
                            else None
                        )
                    )
                ),
            },
        }

        # Drop nulls recursively
        def _drop_nulls(obj):
            if isinstance(obj, dict):
                return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [_drop_nulls(v) for v in obj if v is not None]
            else:
                return obj

        canonical = _drop_nulls(canonical)
        encoded = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        # On resume, prefer previously stored normalized input to avoid drift
        stored_encoded = None
        try:
            if (
                self._orchestrator
                and self._orchestrator.state_manager
                and self._orchestrator.state_manager.current_state
            ):
                stored = self._orchestrator.state_manager.current_state.tasks.get(key)
                if stored and isinstance(stored.get("input"), str):
                    stored_encoded = stored.get("input")
        except Exception:
            stored_encoded = None
        encoded_to_use = stored_encoded or encoded
        fingerprint = hashlib.sha256(encoded_to_use.encode("utf-8")).hexdigest()

        # Register task fingerprint
        try:
            self._orchestrator.state_manager.register_task(
                key, fingerprint, encoded_to_use
            )
        except Exception as e:
            # KeyConflictDifferentFingerprint
            if getattr(self._orchestrator, "event_bus", None):
                self._orchestrator.event_bus.emit(
                    "strategy.debug",
                    {
                        "op": "key_conflict",
                        "key": key,
                        "fingerprint": fingerprint,
                        "error": str(e),
                    },
                )
            raise

        # Derive model and prompt
        prompt = task.get("prompt", "")
        base_branch = task.get("base_branch", "main")
        model = task.get("model") or _default_model
        session_group_key = task.get("session_group_key") or key

        # Global force_import compatibility: treat as overwrite conflict policy
        try:
            if getattr(self._orchestrator, "_force_import", False):
                task.setdefault("import_conflict_policy", "overwrite")
        except Exception:
            pass

        # Spawn instance with durable key metadata (with optional orchestration-level retries)
        self._instance_counter += 1
        # Emit canonical debug event for scheduling request
        try:
            if getattr(self._orchestrator, "event_bus", None):
                # Diagnostic-only; do not write to public events.jsonl
                self._orchestrator.event_bus.emit(
                    "strategy.debug",
                    {
                        "op": "run_request",
                        "key": key,
                        "kind": "task",
                        "fingerprint": fingerprint[:16],
                    },
                )
        except Exception:
            pass
        # Provide default network_egress from orchestrator when task doesn't set it
        _default_egress = getattr(
            self._orchestrator, "default_network_egress", "online"
        )

        # Build metadata once (stable across retries)
        _metadata = {
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
                "plugin_name",
                getattr(self._orchestrator, "default_plugin_name", "claude-code"),
            ),
            **(
                {"container_cpu": int(_task_cpu)}
                if isinstance(_task_cpu, (int, float))
                else {}
            ),
            **(
                {"container_memory_gb": int(_task_mem)}
                if isinstance(_task_mem, (int, float))
                else {}
            ),
        }

        # Include default agent CLI passthrough args when configured
        try:
            _args = getattr(self._orchestrator, "default_agent_cli_args", [])
            if _args and "agent_cli_args" not in _metadata:
                _metadata["agent_cli_args"] = list(_args)
        except Exception:
            pass

        # Orchestration-level scheduling retry policy
        max_attempts = 1
        base_s = 0.0
        factor = 1.0
        max_s = 0.0
        retry_on = set()
        if isinstance(policy, dict):
            try:
                max_attempts = int(policy.get("max_attempts", 1))
            except Exception:
                max_attempts = 1
            bo = policy.get("backoff", {}) or {}
            try:
                base_s = float(bo.get("base_s", 0.0))
            except Exception:
                base_s = 0.0
            try:
                factor = float(bo.get("factor", 1.0))
            except Exception:
                factor = 1.0
            try:
                max_s = float(bo.get("max_s", 0.0))
            except Exception:
                max_s = 0.0
            r = policy.get("retry_on", []) or []
            try:
                retry_on = set(str(x).lower() for x in r)
            except Exception:
                retry_on = set()

        attempt = 0
        delay = base_s
        last_exc: Optional[Exception] = None
        instance_id: Optional[str] = None
        while attempt < max_attempts:
            attempt += 1
            try:
                instance_id = await self._orchestrator.spawn_instance(
                    prompt=prompt,
                    repo_path=self._orchestrator.repo_path,
                    base_branch=base_branch,
                    strategy_name=self._strategy_name,
                    strategy_execution_id=self._strategy_execution_id,
                    instance_index=self._instance_counter,
                    metadata=_metadata,
                    key=key,
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                # Classify error type against retry_on
                etype = "unknown"
                try:
                    from ..exceptions import (
                        DockerError,
                        GitError,
                        TimeoutError,
                        ValidationError,
                        OrchestratorError,
                    )

                    if isinstance(e, DockerError):
                        etype = "docker"
                    elif isinstance(e, GitError):
                        etype = "git"
                    elif isinstance(e, TimeoutError):
                        etype = "timeout"
                    elif isinstance(e, ValidationError):
                        # scheduling-time validation typically shouldn't be retried
                        etype = "validation"
                    elif isinstance(e, OrchestratorError):
                        etype = "unknown"
                    # Heuristic: network-related text
                    emsg = str(e).lower()
                    if any(
                        tok in emsg for tok in ("network", "econnrefused", "timeout")
                    ):
                        etype = "network" if etype not in {"docker", "git"} else etype
                except Exception:
                    pass

                # Decide to retry or raise
                should_retry = (attempt < max_attempts) and (
                    not retry_on or etype in retry_on
                )
                # Emit debug event
                try:
                    if getattr(self._orchestrator, "event_bus", None):
                        # Diagnostic-only; do not write to public events.jsonl
                        self._orchestrator.event_bus.emit(
                            "strategy.debug",
                            {
                                "op": "run_schedule_retry",
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "error_type": etype,
                                "delay_s": max(
                                    0.0, min(delay, max_s) if max_s > 0 else delay
                                ),
                                "message": str(e)[:500],
                            },
                        )
                except Exception:
                    pass
                if not should_retry:
                    raise
                # Backoff
                try:
                    import asyncio as _aio

                    sleep_for = delay
                    if max_s > 0 and sleep_for > max_s:
                        sleep_for = max_s
                    if sleep_for > 0:
                        await _aio.sleep(sleep_for)
                    delay = delay * factor if factor > 0 else delay
                except Exception:
                    pass

        if instance_id is None and last_exc is not None:
            # Exhausted retries
            raise last_exc
        try:
            if getattr(self._orchestrator, "event_bus", None):
                self._orchestrator.event_bus.emit(
                    "strategy.debug",
                    {"op": "run_spawned", "key": key, "iid": instance_id},
                )
        except Exception:
            pass

        # Emit canonical scheduled event
        if getattr(self._orchestrator, "event_bus", None):
            info_map = self._orchestrator.state_manager.current_state.instances
            info = info_map.get(instance_id)
            self._orchestrator.event_bus.emit_canonical(
                type="task.scheduled",
                run_id=self._orchestrator.state_manager.current_state.run_id,
                strategy_execution_id=self._strategy_execution_id,
                key=key,
                payload={
                    "key": key,
                    "instance_id": instance_id,
                    "container_name": info.container_name if info else "",
                    "branch_name": info.branch_name if info else "",
                    "model": model,
                    # Include base_branch so recovery after crashes can reconstruct
                    # a valid workspace target when snapshots lag.
                    "base_branch": base_branch,
                    "task_fingerprint_hash": fingerprint,
                },
            )

        try:
            if getattr(self._orchestrator, "event_bus", None):
                # Diagnostic-only; do not write to public events.jsonl
                self._orchestrator.event_bus.emit(
                    "strategy.debug",
                    {
                        "op": "run_enqueued",
                        "key": key,
                        "instance_id": instance_id,
                    },
                )
        except Exception:
            pass
        return Handle(key=key, instance_id=instance_id, scheduled_at=time.monotonic())

    async def wait(self, handle: Handle) -> InstanceResult:
        results = await self._orchestrator.wait_for_instances([handle.instance_id])
        r = results[handle.instance_id]
        if not getattr(r, "success", False):
            try:
                from ..exceptions import TaskFailed

                raise TaskFailed(
                    handle.key,
                    getattr(r, "error_type", "unknown") or "unknown",
                    getattr(r, "error", "") or "",
                )
            except ImportError:
                # Fallback to returning the result if exceptions module is unavailable
                return r
        return r

    async def wait_all(
        self, handles: List[Handle], tolerate_failures: bool = False
    ) -> Any:
        ids = [h.instance_id for h in handles]
        gathered = await self._orchestrator.wait_for_instances(ids)
        out = [gathered[i] for i in ids]
        if tolerate_failures:
            successes = [r for r in out if getattr(r, "success", False)]
            failures = [r for r in out if not getattr(r, "success", False)]
            return successes, failures
        if any(not getattr(r, "success", False) for r in out):
            try:
                from ..exceptions import AggregateTaskFailed

                failed_keys = [
                    h.key
                    for h, r in zip(handles, out)
                    if not getattr(r, "success", False)
                ]
                raise AggregateTaskFailed(failed_keys)
            except ImportError:
                raise RuntimeError("AggregateTaskFailed")
        return out

    async def parallel(self, handles: List[InstanceHandle]) -> List[InstanceResult]:
        """
        Execute multiple instances in parallel and return all results.

        This is a convenience method for the common pattern of spawning
        multiple instances and waiting for all to complete.

        Args:
            handles: List of instance handles to wait for

        Returns:
            List of instance results in the same order as handles
        """
        instance_ids = [handle.instance_id for handle in handles]
        results_dict = await self._orchestrator.wait_for_instances(instance_ids)

        # Return results in the same order as input handles
        return [results_dict[handle.instance_id] for handle in handles]

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit a strategy-level event.

        Args:
            event_type: Type of event to emit
            data: Event data
        """
        if getattr(self._orchestrator, "event_bus", None):
            self._orchestrator.event_bus.emit(event_type, data)
