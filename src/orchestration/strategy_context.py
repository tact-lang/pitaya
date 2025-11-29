"""Strategy Context abstraction for instance spawning and coordination."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..shared import InstanceResult
from .strategy_handles import Handle, InstanceHandle
from .strategy_random import DeterministicRand
from .strategy_task import _schedule_async


class StrategyContext:
    """Context object providing strategy execution capabilities."""

    def __init__(
        self,
        orchestrator,
        strategy_name: str,
        strategy_execution_id: str,
    ):
        self._orchestrator = orchestrator
        self._strategy_name = strategy_name
        self._strategy_execution_id = strategy_execution_id
        self._instance_counter = 0
        self._rand = DeterministicRand(orchestrator, strategy_execution_id)

    # Deterministic utilities per spec
    def key(self, *parts: Any) -> str:
        base = "/".join(str(p) for p in parts)
        try:
            suffix = getattr(self._orchestrator, "resume_key_suffix", None)
            if suffix:
                return f"{base}/{suffix}"
        except Exception:
            pass
        return base

    def now(self) -> float:
        return time.time()

    def rand(self) -> float:
        return self._rand.rand()

    async def spawn_instance(
        self,
        prompt: str,
        base_branch: str,
        model: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InstanceHandle:
        self._instance_counter += 1
        if not model:
            model = getattr(self._orchestrator, "default_model_alias", "sonnet")
        if metadata is None:
            metadata = {}
        metadata["model"] = model
        try:
            _args = getattr(self._orchestrator, "default_agent_cli_args", [])
            if _args and "agent_cli_args" not in metadata:
                metadata["agent_cli_args"] = list(_args)
        except Exception:
            pass

        instance_id = await self._orchestrator.spawn_instance(
            prompt=prompt,
            repo_path=self._orchestrator.repo_path,
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
        self._instance_counter += 1
        return await _schedule_async(self, task, key=key, policy=policy)

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
                    result=r,
                )
            except ImportError:
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
        instance_ids = [handle.instance_id for handle in handles]
        results_dict = await self._orchestrator.wait_for_instances(instance_ids)
        return [results_dict[handle.instance_id] for handle in handles]

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if getattr(self._orchestrator, "event_bus", None):
            self._orchestrator.event_bus.emit(event_type, data)
