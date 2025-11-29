"""Instance lifecycle management for Pitaya orchestrator.

This module keeps queueing, admission control, and executor loops separate
from the high-level Orchestrator so the main class stays small and readable.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Set

from ..exceptions import OrchestratorError
from ..shared import InstanceResult, InstanceStatus
from .instance_runner import execute_instance
from .instance_spawn import spawn_instance

logger = logging.getLogger(__name__)


class RandomAsyncQueue:
    """Random-pick async queue with a minimal asyncio.Queue-like interface."""

    def __init__(self) -> None:
        self._items: List[Any] = []
        self._cv: asyncio.Condition = asyncio.Condition()

    async def put(self, item: Any) -> None:
        async with self._cv:
            self._items.append(item)
            self._cv.notify(1)

    async def get(self) -> Any:
        async with self._cv:
            while not self._items:
                await self._cv.wait()
            idx = random.randrange(len(self._items))
            return self._items.pop(idx)

    def qsize(self) -> int:
        return len(self._items)


class InstanceManager:
    """Manage instance queueing and execution."""

    def __init__(
        self,
        orchestrator: "Orchestrator",
        *,
        randomize_queue_order: bool = False,
    ) -> None:
        # Lazy import to avoid circular typing
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:  # pragma: no cover - type checking only
            from .orchestrator import Orchestrator

        self._orch = orchestrator
        self._randomize_queue_order = randomize_queue_order
        self._instance_queue: asyncio.Queue | RandomAsyncQueue = (
            RandomAsyncQueue() if randomize_queue_order else asyncio.Queue()
        )
        self._resource_pool: asyncio.Semaphore = asyncio.Semaphore(1)
        self._startup_pool: asyncio.Semaphore = asyncio.Semaphore(1)
        self._instance_futures: Dict[str, asyncio.Future] = {}
        self._active_instances: Set[str] = set()
        self._executor_tasks: List[asyncio.Task] = []
        self._shutdown = False

    @property
    def futures(self) -> Dict[str, asyncio.Future]:
        return self._instance_futures

    @property
    def startup_pool(self) -> asyncio.Semaphore:
        return self._startup_pool

    @property
    def queue_size(self) -> int:
        return self._instance_queue.qsize()

    async def initialize(
        self, max_parallel_instances: int, max_parallel_startup: int
    ) -> None:
        """Configure semaphores and start executor tasks."""
        self._resource_pool = asyncio.Semaphore(max_parallel_instances)
        self._startup_pool = asyncio.Semaphore(max_parallel_startup)
        self._shutdown = False

        for _ in range(max_parallel_instances):
            task = asyncio.create_task(self._executor_loop())
            self._executor_tasks.append(task)

    async def shutdown(self) -> None:
        """Stop executor tasks and resolve any pending futures."""
        self._shutdown = True
        for task in self._executor_tasks:
            task.cancel()
        if self._executor_tasks:
            await asyncio.gather(*self._executor_tasks, return_exceptions=True)
        self._executor_tasks.clear()

        # Resolve any pending futures to prevent awaiters from hanging forever.
        for iid, fut in list(self._instance_futures.items()):
            if fut.done():
                continue
            interrupt_result = InstanceResult(
                success=False,
                error="canceled",
                error_type="canceled",
                status="canceled",
            )
            fut.set_result(interrupt_result)
            try:
                self._orch.state_manager.update_instance_state(
                    instance_id=iid, state=InstanceStatus.INTERRUPTED, result=interrupt_result
                )
            except Exception:
                pass

    async def spawn_instance(
        self,
        *,
        prompt: str,
        repo_path,
        base_branch: str,
        strategy_name: str,
        strategy_execution_id: str,
        instance_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> str:
        """Proxy to the spawn helper while keeping queue/future ownership local."""
        return await spawn_instance(
            orchestrator=self._orch,
            manager=self,
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
            strategy_name=strategy_name,
            strategy_execution_id=strategy_execution_id,
            instance_index=instance_index,
            metadata=metadata,
            key=key,
        )

    async def wait_for_instances(
        self,
        instance_ids: List[str],
    ) -> Dict[str, InstanceResult]:
        """Wait for multiple instances to complete."""
        futures = [self._instance_futures[iid] for iid in instance_ids]
        results = await asyncio.gather(*futures)
        return {iid: result for iid, result in zip(instance_ids, results)}

    async def enqueue(self, instance_id: str) -> None:
        """Put an instance id onto the execution queue."""
        await self._instance_queue.put(instance_id)

    async def _executor_loop(self) -> None:
        """Background worker that drains the queue and runs instances."""
        logger.info("Instance executor started")

        while not self._shutdown:
            try:
                try:
                    instance_id = await asyncio.wait_for(
                        self._instance_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                async with self._resource_pool:
                    if self._shutdown:
                        break
                    self._active_instances.add(instance_id)
                    await self._run_one(instance_id)
                    self._active_instances.discard(instance_id)
            except OrchestratorError as exc:
                logger.exception("Error in instance executor: %s", exc)
            except asyncio.CancelledError:
                break

    async def _run_one(self, instance_id: str) -> None:
        info = self._orch.state_manager.current_state.instances.get(instance_id)
        if info:
            cpu_need = max(
                1,
                int((info.metadata or {}).get("container_cpu", self._orch.container_limits.cpu_count)),
            )
            mem_need = max(
                1,
                int(
                    (info.metadata or {}).get(
                        "container_memory_gb", self._orch.container_limits.memory_gb
                    )
                ),
            )
        else:
            cpu_need = max(1, int(self._orch.container_limits.cpu_count))
            mem_need = max(1, int(self._orch.container_limits.memory_gb))

        await self._admission_wait(cpu_need, mem_need)
        try:
            await execute_instance(
                orchestrator=self._orch,
                instance_id=instance_id,
                startup_pool=self._startup_pool,
                futures=self._instance_futures,
            )
        finally:
            await self._admission_release(cpu_need, mem_need)

    async def _admission_wait(self, cpu_need: int, mem_need_gb: int) -> None:
        """No-op admission hook; kept for future resource gating."""
        _ = (cpu_need, mem_need_gb)
        return

    async def _admission_release(self, cpu: int, mem_gb: int) -> None:
        _ = (cpu, mem_gb)
        return
