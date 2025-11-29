"""Instance and durable task handles."""

from __future__ import annotations

from ..shared import InstanceResult


class InstanceHandle:
    """Handle for tracking a spawned instance."""

    def __init__(self, instance_id: str, orchestrator) -> None:
        self.instance_id = instance_id
        self._orchestrator = orchestrator

    async def result(self) -> InstanceResult:
        results = await self._orchestrator.wait_for_instances([self.instance_id])
        return results[self.instance_id]


class Handle:
    """Durable task handle."""

    def __init__(self, key: str, instance_id: str, scheduled_at: float) -> None:
        self.key = key
        self.instance_id = instance_id
        self.scheduled_at = scheduled_at

    # Not awaitable directly; use context.wait(handle)
