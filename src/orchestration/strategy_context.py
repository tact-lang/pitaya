"""
Strategy Context abstraction for instance spawning and coordination.

This module provides the StrategyContext that strategies use to spawn instances
and coordinate execution, isolating them from orchestrator implementation details.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

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

    async def spawn_instance(
        self,
        prompt: str,
        base_branch: str,
        model: str = "sonnet",
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

        # Add model to metadata
        if metadata is None:
            metadata = {}
        metadata["model"] = model

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
        self._orchestrator.emit_event(event_type, data)
