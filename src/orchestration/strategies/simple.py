"""
Simple strategy that runs exactly one instance.

This is the baseline strategy - no parallelism, no complexity.
Perfect for single tasks where you want precisely one attempt.
"""

import logging
from typing import List, TYPE_CHECKING

from ...shared import InstanceResult
from ...exceptions import StrategyError
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext

logger = logging.getLogger(__name__)


class SimpleStrategy(Strategy):
    """
    Executes exactly one instance with the given prompt.

    If the instance fails, the strategy fails. No retries,
    no fallbacks - just one clean execution.
    """

    @property
    def name(self) -> str:
        return "simple"

    def get_config_class(self) -> type[StrategyConfig]:
        # Simple strategy uses base config with no extensions
        return StrategyConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """
        Execute one instance and return its result.

        Args:
            prompt: Instruction for the AI agent
            base_branch: Starting branch
            ctx: Strategy context for spawning instances

        Returns:
            List containing the single instance result
        """
        if not (prompt or "").strip():
            raise StrategyError("simple: prompt is required (non-empty)")
        logger.info(f"SimpleStrategy.execute called with prompt: {prompt[:50]}...")

        # Durable run with key
        t = {
            "prompt": prompt,
            "base_branch": base_branch,
            "model": self.create_config().model,
        }
        handle = await ctx.run(t, key=ctx.key("gen"))
        result = await ctx.wait(handle)

        # Return as list (even though it's just one)
        return [result]
