"""
Fan-out strategy example that runs two instances in parallel.

Usage (file path):
  pitaya "implement feature" --strategy ./examples/fanout_two.py -S model=sonnet

Usage (module path):
  pitaya "implement feature" --strategy examples.fanout_two:FanOutTwoStrategy
"""

import asyncio
from dataclasses import dataclass
from typing import List

from src.orchestration.strategies.base import Strategy, StrategyConfig
from src.orchestration.strategy_context import StrategyContext
from src.shared import InstanceResult


@dataclass
class FanOutConfig(StrategyConfig):
    # Optional knob to let each branch use same/different models in the future
    parallel: int = 2


class FanOutTwoStrategy(Strategy):
    NAME = "fanout-two"

    def get_config_class(self) -> type[StrategyConfig]:
        return FanOutConfig

    async def execute(
        self, prompt: str, base_branch: str, ctx: StrategyContext
    ) -> List[InstanceResult]:
        cfg = self.create_config()
        n = max(2, int(getattr(cfg, "parallel", 2)))

        # Schedule N tasks with durable keys
        handles = []
        for i in range(1, n + 1):
            t = {"prompt": prompt, "base_branch": base_branch, "model": cfg.model}
            h = await ctx.run(t, key=ctx.key("gen", i))
            handles.append(h)

        # Wait for all in parallel
        results = await asyncio.gather(*(ctx.wait(h) for h in handles))
        return list(results)


STRATEGY = FanOutTwoStrategy
