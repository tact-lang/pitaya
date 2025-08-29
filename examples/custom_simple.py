"""
Simple custom strategy example for Pitaya.

Usage (file path):
  pitaya "hello world" --strategy ./examples/custom_simple.py -S model=sonnet

Usage (module path):
  pitaya "hello" --strategy examples.custom_simple:MyCustomStrategy
"""

from dataclasses import dataclass
from typing import List

from src.orchestration.strategies.base import Strategy, StrategyConfig
from src.orchestration.strategy_context import StrategyContext
from src.shared import InstanceResult


@dataclass
class MyConfig(StrategyConfig):
    greeting: str = "Hello from custom strategy!"


class MyCustomStrategy(Strategy):
    NAME = "custom-simple"

    def get_config_class(self) -> type[StrategyConfig]:
        return MyConfig

    async def execute(
        self, prompt: str, base_branch: str, ctx: StrategyContext
    ) -> List[InstanceResult]:
        cfg = self.create_config()
        self.logger.info("Custom simple starting: %s", cfg.greeting)

        task = {
            "prompt": prompt,
            "base_branch": base_branch,
            "model": cfg.model,
        }
        handle = await ctx.run(task, key=ctx.key("custom", "gen"))
        result = await ctx.wait(handle)
        return [result]


# Allow using the file without specifying :ClassName
STRATEGY = MyCustomStrategy
