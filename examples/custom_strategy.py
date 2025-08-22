"""
Example custom strategy for Pitaya.

Usage:
  pitaya "hello world" --strategy ./examples/custom_strategy.py -S model=sonnet

You can also specify a class explicitly:
  pitaya "hello" --strategy ./examples/custom_strategy.py:MyCustomStrategy
"""

from dataclasses import dataclass
from src.orchestration.strategies.base import Strategy, StrategyConfig


@dataclass
class MyConfig(StrategyConfig):
    # Example of adding your own knobs; available via -S greeting=...
    greeting: str = "Hello from custom strategy!"


class MyCustomStrategy(Strategy):
    # Use class attribute NAME for display/events label
    NAME = "custom-example"

    def get_config_class(self) -> type[StrategyConfig]:
        return MyConfig

    async def execute(self, prompt: str, base_branch: str, ctx):
        cfg = self.create_config()
        # Built-in logger available via base class
        self.logger.info("Custom strategy starting: %s", cfg.greeting)

        # Durable task with a deterministic key
        task = {
            "prompt": prompt,
            "base_branch": base_branch,
            "model": cfg.model,
            # You can pass runner/orchestrator hints here if needed
            # e.g., "import_policy": "always"
        }
        handle = await ctx.run(task, key=ctx.key("custom", "gen"))
        result = await ctx.wait(handle)
        return [result]


# Allow using the file without specifying :ClassName
STRATEGY = MyCustomStrategy
