"""
Two-stage propose → refine example.

Stage 1: generate a draft on a new branch.
Stage 2: refine using the Stage 1 branch as the base.

Usage (file path):
  pitaya "refactor module" --strategy ./examples/propose_refine.py -S model=sonnet

Usage (module path):
  pitaya "refactor module" --strategy examples.propose_refine:ProposeRefineStrategy
"""

from dataclasses import dataclass
from typing import List

from src.orchestration.strategies.base import Strategy, StrategyConfig
from src.orchestration.strategy_context import StrategyContext
from src.shared import InstanceResult


@dataclass
class ProposeRefineConfig(StrategyConfig):
    refine_hint: str = "Improve code quality and tests."


class ProposeRefineStrategy(Strategy):
    NAME = "propose-refine"

    def get_config_class(self) -> type[StrategyConfig]:
        return ProposeRefineConfig

    async def execute(
        self, prompt: str, base_branch: str, ctx: StrategyContext
    ) -> List[InstanceResult]:
        cfg = self.create_config()

        # Stage 1: propose
        t1 = {"prompt": prompt, "base_branch": base_branch, "model": cfg.model}
        h1 = await ctx.run(t1, key=ctx.key("propose", "gen"))
        r1 = await ctx.wait(h1)

        # Stage 2: refine based on Stage 1 branch
        refine_prompt = f"Refine previous changes: {cfg.refine_hint}\n\nOriginal instruction: {prompt}"
        next_base = r1.branch_name or base_branch
        t2 = {"prompt": refine_prompt, "base_branch": next_base, "model": cfg.model}
        h2 = await ctx.run(t2, key=ctx.key("refine", "gen"))
        r2 = await ctx.wait(h2)

        return [r1, r2]


STRATEGY = ProposeRefineStrategy
