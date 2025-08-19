"""
Best-of-N strategy that generates multiple solutions and selects the best.

This is the workhorse strategy - leverages AI's variability as a strength
by running N instances in parallel and selecting the highest scoring result.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig
from .scoring import ScoringStrategy

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext

logger = logging.getLogger(__name__)


@dataclass
class BestOfNConfig(StrategyConfig):
    """Configuration for best-of-n strategy."""

    n: int = 5  # Number of solutions to generate
    scorer_prompt: str = (
        "Review this code implementation and provide a score from 1-10 based on quality, completeness, and correctness. Return the score as a JSON object with 'score' and 'feedback' fields."
    )
    # Optional reviewer model override; defaults to StrategyConfig.model
    scorer_model: str | None = None

    def validate(self) -> None:
        """Validate best-of-n configuration."""
        super().validate()
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.n > 100:
            raise ValueError("n must be at most 100 (for sanity)")


class BestOfNStrategy(Strategy):
    """
    Generate N solutions in parallel, score each, return the best.

    This strategy gracefully handles partial failures - if some
    instances fail, it selects from the successful ones. Only
    fails completely if all instances fail.
    """

    @property
    def name(self) -> str:
        return "best-of-n"

    def get_config_class(self) -> type[StrategyConfig]:
        return BestOfNConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """
        Generate N solutions, score them, return the best.

        Uses ScoringStrategy internally to evaluate each solution,
        demonstrating strategy composition as per spec.

        Args:
            prompt: Instruction for the AI agents
            base_branch: Starting branch
            ctx: Strategy context for spawning instances

        Returns:
            List containing the best scoring result
        """
        config = self.create_config()
        logger.info(f"BestOfNStrategy starting with n={config.n}")

        # Helper function to run a single scoring strategy
        async def run_scoring_strategy(index: int):
            # Create a new ScoringStrategy instance for each execution
            # This ensures no shared state between parallel executions
            scoring_strategy = ScoringStrategy()
            overrides = {
                "model": config.model,
                "timeout_seconds": config.timeout_seconds,
                "container_limits": config.container_limits,
                "scorer_prompt": config.scorer_prompt,
                # Optional per-phase model override for reviewer
                "scorer_model": config.scorer_model,
                # Ensure unique durable keys per candidate
                "key_prefix": f"cand{index}",
            }

            scoring_strategy.set_config_overrides(overrides)

            # Add metadata to track which instance this is
            # Tag keys for durable scheduling inside composed strategy
            results = await scoring_strategy.execute(prompt, base_branch, ctx)
            if results and len(results) > 0:
                if results[0].metrics is None:
                    results[0].metrics = {}
                results[0].metrics["bestofn_index"] = index + 1
                results[0].metrics["bestofn_total"] = config.n
            return results

        # Run N scoring strategies in parallel
        strategy_tasks = [run_scoring_strategy(i) for i in range(config.n)]

        # Execute all scoring strategies in parallel
        all_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        scored_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Strategy execution {i+1} failed: {result}")
                continue
            if isinstance(result, list) and result:
                # ScoringStrategy returns a list with one result
                scored_results.extend(result)

        if not scored_results:
            # All strategies failed - raise per spec (NoViableCandidates)
            logger.error(f"All {config.n} strategy executions failed")
            try:
                from ...exceptions import NoViableCandidates
                raise NoViableCandidates()
            except Exception:
                return []

        # Filter successful results with valid scores
        valid_results = []
        for result in scored_results:
            if (
                result.success
                and result.metrics
                and result.metrics.get("score") is not None
            ):
                score = result.metrics.get("score")
                # Validate score is numeric and in reasonable range
                if isinstance(score, (int, float)) and 0 <= score <= 100:
                    valid_results.append(result)
                else:
                    logger.warning(
                        f"Invalid score {score} for result {result.branch_name}"
                    )

        if not valid_results:
            # No successfully scored results
            successful = [r for r in scored_results if r.success]
            if successful:
                logger.warning("No scored results, returning first successful result")
                return [successful[0]]
            # All failed
            try:
                from ...exceptions import NoViableCandidates
                raise NoViableCandidates()
            except Exception:
                return [scored_results[0]] if scored_results else []

        # Find the best scoring result
        best_result = max(valid_results, key=lambda r: r.metrics.get("score", 0))

        # Mark it as the selected result
        if best_result.metrics is None:
            best_result.metrics = {}
        best_result.metrics["selected"] = True
        best_result.metrics["selection_reason"] = (
            f"Best score: {best_result.metrics.get('score', 'N/A')}"
        )
        best_result.metrics["total_candidates"] = config.n
        best_result.metrics["successful_candidates"] = len(valid_results)

        logger.info(
            f"Selected instance {best_result.metrics.get('bestofn_index', 'N/A')}/{config.n} with score {best_result.metrics.get('score', 'N/A')}"
        )

        return [best_result]
