"""
Scoring strategy that generates a solution then evaluates it.

This two-phase pattern is primarily useful as a building block
for other strategies like BestOfN.
"""

import json
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig
from ...exceptions import StrategyError

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class ScoringConfig(StrategyConfig):
    """Configuration for scoring strategy."""

    scorer_prompt: str = (
        "Review this code implementation and provide a score from 1-10 based on quality, completeness, and correctness. Return the score as a JSON object with 'score' and 'feedback' fields."
    )
    # Optional key prefix for durable task keys when composed by other strategies
    key_prefix: str = ""

    def validate(self) -> None:
        """Validate scoring configuration."""
        super().validate()
        if not self.scorer_prompt:
            raise ValueError("scorer_prompt is required")


class ScoringStrategy(Strategy):
    """
    Two-phase strategy: generate solution, then score it.

    First runs an instance with the main prompt, then runs a
    reviewer instance to evaluate the result. Attaches score
    and feedback to the result metadata.
    """

    @property
    def name(self) -> str:
        return "scoring"

    def get_config_class(self) -> type[StrategyConfig]:
        return ScoringConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """
        Generate a solution and score it.

        Args:
            prompt: Instruction for the AI agent
            base_branch: Starting branch
            ctx: Strategy context for spawning instances

        Returns:
            List containing the scored instance result
        """
        config = self.create_config()

        # Phase 1: Generate solution
        cfg = self.create_config()
        prefix = [cfg.key_prefix] if cfg.key_prefix else []

        gen_task = {"prompt": prompt, "base_branch": base_branch, "model": cfg.model}
        gen_key = ctx.key(*prefix, "gen")
        generation_handle = await ctx.run(gen_task, key=gen_key)
        generation_result = await ctx.wait(generation_handle)

        # If generation failed, the strategy fails
        if not generation_result.success:
            return [generation_result]

        # Phase 2: Score the solution
        # Reviewer starts from the generated branch to review the solution
        # Include the original task in the review prompt
        review_prompt = f"""{config.scorer_prompt}

ORIGINAL TASK: {prompt}"""

        scoring_task = {
            "prompt": review_prompt,
            "base_branch": generation_result.branch_name,
            # Review/scoring tasks should not create branches; run read-only when configured
            "import_policy": "never",
        }
        score_key = ctx.key(*prefix, "score", generation_handle.instance_id, "attempt-1")
        scoring_handle = await ctx.run(scoring_task, key=score_key)
        scoring_result = await ctx.wait(scoring_handle)

        # Extract score from the reviewer's output
        score = None
        feedback = None

        if scoring_result.success and scoring_result.final_message:
            try:
                # Try to parse score from the reviewer's message
                # Look for JSON in the output
                import re

                json_match = re.search(
                    r'\{[^}]*"score"[^}]*\}', scoring_result.final_message
                )
                if json_match:
                    score_data = json.loads(json_match.group())
                    score = score_data.get("score")
                    feedback = score_data.get("feedback", "")
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, try to extract score directly
                score_match = re.search(
                    r"score[:\s]+(\d+)", scoring_result.final_message, re.IGNORECASE
                )
                if score_match:
                    score = int(score_match.group(1))
                feedback = scoring_result.final_message

        # If scoring failed, fail the strategy per spec
        if not scoring_result.success:
            raise StrategyError("Scoring phase failed")

        # Attach score/feedback to result metadata per spec (also keep in metrics for convenience)
        generation_result.metadata = generation_result.metadata or {}
        generation_result.metadata.update(
            {
                "score": score,
                "feedback": feedback,
                "scorer_branch": scoring_result.branch_name,
                "scorer_success": True,
            }
        )
        if generation_result.metrics is None:
            generation_result.metrics = {}
        generation_result.metrics.update({"score": score, "feedback": feedback})

        return [generation_result]
