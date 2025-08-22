"""
Iterative refinement strategy that improves solutions through feedback loops.

This strategy generates an initial solution, gets feedback, then iteratively
improves it. Each iteration continues in the same agent session for context.
"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class IterativeConfig(StrategyConfig):
    """Configuration for iterative refinement strategy."""

    iterations: int = 3  # Number of refinement iterations
    reviewer_prompt: str = (
        "Review this code and provide specific, actionable feedback for improvement. Focus on bugs, edge cases, performance, and code quality."
    )

    def validate(self) -> None:
        """Validate iterative configuration."""
        super().validate()
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.iterations > 10:
            raise ValueError("iterations must be at most 10 (for sanity)")


class IterativeStrategy(Strategy):
    """
    Iteratively refine a solution based on feedback.

    Each iteration runs in the same agent session, allowing
    the AI to build on previous context and feedback.
    """

    @property
    def name(self) -> str:
        return "iterative"

    def get_config_class(self) -> type[StrategyConfig]:
        return IterativeConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """
        Generate solution and iteratively refine it.

        Args:
            prompt: Initial instruction for the AI agent
            base_branch: Starting branch
            ctx: Strategy context for spawning instances

        Returns:
            List containing the final refined result
        """
        config = self.create_config()

        # Start with the initial implementation
        current_branch = base_branch
        _ = None  # placeholder for session id continuity
        final_result = None
        feedback = ""

        for iteration in range(config.iterations):
            if iteration == 0:
                # First iteration: implement from scratch
                iteration_prompt = prompt
            else:
                # Subsequent iterations: refine based on feedback
                iteration_prompt = (
                    f"Based on the previous feedback, please improve the implementation. "
                    f"Address the issues mentioned and enhance the code quality.\n\n"
                    f"Previous feedback:\n{feedback}"
                )

            # Durable run with key and continuation
            task = {"prompt": iteration_prompt, "base_branch": current_branch}
            handle = await ctx.run(task, key=ctx.key("iter", iteration + 1))
            try:
                result = await ctx.wait(handle)
            except Exception as e:
                from ...shared import InstanceResult as _IR

                err_type = getattr(e, "error_type", "unknown")
                msg = getattr(e, "message", str(e))
                result = _IR(
                    success=False, error=msg, error_type=err_type, status="failed"
                )

            # If any iteration fails, stop and return what we have
            if not result.success:
                if final_result:
                    return [final_result]  # Return last successful iteration
                return [result]  # Return the failure if first iteration

            # Update for next iteration
            final_result = result
            current_branch = result.branch_name
            _ = result.session_id  # Continue in same session (kept implicitly)

            # Get feedback for next iteration (if not last)
            if iteration < config.iterations - 1:
                # Spawn a reviewer instance
                review_task = {
                    "prompt": config.reviewer_prompt,
                    "base_branch": current_branch,
                }
                review_handle = await ctx.run(
                    review_task, key=ctx.key("review", iteration + 1)
                )
                try:
                    review_result = await ctx.wait(review_handle)
                except Exception as e:
                    from ...shared import InstanceResult as _IR

                    err_type = getattr(e, "error_type", "unknown")
                    msg = getattr(e, "message", str(e))
                    review_result = _IR(
                        success=False, error=msg, error_type=err_type, status="failed"
                    )

                if review_result.success and review_result.final_message:
                    feedback = review_result.final_message
                else:
                    # If review fails, use generic feedback
                    feedback = "Please review the code for any improvements in error handling, edge cases, and code organization."

        # Mark final result with iteration count
        if final_result and final_result.metrics is None:
            final_result.metrics = {}

        if final_result:
            final_result.metrics.update(
                {
                    "iterations_completed": iteration + 1,
                    "refinement_strategy": "iterative",
                }
            )

        return [final_result] if final_result else []
