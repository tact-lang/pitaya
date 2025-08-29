"""
Iterative Strategy — generate → review → refine for a fixed number of rounds.

Production‑ready refinement loop that:
- Generates an initial solution on a branch
- Gathers structured review feedback in read‑only mode
- Refines based on that feedback, continuing on the same evolving branch

Each iteration uses durable keys and aims for a single commit (agent‑driven) per
iteration to keep history concise. The review task does not modify files.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...shared import InstanceResult
from ...exceptions import StrategyError
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class IterativeConfig(StrategyConfig):
    """Configuration for iterative refinement strategy."""

    iterations: int = 3  # number of iteration rounds
    reviewer_model: str = ""  # defaults to generator model
    review_max_retries: int = 1  # additional attempts after first (total = +1)
    read_only_review: bool = True  # import_policy=never for reviewers
    max_turns: Optional[int] = None  # optional runner hint
    stop_on_no_changes: bool = True  # stop early if an iteration yields no changes
    key_prefix: str = ""  # namespace for durable keys when composed

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.iterations > 10:
            raise ValueError("iterations must be at most 10 (safety)")
        if self.review_max_retries < 0:
            raise ValueError("review_max_retries must be >= 0")


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
        """Generate solution and iteratively refine it based on review feedback."""
        if not (prompt or "").strip():
            raise StrategyError("iterative: prompt is required (non-empty)")
        cfg: IterativeConfig = self.create_config()  # type: ignore

        current_branch = base_branch
        last_result: Optional[InstanceResult] = None
        last_feedback: Optional[str] = None
        prefix = [cfg.key_prefix] if cfg.key_prefix else []

        for i in range(1, int(cfg.iterations) + 1):
            # Build iteration prompt
            iter_prompt = _build_iteration_prompt(
                original_task=prompt, previous_feedback=last_feedback, iteration=i
            )
            iter_task: Dict[str, Any] = {
                "prompt": iter_prompt,
                "base_branch": current_branch,
                "model": cfg.model,
            }
            iter_key = ctx.key(*prefix, "iter", i)
            try:
                h = await ctx.run(iter_task, key=iter_key)
                res = await ctx.wait(h)
            except Exception as e:
                from ...shared import InstanceResult as _IR

                err_type = getattr(e, "error_type", "unknown")
                msg = getattr(e, "message", str(e))
                res = _IR(
                    success=False, error=msg, error_type=err_type, status="failed"
                )

            if not res.success:
                # Abort loop; return last success if available
                return [last_result] if last_result else [res]

            # Early stop if no changes and configured
            if cfg.stop_on_no_changes and not getattr(res, "has_changes", False):
                res.metrics = res.metrics or {}
                res.metrics["iterative_early_stop"] = "no_changes"
                last_result = res
                break

            # Update branch and last result
            last_result = res
            current_branch = res.branch_name or current_branch

            # If not the final iteration, gather review feedback with retries
            if i < cfg.iterations:
                reviewer_model = (cfg.reviewer_model or cfg.model or "sonnet").strip()
                attempts = int(cfg.review_max_retries) + 1
                feedback_text: Optional[str] = None
                for j in range(1, attempts + 1):
                    corrective = j > 1
                    review_prompt = _build_reviewer_prompt(
                        original_task=prompt,
                        iteration=i,
                        corrective=corrective,
                    )
                    r_key = ctx.key(*prefix, "review", i, f"attempt-{j}")
                    r_task: Dict[str, Any] = {
                        "prompt": review_prompt,
                        "base_branch": current_branch,
                        "model": reviewer_model,
                        "import_policy": "never" if cfg.read_only_review else "auto",
                        "skip_empty_import": True,
                    }
                    if cfg.max_turns is not None:
                        r_task["max_turns"] = int(cfg.max_turns)

                    try:
                        rh = await ctx.run(r_task, key=r_key)
                        rres = await ctx.wait(rh)
                    except Exception:
                        rres = None
                    if rres and rres.success and rres.final_message:
                        feedback_text = rres.final_message
                        break
                # Fallback generic feedback if none available
                last_feedback = (
                    feedback_text
                    or "Please improve error handling, edge cases, tests, and code organization based on the task goals."
                )

        # Finalize metrics
        if last_result:
            last_result.metrics = last_result.metrics or {}
            last_result.metrics.update(
                {
                    "iterations_completed": i if "i" in locals() else cfg.iterations,
                    "refinement_strategy": "iterative",
                }
            )

        return [last_result] if last_result else []


# ---------------------------
# Prompt helpers
# ---------------------------


def _build_iteration_prompt(
    *, original_task: str, previous_feedback: Optional[str], iteration: int
) -> str:
    fb = previous_feedback.strip() if previous_feedback else ""
    fb_block = f"\n<previous_feedback>\n{fb}\n</previous_feedback>\n" if fb else ""
    return (
        f"<role>\n"
        f"You are implementing or refining code to achieve the task below. "
        f"This is iteration #{iteration}. Incorporate prior review feedback when provided.\n"
        f"</role>\n\n"
        f"<task>\n{original_task}\n</task>\n\n"
        f"<guidance>\n"
        f"- Focus on correctness first; ensure tests (if present) pass.\n"
        f"- Address edge cases and error handling.\n"
        f"- Keep changes minimal and well-structured.\n"
        f"- Update or add tests/docs when necessary.\n"
        f"</guidance>\n"
        f"<constraints>\n"
        f"- Commit exactly once at the end of this iteration after verifying changes.\n"
        f"- Keep commits clean and messages concise (imperative mood).\n"
        f"</constraints>\n" + fb_block
    )


def _build_reviewer_prompt(
    *, original_task: str, iteration: int, corrective: bool
) -> str:
    return (
        f"<role>\n"
        f"You are reviewing the current implementation produced for the task below. "
        f"Provide precise, actionable feedback to guide the next refinement iteration.\n"
        f"</role>\n\n"
        f"<task>\n{original_task}\n</task>\n\n"
        f"<focus>\n"
        f"- Functional correctness and observable behavior\n"
        f"- Completeness w.r.t. user flows and edge cases\n"
        f"- Code quality: readability, decomposition, tests\n"
        f"</focus>\n\n"
        f"<constraints>\n"
        f"- Do not modify files; this is a review only.\n"
        f"- Keep guidance concise and practical; list concrete actions.\n"
        f"</constraints>\n\n"
        f"<output>\n"
        f"Provide a short numbered list (1–7 items) of improvements.\n"
        f"Each item should contain: issue, why it matters, and the concrete fix.\n"
        f"</output>\n\n"
        + (
            "<retry_guidance>\nYour previous feedback was missing clear, numbered action items. Provide concise, practical steps.\n</retry_guidance>\n"
            if corrective
            else ""
        )
    )
