"""
Best‑of‑N Strategy — parallel candidates with structured scoring and robust selection.

Runs N generation+scoring pipelines in parallel (via ScoringStrategy) and selects
the candidate with the highest score. Designed for production use: deterministic
keys, configurable scoring rubric, optional diversity hints, and clear metadata.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from ...exceptions import StrategyError
from .base import Strategy, StrategyConfig
from .scoring import ScoringStrategy

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext

logger = logging.getLogger(__name__)


@dataclass
class BestOfNConfig(StrategyConfig):
    """Configuration for best-of-n strategy."""

    n: int = 5  # number of candidates
    # Scoring controls (passed through to ScoringStrategy)
    scorer_model: str = ""  # default: generator model
    weight_correctness: float = 0.5
    weight_completeness: float = 0.3
    weight_quality: float = 0.2
    score_scale_max: int = 10
    fail_below_score: Optional[float] = None
    scorer_max_retries: int = 2
    read_only_scoring: bool = True
    max_turns: Optional[int] = None
    # Diversity: optional hints to encourage different approaches per candidate
    diversity_hints: List[str] = field(default_factory=list)
    # Selection behavior
    tie_breaker: str = "first"  # "first" | "random"
    require_min_success: int = 1

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.n > 50:
            raise ValueError("n must be at most 50 (safety)")
        if self.score_scale_max not in (10, 100):
            raise ValueError("score_scale_max must be 10 or 100")
        for name in ("weight_correctness", "weight_completeness", "weight_quality"):
            v = getattr(self, name)
            try:
                if float(v) < 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"{name} must be a non‑negative number")
        if self.scorer_max_retries < 0:
            raise ValueError("scorer_max_retries must be >= 0")
        if self.fail_below_score is not None and self.fail_below_score < 0:
            raise ValueError("fail_below_score must be >= 0 when provided")
        if self.tie_breaker not in ("first", "random"):
            raise ValueError("tie_breaker must be 'first' or 'random'")


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
        if not (prompt or "").strip():
            raise StrategyError("best-of-n: prompt is required (non-empty)")
        cfg: BestOfNConfig = self.create_config()  # type: ignore
        logger.info(f"BestOfNStrategy starting with n={cfg.n}")

        async def run_candidate(idx: int) -> List[InstanceResult]:
            # Prepare candidate-specific prompt (diversity hint if provided)
            c_prompt = prompt
            hint = None
            if 0 <= idx < len(cfg.diversity_hints):
                hint = str(cfg.diversity_hints[idx]).strip()
                if hint:
                    c_prompt = f"{prompt}\n\n[Diversity hint #{idx+1}] Consider this approach: {hint}"

            scoring = ScoringStrategy()
            overrides: Dict[str, Any] = {
                "model": cfg.model,
                "scorer_model": cfg.scorer_model,
                "weight_correctness": cfg.weight_correctness,
                "weight_completeness": cfg.weight_completeness,
                "weight_quality": cfg.weight_quality,
                "score_scale_max": cfg.score_scale_max,
                "fail_below_score": cfg.fail_below_score,
                "scorer_max_retries": cfg.scorer_max_retries,
                "read_only_scoring": cfg.read_only_scoring,
                "max_turns": cfg.max_turns,
                # Unique durable key namespace per candidate
                "key_prefix": f"cand{idx+1}",
            }
            scoring.set_config_overrides(overrides)
            res = await scoring.execute(c_prompt, base_branch, ctx)
            # Annotate candidate metadata
            if res and len(res) > 0:
                r0 = res[0]
                if r0.metrics is None:
                    r0.metrics = {}
                r0.metrics["bestofn_index"] = idx + 1
                r0.metrics["bestofn_total"] = cfg.n
                if hint:
                    r0.metadata = r0.metadata or {}
                    r0.metadata["diversity_hint"] = hint
            return res

        tasks = [run_candidate(i) for i in range(cfg.n)]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter
        candidates: List[InstanceResult] = []
        for i, r in enumerate(results_list):
            if isinstance(r, Exception):
                logger.error(f"candidate {i+1} failed: {r}")
                continue
            if isinstance(r, list) and r:
                candidates.extend(r)

        if not candidates:
            logger.error(f"All {cfg.n} candidates failed")
            try:
                from ...exceptions import NoViableCandidates

                raise NoViableCandidates()
            except Exception:
                return []

        # Successful and scored
        valid: List[Tuple[float, InstanceResult]] = []
        for r in candidates:
            if not r.success:
                continue
            s = (r.metrics or {}).get("score") if r.metrics is not None else None
            if isinstance(s, (int, float)):
                # Accept either 10 or 100 scale; normalize to 100 for comparison
                scale = (r.metrics or {}).get("score_scale_max") or cfg.score_scale_max
                try:
                    scale = int(scale)
                except Exception:
                    scale = cfg.score_scale_max
                norm = float(s) * (100.0 / float(scale or 100))
                valid.append((norm, r))

        # Enforce minimum successes when configured
        successful = [r for r in candidates if r.success]
        min_required = max(1, int(cfg.require_min_success))
        if len(successful) < min_required:
            logger.error(
                f"successful candidates {len(successful)} < require_min_success={cfg.require_min_success}"
            )
            try:
                from ...exceptions import NoViableCandidates

                raise NoViableCandidates()
            except Exception:
                return successful if successful else []

        if not valid:
            if successful:
                logger.warning(
                    "No scored candidates; returning first successful result"
                )
                r0 = successful[0]
                r0.metrics = r0.metrics or {}
                r0.metrics["selected"] = True
                r0.metrics["selection_reason"] = "first_success_no_scores"
                r0.metrics["total_candidates"] = cfg.n
                r0.metrics["successful_candidates"] = len(successful)
                return [r0]
            # All failed
            try:
                from ...exceptions import NoViableCandidates

                raise NoViableCandidates()
            except Exception:
                return [candidates[0]] if candidates else []

        # Choose best with tie‑breaker
        best_score = max(v for v, _ in valid)
        top = [r for v, r in valid if abs(v - best_score) < 1e-9]
        selected: InstanceResult
        tie = False
        if len(top) == 1 or cfg.tie_breaker == "first":
            selected = top[0]
            tie = len(top) > 1
        else:
            # Deterministic random using strategy context RNG
            # Emit strategy.rand events for observability
            import math

            idx = int(math.floor(ctx.rand() * len(top)))
            if idx >= len(top):
                idx = 0
            selected = top[idx]
            tie = len(top) > 1

        # Annotate selection metadata
        sel_metrics = selected.metrics or {}
        sel_metrics["selected"] = True
        sel_metrics["selection_reason"] = (
            "highest_score_tie_rand"
            if tie and cfg.tie_breaker == "random"
            else "highest_score"
        )
        sel_metrics["total_candidates"] = cfg.n
        sel_metrics["successful_candidates"] = len(successful)
        selected.metrics = sel_metrics

        logger.info(
            f"Selected candidate index={sel_metrics.get('bestofn_index','?')}/{cfg.n} score={sel_metrics.get('score','?')}"
        )

        return [selected]
