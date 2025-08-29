"""
Scoring Strategy — generate, then evaluate with a structured rubric.

Production-ready, plugin‑agnostic two‑phase flow:
1) Generate a solution on a branch from the provided base.
2) Run a read‑only scoring task on that branch that outputs strictly formatted JSON
   containing a numeric score and concise rationale. The score and feedback are
   attached to the generation result's metadata and metrics.

Intended for composition (e.g., Best‑of‑N) or standalone evaluation runs.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from ...exceptions import StrategyError
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class ScoringConfig(StrategyConfig):
    """Configuration for scoring strategy."""

    # Scoring model (defaults to the generator model when empty)
    scorer_model: str = ""
    # Scoring rubric weights (sum does not need to be 1; normalized at runtime)
    weight_correctness: float = 0.5
    weight_completeness: float = 0.3
    weight_quality: float = 0.2  # code readability, tests, structure
    # Score range and gating
    score_scale_max: int = 10  # e.g., 10 or 100
    fail_below_score: Optional[float] = (
        None  # if set, mark generation failed when below
    )
    # Attempts
    scorer_max_retries: int = 2  # additional attempts after initial (total = retries+1)
    # Controls
    read_only_scoring: bool = True  # import_policy=never to prevent commits
    max_turns: Optional[int] = None  # optional hint for runner
    # Optional key prefix for durable keys when composed
    key_prefix: str = ""

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
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
        """Generate a solution and evaluate it with a structured rubric.

        Returns a single InstanceResult with score metadata attached. Optionally
        marks the result failed when below `fail_below_score`.
        """
        if not (prompt or "").strip():
            raise StrategyError("scoring: prompt is required (non-empty)")
        cfg: ScoringConfig = self.create_config()  # type: ignore

        # Phase 1 — Generate
        prefix = [cfg.key_prefix] if cfg.key_prefix else []
        gen_key = ctx.key(*prefix, "gen")
        gen_task: Dict[str, Any] = {
            "prompt": prompt,
            "base_branch": base_branch,
            "model": cfg.model,
        }
        gen_handle = await ctx.run(gen_task, key=gen_key)
        try:
            gen_result = await ctx.wait(gen_handle)
        except Exception as e:
            from ...shared import InstanceResult as _IR

            err_type = getattr(e, "error_type", "unknown")
            msg = getattr(e, "message", str(e))
            gen_result = _IR(
                success=False, error=msg, error_type=err_type, status="failed"
            )

        if not gen_result.success:
            return [gen_result]

        # Phase 2 — Score with retries
        effective_base = gen_result.branch_name or base_branch
        scorer_model = (cfg.scorer_model or cfg.model or "sonnet").strip()
        attempts = int(cfg.scorer_max_retries) + 1
        last_score: Optional[float] = None
        last_feedback: Optional[str] = None
        scoring_success = False
        scoring_branch: Optional[str] = None

        for i in range(attempts):
            corrective = i > 0
            s_key = ctx.key(
                *prefix,
                "score",
                getattr(gen_handle, "instance_id", "gen"),
                f"attempt-{i+1}",
            )
            s_prompt = _build_scoring_prompt(
                original_task=prompt,
                score_scale_max=cfg.score_scale_max,
                weights=(
                    cfg.weight_correctness,
                    cfg.weight_completeness,
                    cfg.weight_quality,
                ),
                corrective=corrective,
            )
            s_task: Dict[str, Any] = {
                "prompt": s_prompt,
                "base_branch": effective_base,
                "model": scorer_model,
                "import_policy": "never" if cfg.read_only_scoring else "auto",
                "skip_empty_import": True,
            }
            if cfg.max_turns is not None:
                s_task["max_turns"] = int(cfg.max_turns)

            try:
                s_handle = await ctx.run(s_task, key=s_key)
                s_res = await ctx.wait(s_handle)
            except Exception:
                # Keep trying subsequent attempts
                continue

            if s_res and s_res.success:
                s, fb = _parse_score_and_feedback(
                    s_res.final_message or "",
                    max_scale=cfg.score_scale_max,
                )
                scoring_branch = s_res.branch_name
                if s is not None:
                    last_score, last_feedback = s, fb
                    scoring_success = True
                    break
                else:
                    last_feedback = fb or s_res.final_message or last_feedback
                    # retry to obtain structured score
                    continue
            else:
                # retry on failed scoring attempt
                continue

        # Finalize — attach metadata and optional gating
        gen_result.metadata = gen_result.metadata or {}
        gen_result.metadata.update(
            {
                "score": last_score,
                "feedback": last_feedback,
                "score_scale_max": cfg.score_scale_max,
                "scorer_model": scorer_model,
                "scorer_attempts": attempts,
                "scorer_success": scoring_success,
                "scorer_branch": scoring_branch,
                "weights": {
                    "correctness": cfg.weight_correctness,
                    "completeness": cfg.weight_completeness,
                    "quality": cfg.weight_quality,
                },
            }
        )
        if gen_result.metrics is None:
            gen_result.metrics = {}
        gen_result.metrics.update(
            {
                "score": last_score,
                "feedback": last_feedback,
                "score_scale_max": cfg.score_scale_max,
            }
        )

        # Optional gating: mark failed if below threshold
        if (
            cfg.fail_below_score is not None
            and last_score is not None
            and float(last_score) < float(cfg.fail_below_score)
        ):
            gen_result.success = False
            gen_result.status = "failed"
            gen_result.error_type = "score_below_threshold"
            gen_result.error = f"score {last_score} < threshold {cfg.fail_below_score}"

        return [gen_result]


# ---------------------------
# Prompt + parsing helpers
# ---------------------------


def _build_scoring_prompt(
    *,
    original_task: str,
    score_scale_max: int,
    weights: Tuple[float, float, float],
    corrective: bool,
) -> str:
    w_correctness, w_completeness, w_quality = weights

    return (
        f"<role>\n"
        f"You are a rigorous code evaluator. Read the repository on the current branch and evaluate the work for the original task below.\n"
        f"Return a strictly formatted JSON object with an overall numeric score and rationale. No extra text outside JSON.\n"
        f"</role>\n\n"
        f"<task>\n{original_task}\n</task>\n\n"
        f"<rubric>\n"
        f"Scoring scale: 0–{score_scale_max}. Weights (normalized):\n"
        f"- correctness: {w_correctness}\n"
        f"- completeness: {w_completeness}\n"
        f"- quality: {w_quality}\n"
        f"Interpretation:\n"
        f"- correctness: functional behavior, passing tests, absence of obvious bugs.\n"
        f"- completeness: task coverage, edge cases, docs/README updates if relevant.\n"
        f"- quality: readability, structure, idiomatic style, tests added/updated where appropriate.\n"
        f"</rubric>\n\n"
        f"<constraints>\n"
        f"- Do not modify files.\n"
        f"- Do not commit.\n"
        f"- If insufficient changes exist to evaluate, explain briefly and set score conservatively.\n"
        f"</constraints>\n\n"
        f"<output>\n"
        f"Return ONLY one JSON object with keys: score (number), reasoning (string), risks (string), suggestions (string).\n"
        f'Example: {{\n  "score": 7,\n  "reasoning": "...",\n  "risks": "...",\n  "suggestions": "..."\n}}\n'
        f"</output>\n\n"
        + (
            "<retry_guidance>\nYour previous output lacked a valid JSON object or score. Ensure the response is exactly one JSON object, no fences, no prose.\n</retry_guidance>\n"
            if corrective
            else ""
        )
    )


def _parse_score_and_feedback(
    text: str, *, max_scale: int
) -> Tuple[Optional[float], Optional[str]]:
    """Extract score and feedback from text containing a JSON block.

    Strategy: scan for JSON-looking snippets and return the first that contains
    a numeric 'score'. Accept ints/floats; clamp to [0, max_scale]. The feedback is
    the remaining fields rendered as a compact summary when possible.
    """
    if not text:
        return None, None
    # Fast path: try to parse the entire string
    candidates: List[str] = []
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        candidates.append(s)
    # Extract JSON code blocks or inline objects
    import re

    # Code block with json
    for m in re.finditer(r"```(?:json)?\n(\{[\s\S]*?\})\n```", text, re.IGNORECASE):
        candidates.append(m.group(1))
    # Inline object (best‑effort)
    for m in re.finditer(r"(\{[^{}]*\})", text):
        if '"score"' in m.group(1):
            candidates.append(m.group(1))

    def _try_parse(js: str) -> Tuple[Optional[float], Optional[str]]:
        try:
            data = json.loads(js)
            if not isinstance(data, dict):
                return None, None
            raw = data.get("score")
            if raw is None:
                return None, None
            try:
                val = float(raw)
            except Exception:
                return None, None
            # clamp
            if val < 0:
                val = 0.0
            if max_scale > 0 and val > max_scale:
                val = float(max_scale)
            # Build feedback summary from remaining keys when present
            fb_parts = []
            for k in ("reasoning", "risks", "suggestions", "feedback"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    fb_parts.append(f"{k}: {v.strip()}")
            feedback = " | ".join(fb_parts) if fb_parts else None
            return val, feedback
        except Exception:
            return None, None

    for js in candidates:
        score, fb = _try_parse(js)
        if score is not None:
            return score, fb
    return None, None
