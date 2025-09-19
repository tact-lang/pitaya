"""
Pull Request Review Strategy (simple, clean, CI‑friendly).

Flow:
- N parallel reviewers (no file splitting). Each reviews the whole PR relative to
  a base branch and writes a Markdown report under reports/pr-review/raw/.
- One validator per reviewer refines and sanity‑checks the report in place.
- A final composer aggregates validated reports into a single summary Markdown.

CI behavior:
- Parse validator reports for a compact JSON summary (verdict + severity counts).
- If any validator returns a severity in fail_on (e.g., BLOCKER/HIGH), the
  strategy marks the final result as failed so headless mode exits non‑zero.

Configuration aims to stay minimal and predictable.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class PRReviewConfig(StrategyConfig):
    """Configuration for PR review strategy."""

    reviewers: int = 3
    reviewer_max_retries: int = 0
    validator_max_retries: int = 1
    report_dir: str = "reports/pr-review"
    base_branch: str = "origin/main"
    fail_on: List[str] = field(
        default_factory=lambda: ["BLOCKER", "HIGH"]
    )  # severities
    composer_model: Optional[str] = None  # default to `model` if None
    # Optional custom instruction hooks. Either provide long text directly or via *_path.
    review_instructions: Optional[str] = None
    review_instructions_path: Optional[str] = None
    validator_instructions: Optional[str] = None
    validator_instructions_path: Optional[str] = None
    composer_instructions: Optional[str] = None
    composer_instructions_path: Optional[str] = None

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.reviewers < 1:
            raise ValueError("reviewers must be at least 1")
        if self.reviewer_max_retries < 0 or self.validator_max_retries < 0:
            raise ValueError("max retries must be >= 0")


class PRReviewStrategy(Strategy):
    @property
    def name(self) -> str:
        return "pr-review"

    def get_config_class(self) -> type[StrategyConfig]:
        return PRReviewConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        cfg: PRReviewConfig = self.create_config()  # type: ignore
        repo_path: Path = getattr(ctx._orchestrator, "repo_path", Path.cwd())  # noqa

        # Ensure report directories exist in the repo (helps agents discover paths)
        try:
            (repo_path / cfg.report_dir / "raw").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Resolve optional long-form instruction text (file overrides inline)
        review_extra = _resolve_instructions(
            cfg.review_instructions, cfg.review_instructions_path, default=""
        )
        validator_extra = _resolve_instructions(
            cfg.validator_instructions, cfg.validator_instructions_path, default=""
        )
        composer_extra = _resolve_instructions(
            cfg.composer_instructions, cfg.composer_instructions_path, default=""
        )

        async def _run_reviewer_with_retries(idx: int) -> Optional[InstanceResult]:
            rel_path = f"{cfg.report_dir}/raw/review_r{idx}.md"
            attempts = cfg.reviewer_max_retries + 1
            last: Optional[InstanceResult] = None
            for attempt in range(attempts):
                key = ctx.key(
                    "r", idx, "review", f"attempt-{attempt}" if attempt else "initial"
                )
                task = {
                    "prompt": _build_reviewer_prompt(
                        base_branch=(cfg.base_branch or base_branch),
                        report_path=rel_path,
                        extra_instructions=review_extra,
                    ),
                    "base_branch": base_branch,
                    "model": cfg.model,
                }
                try:
                    h = await ctx.run(task, key=key)
                    res = await ctx.wait(h)
                except Exception:
                    res = None
                if res:
                    last = res
                    if res.success and _git_show_ok(
                        repo_path, res.branch_name, rel_path
                    ):
                        return res
            return last if (last and last.success) else None

        async def _run_validator_with_retries(
            idx: int, reviewer_branch: str
        ) -> Optional[InstanceResult]:
            rel_path = f"{cfg.report_dir}/raw/review_r{idx}.md"
            attempts = cfg.validator_max_retries + 1
            last: Optional[InstanceResult] = None
            for attempt in range(attempts):
                key = ctx.key(
                    "r", idx, "validate", f"attempt-{attempt}" if attempt else "initial"
                )
                task = {
                    "prompt": _build_validator_prompt(
                        report_path=rel_path, extra_instructions=validator_extra
                    ),
                    "base_branch": reviewer_branch,
                    "model": cfg.model,
                }
                try:
                    h = await ctx.run(task, key=key)
                    res = await ctx.wait(h)
                except Exception:
                    res = None
                if res:
                    last = res
                    if res.success and _git_show_ok(
                        repo_path, res.branch_name, rel_path
                    ):
                        return res
            return last if (last and last.success) else None

        # Stage 1+2: reviewers and validators
        async def _review_then_validate(
            idx: int,
        ) -> Optional[Tuple[int, InstanceResult]]:
            rres = await _run_reviewer_with_retries(idx)
            if not (rres and rres.success and rres.branch_name):
                return None
            vres = await _run_validator_with_retries(idx, rres.branch_name)
            if not (vres and vres.success and vres.branch_name):
                return None
            return (idx, vres)

        rv_tasks = [
            asyncio.create_task(_review_then_validate(i))
            for i in range(1, int(cfg.reviewers) + 1)
        ]
        validated: List[Tuple[int, InstanceResult]] = []
        done = await asyncio.gather(*rv_tasks, return_exceptions=True)
        for item in done:
            if isinstance(item, tuple) and len(item) == 2:
                validated.append(item)

        # Stage 3: composer
        summary_rel = f"{cfg.report_dir}/summary.md"
        composer_model = cfg.composer_model or cfg.model
        key_comp = ctx.key("compose", "final")
        comp_task = {
            "prompt": _build_composer_prompt(
                base_branch=(cfg.base_branch or base_branch),
                report_paths=[
                    f"{cfg.report_dir}/raw/review_r{i}.md" for i, _ in validated
                ],
                output_path=summary_rel,
                extra_instructions=composer_extra,
            ),
            "base_branch": base_branch,
            "model": composer_model,
        }

        comp_handle = await ctx.run(comp_task, key=key_comp)
        comp_res = await ctx.wait(comp_handle)

        # Evaluate severities from validator outputs (prefer JSON trailer if present)
        should_fail = False
        agg_counts: Dict[str, int] = {
            "BLOCKER": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        }
        for i, vres in validated:
            if not vres.branch_name:
                continue
            text = _git_show_text(
                repo_path, vres.branch_name, f"{cfg.report_dir}/raw/review_r{i}.md"
            )
            verdict, counts = _extract_summary(text)
            if verdict and verdict.upper() in ("NEEDS_CHANGES", "FAIL"):
                should_fail = True
            for sev in cfg.fail_on or []:
                c = counts.get(sev.upper(), 0)
                if isinstance(c, (int, float)) and c > 0:
                    should_fail = True
            for k, v in counts.items():
                try:
                    agg_counts[k.upper()] = agg_counts.get(k.upper(), 0) + int(v)
                except Exception:
                    continue

        # If composer succeeded, annotate its result as the strategy result; flip success on fail conditions
        results: List[InstanceResult] = []
        if comp_res:
            try:
                # annotate metrics and metadata for publishers
                if comp_res.metrics is None:
                    comp_res.metrics = {}
                comp_res.metrics["pr_review_should_fail"] = bool(should_fail)
                comp_res.metrics["pr_review_verdict"] = (
                    "NEEDS_CHANGES" if should_fail else "PASS"
                )
                comp_res.metrics["pr_review_counts"] = agg_counts
                comp_res.metadata = comp_res.metadata or {}
                comp_res.metadata["pr_review"] = {
                    "role": "composer",
                    "summary_path": summary_rel,
                    "reviewers": int(cfg.reviewers),
                }
            except Exception:
                pass
            if should_fail:
                try:
                    comp_res.success = False
                    comp_res.status = "failed"
                except Exception:
                    pass
            results.append(comp_res)

        return results


def _git_show_ok(repo: Path, branch: Optional[str], rel_path: str) -> bool:
    if not branch:
        return False
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "show", f"{branch}:{rel_path}"],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and bool(r.stdout)
    except Exception:
        return False


def _git_show_text(repo: Path, branch: Optional[str], rel_path: str) -> str:
    if not branch:
        return ""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "show", f"{branch}:{rel_path}"],
            capture_output=True,
            text=True,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _extract_summary(md_text: str) -> Tuple[Optional[str], Dict[str, int]]:
    """Extract a JSON trailer from the validator report.

    Expect a fenced block at the end:

    ```json
    {"verdict":"NEEDS_CHANGES","counts":{"BLOCKER":1,"HIGH":0,"MEDIUM":2,"LOW":1,"INFO":0}}
    ```
    """
    verdict = None
    counts: Dict[str, int] = {}
    if not md_text:
        return verdict, counts
    # Try JSON fenced code block
    m = re.search(
        r"```json\s*(\{[\s\S]*?\})\s*```\s*$",
        md_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                v = obj.get("verdict")
                if isinstance(v, str):
                    verdict = v
                c = obj.get("counts")
                if isinstance(c, dict):
                    for k, v in c.items():
                        try:
                            counts[str(k).upper()] = int(v)
                        except Exception:
                            continue
        except Exception:
            pass
    # Fallback: scan for a Verdict line
    if not verdict:
        vm = re.search(
            r"^\s*Overall Verdict\s*:\s*(.+)$",
            md_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if vm:
            verdict = vm.group(1).strip()
    return verdict, counts


def _build_reviewer_prompt(
    *, base_branch: str, report_path: str, extra_instructions: str = ""
) -> str:
    parts: List[str] = []
    parts += [
        "<role>",
        "Senior code reviewer. Be concise, specific, and actionable.",
        "</role>",
        "",
        "<task>",
        f"Review the current changes in this repository relative to {base_branch}.",
        f"Use git to inspect the diff (e.g., git diff {base_branch}...HEAD --unified=0).",
        f"Write your review to {report_path} and commit it.",
        "</task>",
        "",
        "<requirements>",
        "- Focus on correctness, security, performance, maintainability, API stability, and tests.",
        "- Propose minimal, concrete fixes (filenames, lines, snippets).",
        "- Use severities: BLOCKER, HIGH, MEDIUM, LOW, INFO.",
        "- Keep total length reasonable; avoid repetition.",
        "</requirements>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
        ]
    parts += [
        "<output_format>",
        "# PR Review Report",
        "",
        "## Summary",
        "One concise paragraph.",
        "",
        "## Findings",
        "| Area | Severity | Description | Suggested change |",
        "| ---- | -------- | ----------- | ---------------- |",
        "(Add one row per distinct finding)",
        "",
        "## Overall Verdict: PASS or NEEDS_CHANGES",
        "</output_format>",
    ]
    return "\n".join(parts) + "\n"


def _build_validator_prompt(*, report_path: str, extra_instructions: str = "") -> str:
    parts: List[str] = []
    parts += [
        "<role>",
        "Gatekeeper validator. Confirm accuracy and appropriateness.",
        "</role>",
        "",
        "<task>",
        f"Open {report_path}. Validate each finding: drop duplicates, merge overlapping items, ",
        "correct severities, and refine suggestions to be minimal and clear. Keep the same format.",
        f"Overwrite {report_path} and commit.",
        "Append a final fenced JSON block with counts per severity and a verdict.",
        "</task>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
        ]
    parts += [
        "<json_trailer>",
        "```json",
        '{\n  "verdict": "PASS|NEEDS_CHANGES",\n  "counts": {\n    "BLOCKER": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0\n  }\n}',
        "```",
        "</json_trailer>",
    ]
    return "\n".join(parts) + "\n"


def _build_composer_prompt(
    *,
    base_branch: str,
    report_paths: List[str],
    output_path: str,
    extra_instructions: str = "",
) -> str:
    joined = "\n".join(f"- {p}" for p in report_paths)
    parts: List[str] = []
    parts += [
        "<role>",
        "Lead reviewer. Merge validated reviews into a single, polished comment for a PR.",
        "</role>",
        "",
        "<inputs>",
        "Validated report files (read and aggregate):",
        joined,
        "</inputs>",
        "",
        "<task>",
        f"Compose a final review at {output_path} and commit it. Deduplicate and group findings, ",
        "preserve severities, and include an Overall Verdict (PASS or NEEDS_CHANGES).",
        "Include a short top-level summary and a compact checklist if useful.",
        "</task>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
        ]
    return "\n".join(parts) + "\n"


def _resolve_instructions(
    text: Optional[str], path: Optional[str], default: str = ""
) -> str:
    if text and str(text).strip():
        return str(text)
    if path and str(path).strip():
        p = Path(path)
        try:
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return default
    return default
