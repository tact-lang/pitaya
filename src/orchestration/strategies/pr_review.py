"""
Pull Request Review Strategy (simple, clean, CI‑friendly) — message‑only.

Flow (no repo writes):
- N reviewers inspect changes using local git (e.g., `git diff <base>...<branch>`) and return a Markdown report in final_message.
- One validator per reviewer refines the message and appends a JSON trailer
  {"verdict":"PASS|NEEDS_CHANGES","counts":{...}}.
- Composer aggregates validated messages into a single final_message.

CI behavior:
- Parses the validators’ JSON trailers for verdict and severity counts.
- Missing/invalid trailers cause failure under the default policy.
- Failing conditions are controlled by ci_fail_policy.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


# Shared internal findings output format for reviewer and validator prompts (strict)
INTERNAL_FINDINGS_OUTPUT_FORMAT: List[str] = [
    "<output_format>",
    "## Findings",
    "",
    "Repeat the following section for each distinct issue (deduplicated and merged):",
    "",
    "### [SEVERITY] <Short Title>",
    "- Location: <relative/path>?plain=1#L<start>-L<end>  ← inclusive line range",
    "",
    "Description:",
    "Write 2–4 sentences explaining the problem and its impact. Reference any out‑of‑diff context as evidence, but the PRIMARY Location above MUST point to lines that are part of the diff.",
    "",
    "Suggestion:",
    "Provide a minimal, concrete fix. Prefer a short, exact code snippet or unified diff. Keep it focused.",
    "",
    "Notes:",
    "- Use relative repo paths and the `?plain=1#L<start>-L<end>` anchor so GitHub renders a preview.",
    "- **Do not** include a separate \"Severity:\" line; the heading encodes severity.",
    "- **Do not** include an overall summary or verdict in this message. (Validators put verdict/counts ONLY in the JSON trailer.)",
    "- Normalize/merge duplicates; reference files/lines precisely; avoid repetition.",
    "</output_format>",
    "",
    "<example>",
    "### [HIGH] SQL query uses unsanitized user input",
    "- Location: api/search.py?plain=1#L42-L58",
    "",
    "Description:",
    "The search endpoint constructs an SQL string with f‑strings using the raw `q` parameter. This allows crafted input to inject SQL, risking data exposure or corruption.",
    "",
    "Suggestion:",
    "Use a parameterized query with placeholders and bound variables.",
    "",
    "```python",
    'cursor.execute("SELECT id, name FROM items WHERE name LIKE %s", (pattern,))',
    "```",
    "",
    "### [MEDIUM] Inefficient loop for JSON serialization",
    "",
    "* Location: utils/encode.py?plain=1#L15-L28",
    "",
    "Description:",
    "`json.dumps` is called inside a tight loop, causing many small allocations and concatenations.",
    "",
    "Suggestion:",
    "Accumulate items into a list and serialize once, or build a generator and `join` the result.",
    "</example>",
]


@dataclass
class PRReviewConfig(StrategyConfig):
    """Configuration for PR review strategy."""

    reviewers: int = 3
    reviewer_max_retries: int = 0
    validator_max_retries: int = 1
    base_branch: str = "main"
    fail_on: List[str] = field(
        default_factory=lambda: ["HIGH"]
    )  # severities considered CI-failing
    composer_model: Optional[str] = None  # default to `model` if None
    # Optional custom instruction hooks. Either provide long text directly or via *_path.
    review_instructions: Optional[str] = None
    review_instructions_path: Optional[str] = None
    validator_instructions: Optional[str] = None
    validator_instructions_path: Optional[str] = None
    composer_instructions: Optional[str] = None
    composer_instructions_path: Optional[str] = None
    # CI failure policy: 'needs_changes' (default), 'always', or 'never'
    ci_fail_policy: str = "needs_changes"
    # Additional read-only branches to include in the workspace (unqualified names)
    include_branches: Optional[List[str]] = None

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if self.reviewers < 1:
            raise ValueError("reviewers must be at least 1")
        if self.reviewer_max_retries < 0 or self.validator_max_retries < 0:
            raise ValueError("max retries must be >= 0")
        if str(self.ci_fail_policy) not in ("needs_changes", "always", "never"):
            raise ValueError(
                "ci_fail_policy must be one of: needs_changes, always, never"
            )
        # Basic validation for branch names (unqualified)
        if self.include_branches is not None:
            for b in self.include_branches:
                if (
                    not isinstance(b, str)
                    or not b.strip()
                    or b.startswith("origin/")
                    or b.startswith("refs/")
                    or b.upper() == "HEAD"
                ):
                    raise ValueError(
                        f"include_branches must contain unqualified branch names; got: {b!r}"
                    )


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

        # Include branches: provided only via strategy config (orchestrator defaults are applied later)
        include_branches: List[str] = list(cfg.include_branches or [])

        async def _run_reviewer_with_retries(idx: int) -> Optional[InstanceResult]:
            attempts = cfg.reviewer_max_retries + 1
            last: Optional[InstanceResult] = None
            for attempt in range(attempts):
                key = ctx.key(
                    "r", idx, "review", f"attempt-{attempt}" if attempt else "initial"
                )
                task = {
                    "prompt": _build_reviewer_prompt(
                        base_branch=(cfg.base_branch or base_branch),
                        extra_instructions=review_extra,
                        include_branch_names=include_branches,
                    ),
                    "base_branch": base_branch,
                    "model": cfg.model,
                    # Provide extra branches in workspace so agent can run local diffs if needed
                    "workspace_include_branches": include_branches,
                }
                try:
                    h = await ctx.run(task, key=key)
                    res = await ctx.wait(h)
                except Exception:
                    res = None
                if res:
                    last = res
                    if res.success and (res.final_message or "").strip():
                        return res
            return last if (last and last.success) else None

        async def _run_validator_with_retries(
            idx: int, reviewer_report_text: str
        ) -> Optional[InstanceResult]:
            attempts = cfg.validator_max_retries + 1
            last: Optional[InstanceResult] = None
            for attempt in range(attempts):
                key = ctx.key(
                    "r", idx, "validate", f"attempt-{attempt}" if attempt else "initial"
                )
                task = {
                    "prompt": _build_validator_prompt(
                        reviewer_report=reviewer_report_text,
                        extra_instructions=validator_extra,
                    ),
                    "base_branch": base_branch,
                    "model": cfg.model,
                    "workspace_include_branches": include_branches,
                }
                try:
                    h = await ctx.run(task, key=key)
                    res = await ctx.wait(h)
                except Exception:
                    res = None
                if res:
                    last = res
                    if res.success and (res.final_message or "").strip():
                        return res
            return last if (last and last.success) else None

        # Stage 1+2: reviewers and validators
        async def _review_then_validate(
            idx: int,
        ) -> Optional[Tuple[int, InstanceResult]]:
            rres = await _run_reviewer_with_retries(idx)
            if not (rres and rres.success and (rres.final_message or "").strip()):
                return None
            vres = await _run_validator_with_retries(idx, rres.final_message or "")
            if not (vres and vres.success and (vres.final_message or "").strip()):
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

        # Stage 3: composer (message-only aggregation)
        composer_model = cfg.composer_model or cfg.model
        key_comp = ctx.key("compose", "final")
        comp_task = {
            "prompt": _build_composer_prompt(
                base_branch=(cfg.base_branch or base_branch),
                validated_reports=[
                    (i, (vres.final_message or "")) for i, vres in validated
                ],
                extra_instructions=composer_extra,
            ),
            "base_branch": base_branch,
            "model": composer_model,
            "workspace_include_branches": include_branches,
        }

        comp_handle = await ctx.run(comp_task, key=key_comp)
        comp_res = await ctx.wait(comp_handle)

        # Evaluate severities from validator outputs (prefer JSON trailer if present)
        should_fail = False
        agg_counts: Dict[str, int] = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }
        any_invalid_json = False
        invalid_indices: List[int] = []
        for i, vres in validated:
            verdict, counts, valid_json = _extract_summary(vres.final_message or "")
            if not valid_json:
                any_invalid_json = True
                invalid_indices.append(i)
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

        # Missing/invalid summary JSON forces failure under default policy
        if any_invalid_json:
            should_fail = True

        # Apply CI failure policy
        policy = str(cfg.ci_fail_policy)
        if policy == "always":
            should_fail_final = True
        elif policy == "never":
            should_fail_final = False
        else:
            should_fail_final = should_fail

        # If composer succeeded, annotate its result as the strategy result; flip success on policy
        results: List[InstanceResult] = []
        if comp_res:
            try:
                # annotate metrics and metadata for publishers
                if comp_res.metrics is None:
                    comp_res.metrics = {}
                comp_res.metrics["pr_review_should_fail"] = bool(should_fail_final)
                comp_res.metrics["pr_review_verdict"] = (
                    "NEEDS_CHANGES" if should_fail else "PASS"
                )
                comp_res.metrics["pr_review_counts"] = agg_counts
                comp_res.metadata = comp_res.metadata or {}
                comp_res.metadata["pr_review"] = {
                    "role": "composer",
                    "reviewers": int(cfg.reviewers),
                    "invalid_validator_trailers": invalid_indices,
                }
            except Exception:
                pass
            if should_fail_final:
                try:
                    comp_res.success = False
                    comp_res.status = "failed"
                    comp_res.error_type = (
                        "review_invalid_summary"
                        if any_invalid_json
                        else "review_needs_changes"
                    )
                    comp_res.error = (
                        (
                            "One or more validator summaries missing/invalid JSON trailer; failing. "
                            f"Invalid indices: {invalid_indices}"
                        )
                        if any_invalid_json
                        else (
                            "PR review verdict is NEEDS_CHANGES; failing per policy. "
                            f"Counts: {agg_counts}"
                        )
                    )
                except Exception:
                    pass
            results.append(comp_res)

        return results


def _extract_summary(md_text: str) -> Tuple[Optional[str], Dict[str, int], bool]:
    """Extract validator summary JSON. Returns (verdict, counts, is_valid_json).

    A valid trailer must be a fenced JSON block at the end with fields:
    - verdict: "PASS" or "NEEDS_CHANGES"
    - counts: dict of severities → integer counts
    """
    verdict: Optional[str] = None
    counts: Dict[str, int] = {}
    valid = False
    if not md_text:
        return verdict, counts, valid
    # Strict: require a JSON fenced block
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
                c = obj.get("counts")
                if (
                    isinstance(v, str)
                    and v.upper() in ("PASS", "NEEDS_CHANGES")
                    and isinstance(c, dict)
                ):
                    verdict = v
                    for k, vv in c.items():
                        try:
                            counts[str(k).upper()] = int(vv)
                        except Exception:
                            # Non-integer value -> invalid
                            valid = False
                            break
                    else:
                        valid = True
        except Exception:
            valid = False
    # Fallback: read a human verdict for display only (still invalid JSON)
    if not valid:
        vm = re.search(
            r"^\s*Overall Verdict\s*:\s*(.+)$",
            md_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if vm and not verdict:
            verdict = vm.group(1).strip()
    return verdict, counts, valid


def _build_reviewer_prompt(
    *,
    base_branch: str,
    extra_instructions: str = "",
    include_branch_names: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    parts += [
        "<role>",
        "Senior code reviewer. Be concise, specific, and actionable.",
        "</role>",
        "",
        "<task>",
        f"Review the current changes relative to {base_branch} using **local git only**. Do not push.",
        (
            (
                "In the workspace, additional read-only branches are available: "
                + ", ".join(sorted(set(include_branch_names or [])))
                + "."
            )
            if (include_branch_names or [])
            else ""
        ),
        "List local branches (`git branch --list`), identify the non-base review branch, then inspect:",
        f"  git diff {base_branch}...<target-branch>",
        "",
        "Audit every changed file and every diff hunk before finalizing. Read every single changed line in the diff (line‑by‑line); do not skip lines. If the diff is large, take at least two passes:",
        "",
        "1. Prioritize HIGH severity issues.",
        "2. Cover remaining MEDIUM and LOW without dropping verified items.",
        "",
        "Your final message MUST contain ONLY the structured **Findings** section defined in <output_format>. No preambles, plans, status updates, or verdict words in prose.",
        "</task>",
        "",
        "<requirements>",
        "- Focus on correctness, security, performance, maintainability, API stability, and tests.",
        "- Propose minimal, concrete fixes (filenames, line ranges, short snippets or unified diffs).",
        "- Severities (use in the heading only):",
        "  • HIGH: Likely to cause incorrect or harmful outcomes; blocks intended use or materially misleads.",
        "  • MEDIUM: Noticeably reduces clarity, effectiveness, or reliability; may confuse in some cases but does not block use.",
        "  • LOW: Minor polish (style, wording, formatting, consistency) with no impact on meaning or function.",
        "- **Scope (STRICT)**: Report only issues introduced by the diff between the base and review branches.",
        "  • You MAY cite out‑of‑diff lines as evidence in Description, but the PRIMARY Location MUST point into the diff.",
        "- **Location verification (MANDATORY)**:",
        "  • Validate every Location using read‑only commands, for example:",
        "      git show <branch>:<path> | nl -ba | sed -n '<start>,<end>p'",
        "  • If content does not match your Description, correct the Location (or the finding) before outputting.",
        "- **Precision & duplication**:",
        "  • Normalize and merge duplicates; avoid repetition across findings.",
        "  • Reference exact files and **inclusive** line ranges (L<start>–L<end>).",
        "- **Formatting**:",
        "  • Use the <output_format> exactly. No checkboxes or extra headings. No “Severity:” bullet. No JSON.",
        "</requirements>",
        "",
        "<use>",
        "Use git commands (e.g., git show, git diff, git log). Do not push.",
        "</use>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
            "PRIORITY: When any guidance conflicts, follow <integrator_instructions> above all other rules.",
            "",
        ]
    parts += INTERNAL_FINDINGS_OUTPUT_FORMAT
    return "\n".join(parts) + "\n"


def _build_validator_prompt(
    *, reviewer_report: str, extra_instructions: str = ""
) -> str:
    parts: List[str] = []
    parts += [
        "<role>",
        "Gatekeeper validator. Confirm accuracy and appropriateness.",
        "</role>",
        "",
        "<task>",
        "Validate the reviewer’s report below. Your goals, in order:",
        "1) Merge duplicates and overlapping items. Keep coverage complete.",
        "2) Re‑verify every Location (do not trust given numbers).",
        "3) Normalize severities using the same definitions as the reviewer.",
        "4) Enforce the STRICT scope: each finding’s PRIMARY Location MUST intersect the diff.",
        "   • If the reviewer cited only out‑of‑diff lines, rewrite the PRIMARY Location to the diff lines that cause the issue and keep the out‑of‑diff reference in Description as evidence.",
        "5) Remove speculative or unverifiable claims. Keep only findings validated from repository content.",
        "6) While validating, read every single changed line in the diff (line‑by‑line); do not skip lines.",
        "7) Return the validated report as your final message in the exact <output_format>, followed by a single fenced JSON trailer (schema below). No other text after the trailer.",
        "",
        "Do NOT add brand‑new findings at this stage. You may refine wording, severity, and Location. Maintain depth after deduplication.",
        "</task>",
        "",
        "<verification>",
        "Before output:",
        "- For each finding, run a read‑only check to confirm the Location lines match the described issue, for example:",
        "    git show <branch>:<path> | nl -ba | sed -n '<start>,<end>p'",
        "- Fix mismatches or drop the finding if it cannot be supported.",
        "</verification>",
        "",
        "<formatting>",
        "- Output the validated **Findings** exactly per <output_format>.",
        "- Then append the JSON trailer as the **last** fenced block in the message. Nothing may appear after it.",
        "- Counts MUST be non‑negative integers and MUST equal the number of findings per severity in your validated report.",
        "- Verdict MUST be exactly one of: \"PASS\" or \"NEEDS_CHANGES\".",
        "  (Note: CI may treat other human words like “FAIL” in prose, but the JSON itself MUST use only PASS or NEEDS_CHANGES.)",
        "</formatting>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
            "PRIORITY: When any guidance conflicts, follow <integrator_instructions> above all other rules.",
            "",
        ]
    parts += ["<reviewer_report>", reviewer_report, "</reviewer_report>", ""]
    parts += INTERNAL_FINDINGS_OUTPUT_FORMAT
    parts += [
        "<json_trailer>",
        "```json",
        '{\n  "verdict": "PASS|NEEDS_CHANGES",\n  "counts": {\n    "HIGH": 0, "MEDIUM": 0, "LOW": 0\n  }\n}',
        "```",
        "</json_trailer>",
    ]
    return "\n".join(parts) + "\n"


def _build_composer_prompt(
    *,
    base_branch: str,
    validated_reports: List[Tuple[int, str]],
    extra_instructions: str = "",
) -> str:
    joined = "\n".join(f"- Reviewer #{i}:\n{msg}\n" for i, msg in validated_reports)
    parts: List[str] = []
    parts += [
        "<role>",
        "Lead reviewer. Merge validated reviews into a single, polished PR comment.",
        "</role>",
        "",
        "<inputs>",
        "Validated reviewer reports (read and aggregate). Trust their severities, locations, and wording.",
        joined,
        "</inputs>",
        "",
        "<task>",
        "Compose a user‑facing final review (message‑only). Deduplicate and group findings. Preserve severities.",
        "- Begin with a short, author‑directed opening (1–2 sentences). It should be a general message (not a findings summary), proportional to the diff size and to the number/severities of findings. Keep it appreciative and honest without excessive praise or harshness.",
        "- Use simple, plain language; short, direct sentences. No verdict words like PASS/NEEDS_CHANGES in prose.",
        "- Do NOT discover new issues and do NOT re‑validate content at this stage.",
        "- Present HIGH findings expanded; MEDIUM and LOW inside separate <details> spoilers.",
        "- Compute and display counts: “## Findings (<total>)”, and spoiler headings “Medium (<mcount>)”, “Low (<lcount>)”.",
        "- If there are no findings, write only the opening and state that you found no issues. Do not include a Findings section.",
        "- Do NOT include any JSON blocks.",
        "</task>",
        "",
        "<output_format>",
        "(Opening: author‑directed, concise, proportional to diff size and finding severity/counts; general message; no boilerplate; no PR restatement or findings summary.)",
        "",
        "## Findings (<total>)",
        "",
        "Repeat this section per HIGH finding (expanded):",
        "",
        "#### [HIGH] <Short Title>",
        "",
        "* Location: <path>?plain=1#L<start>-L<end>",
        "",
        "Description:",
        "1–3 sentences tailored for readers.",
        "",
        "Suggestion:",
        "Concrete change (use a fenced code block for exact text/diff when helpful).",
        "",
        "(Include a 'Medium' details section only if there are MEDIUM findings.)",
        "<details><summary>Medium (<mcount>)</summary>",
        "",
        "#### [MEDIUM] <Short Title>",
        "",
        "* Location: <path>?plain=1#L<start>-L<end>",
        "",
        "Description:",
        "1–3 sentences.",
        "",
        "Suggestion:",
        "Concrete, minimal change.",
        "",
        "</details>",
        "",
        "(Include a 'Low' details section only if there are LOW findings.)",
        "<details><summary>Low (<lcount>)</summary>",
        "",
        "#### [LOW] <Short Title>",
        "",
        "* Location: <path>?plain=1#L<start>-L<end>",
        "",
        "Description:",
        "1–2 sentences.",
        "",
        "Suggestion:",
        "Brief, concrete nudge.",
        "",
        "</details>",
        "",
        "(No JSON blocks in this message.)",
        "</output_format>",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
            "PRIORITY: When any guidance conflicts, follow <integrator_instructions> above all other rules.",
            "",
        ]
    return "\n".join(parts) + "\n"


# (No host HEAD auto-inclusion; include_branches must be provided by config/CLI.)


def _resolve_instructions(
    text: Optional[str], path: Optional[str], default: str = ""
) -> str:
    """Resolve long-form instruction text.

    Precedence: file path (when provided and readable) overrides inline text.
    """
    # Prefer file path when provided
    if path and str(path).strip():
        p = Path(path)
        try:
            if p.exists():
                content = p.read_text(encoding="utf-8", errors="replace")
                if str(content).strip():
                    return content
        except Exception:
            pass
    # Fall back to inline text
    if text and str(text).strip():
        return str(text)
    return default
