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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


# Shared internal findings output format for reviewer and validator prompts
INTERNAL_FINDINGS_OUTPUT_FORMAT: List[str] = [
    "<output_format>",
    "## Findings",
    "",
    "Repeat the following section for each distinct issue:",
    "",
    "### [SEVERITY] <Short Title>",
    "- Severity: HIGH|MEDIUM|LOW",
    "- Location: <path>?plain=1#L<start>-L<end> (GitHub preview link)",
    "Description:",
    "Write 2–4 sentences explaining the problem and its impact.",
    "",
    "Suggestion:",
    "Provide a minimal, concrete fix. Include a short code block when helpful.",
    "",
    "Notes:",
    "- Use relative repo paths and ?plain=1#L.. anchors so GitHub renders a preview.",
    "- Normalize/merge duplicates; reference files/lines precisely; avoid repetition.",
    "- Do NOT include a summary or verdict in this message (verdict goes to JSON only).",
    "</output_format>",
    "",
    "<example>",
    "### [HIGH] SQL query uses unsanitized user input",
    "- Severity: HIGH",
    "- Location: api/search.py?plain=1#L42-L58",
    "Description:",
    "The search endpoint builds an SQL string using f-strings with the raw `q` parameter. An attacker can inject SQL via crafted input, risking data exposure or corruption.",
    "",
    "Suggestion:",
    "Use a parameterized query with placeholders and bind variables.",
    "",
    "```python",
    "cursor.execute(\"SELECT id, name FROM items WHERE name LIKE %s\", (pattern,))",
    "```",
    "",
    "### [MEDIUM] Inefficient loop for JSON serialization",
    "- Severity: MEDIUM",
    "- Location: utils/encode.py?plain=1#L15-L28",
    "Description:",
    "The code repeatedly calls `json.dumps` inside a loop, causing N small allocations and string concatenations.",
    "",
    "Suggestion:",
    "Build a list and serialize once, or use a generator with `join` to reduce allocations.",
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

        # Resolve host HEAD branch name (generic; not PR-specific)
        host_head_branch, _ = _resolve_host_head(repo_path)
        include_branches: List[str] = []
        # Strategy config-specified includes (if any)
        if cfg.include_branches:
            include_branches.extend(list(cfg.include_branches))
        # Also include the host HEAD branch when different from base
        if host_head_branch and host_head_branch != (cfg.base_branch or base_branch):
            include_branches.append(host_head_branch)
        # Deduplicate while preserving order
        seen = set()
        include_branches = [
            b for b in include_branches if not (b in seen or seen.add(b))
        ]

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
        f"Review the current changes relative to {base_branch} using local git in the workspace.",
        "Run git commands to inspect diffs (do NOT write files).",
        (
            (
                "In the workspace, additional read-only branches are available: "
                + ", ".join(sorted(set(include_branch_names or [])))
                + "."
            )
            if (include_branch_names or [])
            else ""
        ),
        (
            (
                "You may run local diffs, e.g., 'git diff "
                + f"{base_branch}...{(include_branch_names or [''])[0]}'"
            )
            if (include_branch_names or [])
            else ""
        ),
        "Do not push or modify branches.",
        "</task>",
        "",
        "<requirements>",
        "- Focus on correctness, security, performance, maintainability, API stability, and tests.",
        "- Propose minimal, concrete fixes (filenames, lines, snippets).",
        "- Use severities: HIGH, MEDIUM, LOW.",
        "- For every finding, VALIDATE the Location line range against the actual file content.",
        "  Do not guess line numbers.",
        "  Use non-destructive commands to inspect content at a branch/ref, e.g:",
        "    git show <branch>:<path> | nl -ba | sed -n '<start>,<end>p'",
        "  If the snippet does not match your description, correct the Location (or the finding) before outputting.",
        "- When referring to exact text (e.g., replace X with Y), use fenced code blocks, not quoted prose.",
        "  Prefer short before/after blocks or unified diff snippets for clarity.",
        "- Do not offer to perform further actions (e.g., 'I can draft the exact changes'); provide the concrete suggestion instead.",
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
        "Validate the reviewer report below: drop duplicates, merge overlapping items, ",
        "correct severities, and refine suggestions to be minimal and clear.",
        "Ensure the Findings section follows the structured format in <output_format> (no checkboxes).",
        "",
        "Critically: re-validate every Location line range and link before output.",
        "- Do not rely on the reviewer’s numbers; check them again.",
        "- Use read-only commands to inspect the exact blob content, for example:",
        "    git show <branch>:<path> | nl -ba | sed -n '<start>,<end>p'",
        "- If the snippet does not match the Description, fix the Location (or adjust the finding).",
        "- Do NOT add new findings at this stage; your goal is to validate, filter, merge, and adjust severity/wording/locations of the reviewer’s list only.",
        "Return the validated report as your final message and append a fenced JSON block with counts per severity and a verdict.",
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
        "Lead reviewer. Merge validated reviews into a single, polished comment for a PR.",
        "</role>",
        "",
        "<inputs>",
        "Validated reviewer reports (read and aggregate):",
        joined,
        "</inputs>",
        "",
        "<task>",
        "Compose a user-facing final review (do NOT write files). Deduplicate and group findings, ",
        "preserve severities, and begin with 2–3 friendly sentences summarizing the overall state.",
        "Do NOT include any 'Verdict', 'Status', or 'Outcome' line or heading.",
        "Do NOT discover new issues and do NOT re-validate content at this stage: you receive VALIDATED findings and can trust their severities, locations, and wording.",
        "Use simple, plain language; short, direct sentences. Do NOT use the tokens PASS/NEEDS_CHANGES in prose.",
        "Present HIGH and MEDIUM findings as expanded sections; put LOW findings inside a <details> spoiler.",
        "If there are no findings, do not include any Findings section; keep it to the opening sentences and state that you found no issues.",
        "</task>",
        "",
        "<output_format>",
        "(Opening 2–3 sentences; friendly and encouraging.)",
        "",
        "## Findings",
        "",
        "### High & Medium",
        "Repeat this section per finding:",
        "",
        "#### [SEVERITY] <Short Title>",
        "- Severity: HIGH|MEDIUM",
        "- Location: <path>?plain=1#L<start>-L<end>",
        "Description:",
        "1–3 sentences tailored for readers.",
        "",
        "Suggestion:",
        "Concrete change (use a fenced code block for exact text/diff when helpful).",
        "",
        "### Low (optional)",
        "<details><summary>Show low-severity notes</summary>",
        "",
        "Repeat per finding:",
        "",
        "#### [LOW] <Short Title>",
        "- Location: <path>?plain=1#L<start>-L<end>",
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
        "",
        "<example>",
        "Thanks for the improvements — the UI feels snappier. I found a couple of issues to address before merging.",
        "",
        "## Findings",
        "",
        "### High & Medium",
        "",
        "#### [HIGH] SQL query uses unsanitized user input",
        "- Severity: HIGH",
        "- Location: api/search.py?plain=1#L42-L58",
        "Description:",
        "The search endpoint builds an SQL string using f-strings with the raw `q` parameter. An attacker can inject SQL via crafted input, risking data exposure or corruption.",
        "",
        "Suggestion:",
        "Use a parameterized query with placeholders and bind variables.",
        "",
        "```python",
        "cursor.execute(\"SELECT id, name FROM items WHERE name LIKE %s\", (pattern,))",
        "```",
        "",
        "#### [MEDIUM] Inefficient loop for JSON serialization",
        "- Severity: MEDIUM",
        "- Location: utils/encode.py?plain=1#L15-L28",
        "Description:",
        "The code repeatedly calls `json.dumps` inside a loop, causing N small allocations and string concatenations.",
        "",
        "Suggestion:",
        "Build a list and serialize once, or use a generator with `join`.",
        "",
        "### Low (optional)",
        "<details><summary>Show low-severity notes</summary>",
        "",
        "#### [LOW] Typo in README",
        "- Location: README.md?plain=1#L12-L12",
        "Description:",
        "The word 'teh' appears in the installation section.",
        "",
        "Suggestion:",
        "Replace with 'the'.",
        "",
        "</details>",
        "",
        "(No-issues example)",
        "",
        "Thanks for the improvements — everything looks correct on my end. I didn’t spot any issues to address before merging.",
        "</example>",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
        ]
    return "\n".join(parts) + "\n"


def _resolve_host_head(repo: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (branch_name, sha) for the host repository's current HEAD.

    If HEAD is detached, branch_name will be None and sha will be set.
    """
    try:
        b = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        s = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        branch = (b.stdout or "").strip()
        sha = (s.stdout or "").strip()
        if branch and branch != "HEAD":
            return branch, sha
        return None, sha if sha else None
    except Exception:
        return None, None


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
