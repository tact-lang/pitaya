"""
Pull Request Review Strategy — validator‑anchored, JSON‑selected.

Flow:
- Reviewers: N reviewers inspect the diff (e.g., `git diff <base>...<branch>`) and return a Markdown report containing only a strict Findings section.
- Validators: One validator per reviewer strictly validates, adds a canonical UID per finding, inserts an HTML marker `<!-- fid:<uid> -->` after each finding heading, and appends a JSON trailer
  {"verdict":"PASS|NEEDS_CHANGES","counts":{...},"findings":[{"uid":...,"path":...,"start":...,"end":...,"severity":...,"title":...}, ...]}.
- Composer: Intelligently filters and de‑duplicates findings across validators and returns JSON only:
  {"intro": "1–2 sentences", "selected_ids": ["fid:...", ...]}.
  Strategy parses that JSON, sets the intro as the human final_message, and builds the sidecar (selected_details) by stitching exactly those IDs from validators.

CI behavior:
- Parses the validators’ JSON trailers for verdict and severity counts (unchanged), and now also expects the `findings` array with valid UIDs.
- Missing/invalid trailers (including missing UIDs) cause failure under the default policy.
- Failing conditions are controlled by ci_fail_policy.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


# Shared internal findings output format for reviewer and validator prompts (strict)
# Reviewer version: no UID marker.
INTERNAL_FINDINGS_OUTPUT_FORMAT_REVIEW: List[str] = [
    "<output_format>",
    "## Findings",
    "",
    "Repeat the following section for each distinct issue (deduplicated and merged):",
    "",
    "### [SEVERITY] <Short Title>",
    "Location: <relative/path>?plain=1#L<start>-L<end>  ← inclusive line range",
    "",
    "Description:",
    "Write 2–4 sentences explaining the problem and its impact. Reference any out‑of‑diff context as evidence, but the PRIMARY Location above MUST point to lines that are part of the diff.",
    "",
    "Suggestion:",
    "Provide a minimal, concrete fix. Prefer a short, exact code snippet or unified diff. Keep it focused.",
    "",
    "Notes:",
    "* Use relative repo paths and the `?plain=1#L<start>-L<end>` anchor so GitHub renders a preview.",
    "* Link formatting (strict):",
    "  • Inline references → use a Markdown link with text: `[tvm/runvm.mdx L5–L5](tvm/runvm.mdx?plain=1#L5-L5)`.",
    "  • Absolute GitHub links (commit/branch) that render a rich preview → place on their own line, without parentheses or trailing punctuation.",
    "    Example (standalone preview):",
    "    https://github.com/<owner>/<repo>/blob/<sha>/path/to/file?plain=1#L5-L5",
    '* **Do not** include a separate "Severity:" line; the heading encodes severity.',
    "* **Do not** include an overall summary or verdict in this message. (Validators put verdict/counts ONLY in the JSON trailer.)",
    "* Normalize/merge duplicates; reference files/lines precisely; avoid repetition.",
    "</output_format>",
    "",
    "<example>",
    "### [HIGH] SQL query uses unsanitized user input",
    "Location: api/search.py?plain=1#L42-L58",
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
    "Location: utils/encode.py?plain=1#L15-L28",
    "",
    "Description:",
    "`json.dumps` is called inside a tight loop, causing many small allocations and concatenations.",
    "",
    "Suggestion:",
    "Accumulate items into a list and serialize once, or build a generator and `join` the result.",
    "</example>",
]

# Validator version: includes hidden UID marker requirement as a literal line
# that MUST be included (with a real UID) in the final message.
INTERNAL_FINDINGS_OUTPUT_FORMAT_VALIDATOR: List[str] = [
    "<output_format>",
    "## Findings",
    "",
    "Repeat the following section for each distinct issue (deduplicated and merged):",
    "",
    "### [SEVERITY] <Short Title>",
    "Location: <relative/path>?plain=1#L<start>-L<end>  ← inclusive line range",
    "<!-- fid:<uid> -->  ← insert the computed UID here as a hidden HTML comment.",
    "",
    "Description:",
    "Write 2–4 sentences explaining the problem and its impact. Reference any out‑of‑diff context as evidence, but the PRIMARY Location above MUST point to lines that are part of the diff.",
    "",
    "Suggestion:",
    "Provide a minimal, concrete fix. Prefer a short, exact code snippet or unified diff. Keep it focused.",
    "",
    "Notes:",
    "* Use relative repo paths and the `?plain=1#L<start>-L<end>` anchor so GitHub renders a preview.",
    '* **Do not** include a separate "Severity:" line; the heading encodes severity.',
    "* **Do not** include an overall summary or verdict in this message. (Validators put verdict/counts ONLY in the JSON trailer.)",
    "* Normalize/merge duplicates; reference files/lines precisely; avoid repetition.",
    "</output_format>",
]


@dataclass
class PRReviewConfig(StrategyConfig):
    """Configuration for PR review strategy."""

    reviewers: int = 3
    reviewer_max_retries: int = 0
    validator_max_retries: int = 1
    base_branch: Optional[str] = None
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
        if self.base_branch is not None:
            if not isinstance(self.base_branch, str) or not self.base_branch.strip():
                raise ValueError("base_branch must be a non-empty string when provided")


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
        # Ensure the workspace contains the branch under review (e.g., PR head).
        # When the orchestrator detects a non-base HEAD it exposes it via
        # default_workspace_include_branches; merge that here so reviewers,
        # validators, and the composer all see the same diff source.
        try:
            default_includes = getattr(
                ctx._orchestrator,
                "default_workspace_include_branches",
                None,  # type: ignore[attr-defined]
            )
        except Exception:
            default_includes = None
        if default_includes:
            for name in default_includes:
                if name and name not in include_branches:
                    include_branches.append(name)

        workspace_branch = base_branch

        def _resolve_compare_branch() -> str:
            if cfg.base_branch and cfg.base_branch.strip():
                return cfg.base_branch.strip()
            for name in include_branches:
                if (
                    isinstance(name, str)
                    and name.strip()
                    and name.strip() != workspace_branch
                ):
                    return name.strip()
            return workspace_branch

        compare_branch = _resolve_compare_branch()

        if (
            compare_branch != workspace_branch
            and compare_branch not in include_branches
        ):
            include_branches.append(compare_branch)

        async def _run_reviewer_with_retries(idx: int) -> Optional[InstanceResult]:
            attempts = cfg.reviewer_max_retries + 1
            last: Optional[InstanceResult] = None
            for attempt in range(attempts):
                key = ctx.key(
                    "r", idx, "review", f"attempt-{attempt}" if attempt else "initial"
                )
                task = {
                    "prompt": _build_reviewer_prompt(
                        base_branch=compare_branch,
                        workspace_branch=workspace_branch,
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
                        base_branch=compare_branch,
                        workspace_branch=workspace_branch,
                        reviewer_report=reviewer_report_text,
                        extra_instructions=validator_extra,
                        include_branch_names=include_branches,
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

        # Gather validator trailers and raw texts; treat missing/invalid as invalid JSON
        # The composer will perform semantic filtering/dedup; we pass candidates through.
        any_invalid_json = False
        invalid_indices: List[int] = []
        validator_uid_by_idx: Dict[int, List[Dict[str, object]]] = {}
        validator_text_by_idx: Dict[int, str] = {}
        # Basic stability guardrails (do not hard-merge; just flag inconsistencies)
        seen_by_coords: Dict[Tuple[str, int, int], str] = {}
        seen_by_uid: Dict[str, Tuple[str, int, int]] = {}
        for i, vres in validated:
            vtext = vres.final_message or ""
            v_payload = _extract_validator_payload(vtext)
            if not v_payload.valid:
                any_invalid_json = True
                invalid_indices.append(i)
                continue
            # Also require findings with uids
            if not v_payload.findings:
                any_invalid_json = True
                invalid_indices.append(i)
                continue
            # Enforce per-trailer UID uniqueness
            _seen_local: set[str] = set()
            for _it in v_payload.findings:
                _u = str(_it.get("uid", "")).strip()
                if _u in _seen_local:
                    any_invalid_json = True
                    if i not in invalid_indices:
                        invalid_indices.append(i)
                else:
                    _seen_local.add(_u)
            validator_uid_by_idx[i] = v_payload.findings
            validator_text_by_idx[i] = vtext
            for f in v_payload.findings:
                uid = str(f.get("uid", "")).strip()
                path = str(f.get("path", "")).strip()
                try:
                    start = int(f.get("start", 0))
                    end = int(f.get("end", 0))
                except Exception:
                    start, end = 0, 0
                if uid and path and start and end:
                    coords = (path, start, end)
                    prev_uid = seen_by_coords.get(coords)
                    if prev_uid and prev_uid != uid and i not in invalid_indices:
                        invalid_indices.append(i)
                        any_invalid_json = True
                    prev_coords = seen_by_uid.get(uid)
                    if (
                        prev_coords
                        and prev_coords != coords
                        and i not in invalid_indices
                    ):
                        invalid_indices.append(i)
                        any_invalid_json = True
                    seen_by_coords[coords] = uid
                    seen_by_uid[uid] = coords

        # Stage 3: composer (JSON with intro + selected_ids; LLM handles semantic dedup/filter)
        composer_model = cfg.composer_model or cfg.model
        key_comp = ctx.key("compose", "final")
        # Build candidate list (may include duplicates across validators)
        candidates: List[Dict[str, object]] = []
        for vidx, flist in validator_uid_by_idx.items():
            for f in flist:
                candidates.append(
                    {
                        "uid": str(f.get("uid", "")),
                        "path": str(f.get("path", "")),
                        "start": int(f.get("start", 0) or 0),
                        "end": int(f.get("end", 0) or 0),
                        "severity": str(f.get("severity", "")).upper(),
                        "title": str(f.get("title", "")),
                        "validator_index": int(vidx),
                    }
                )
        comp_task = {
            "prompt": _build_composer_prompt(
                base_branch=compare_branch,
                workspace_branch=workspace_branch,
                validated_reports=[
                    (i, (vres.final_message or "")) for i, vres in validated
                ],
                candidates=candidates,
                extra_instructions=composer_extra,
                include_branch_names=include_branches,
            ),
            "base_branch": base_branch,
            "model": composer_model,
            "workspace_include_branches": include_branches,
        }

        comp_handle = await ctx.run(comp_task, key=key_comp)
        comp_res = await ctx.wait(comp_handle)
        # Parse composer JSON (intro + selected_ids). No fallbacks.
        intro_text = ""
        selected_ids: List[str] = []
        composer_invalid = False
        if comp_res and (comp_res.final_message or "").strip():
            _intro, _ids = _extract_composer_selection(comp_res.final_message or "")
            if _ids:
                intro_text, selected_ids = _intro, _ids
            else:
                composer_invalid = True
        else:
            composer_invalid = True
        # Force composer final_message to intro text only for human output (may be empty on invalid)
        if comp_res:
            try:
                comp_res.final_message = (intro_text or "").strip()
            except Exception:
                pass

        # Evaluate severities from validator outputs (prefer JSON trailer if present)
        should_fail = False
        agg_counts: Dict[str, int] = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }
        # `any_invalid_json` and `invalid_indices` were already computed during selection;
        # continue to add additional invalids discovered by summary parsing below.
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

        # Missing/invalid summary JSON (including missing findings/uids) or invalid composer selection JSON forces failure
        if any_invalid_json or composer_invalid:
            should_fail = True

        # Apply CI failure policy
        policy = str(cfg.ci_fail_policy)
        if policy == "always":
            should_fail_final = True
        elif policy == "never":
            should_fail_final = False
        else:
            should_fail_final = should_fail

        # If composer succeeded, prepare sidecar metadata and annotate result; flip success on policy
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
                    "invalid_composer": bool(composer_invalid),
                }
                # Compute event type for sidecar
                event = (
                    "REQUEST_CHANGES"
                    if should_fail
                    else ("APPROVE" if len(selected_ids) == 0 else "COMMENT")
                )
                # Extract selected details by stitching validator messages
                selected_details = _build_selected_details(
                    selected_uids=selected_ids,
                    validator_trailers=validator_uid_by_idx,
                    validator_texts=validator_text_by_idx,
                )
                # Resolve commit id (review head if available; fallback to HEAD)
                commit_id = _resolve_commit_id(
                    repo_path=repo_path,
                    base=compare_branch,
                    workspace_branch=workspace_branch,
                    include_branch_names=include_branches,
                )
                # Persist sidecar file under results/<run_id>/review/index.json
                try:
                    run_id = getattr(ctx._orchestrator.state_manager.current_state, "run_id", None)  # type: ignore[attr-defined]
                except Exception:
                    run_id = None
                sidecar_path = None
                if run_id:
                    sidecar_path = _write_review_sidecar(
                        run_id=str(run_id),
                        intro=(comp_res.final_message or "").strip(),
                        event=event,
                        selected_ids=selected_ids,
                        selected_details=selected_details,
                        commit_id=commit_id,
                    )
                # Mirror core data into metadata for programmatic consumers
                comp_res.metadata["pr_review"].update(
                    {
                        "selected_ids": selected_ids,
                        "selected_details_count": len(selected_details),
                        "event": event,
                        "commit_id": commit_id,
                        "sidecar_path": sidecar_path,
                    }
                )
            except Exception:
                pass
            if should_fail_final:
                try:
                    comp_res.success = False
                    comp_res.status = "failed"
                    comp_res.error_type = (
                        "review_invalid_summary"
                        if (any_invalid_json or composer_invalid)
                        else "review_needs_changes"
                    )
                    comp_res.error = (
                        (
                            "Invalid JSON inputs: "
                            + ("composer selection; " if composer_invalid else "")
                            + ("validator trailers; " if any_invalid_json else "")
                            + f"Invalid validator indices: {invalid_indices}"
                        )
                        if (any_invalid_json or composer_invalid)
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


@dataclass
class _ValidatorPayload:
    verdict: Optional[str]
    counts: Dict[str, int]
    findings: List[Dict[str, object]]
    valid: bool


def _extract_validator_payload(md_text: str) -> _ValidatorPayload:
    """Parse the validator's fenced JSON trailer into a structured payload.

    Returns a payload with `valid=False` if the fenced JSON block is missing,
    malformed, or any required fields are invalid. `findings` is validated to be
    a list of objects each containing a non-empty `uid` string and numeric
    `start`/`end` where present.
    """
    verdict: Optional[str] = None
    counts: Dict[str, int] = {}
    findings: List[Dict[str, object]] = []
    # locate fenced JSON (same pattern as _extract_summary)
    m = re.search(
        r"```json\s*(\{[\s\S]*?\})\s*```\s*$",
        md_text or "",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return _ValidatorPayload(verdict, counts, findings, False)
    try:
        obj = json.loads(m.group(1))
        if not isinstance(obj, dict):
            return _ValidatorPayload(verdict, counts, findings, False)
        # mandatory fields as in _extract_summary
        v = obj.get("verdict")
        c = obj.get("counts")
        if not (
            isinstance(v, str)
            and v.upper() in ("PASS", "NEEDS_CHANGES")
            and isinstance(c, dict)
        ):
            return _ValidatorPayload(verdict, counts, findings, False)
        verdict = v
        for k, vv in c.items():
            counts[str(k).upper()] = int(vv)
    except Exception:
        return _ValidatorPayload(verdict, counts, findings, False)

    # findings are required for this strategy to compose
    f = obj.get("findings")
    if not isinstance(f, list):
        return _ValidatorPayload(verdict, counts, findings, False)
    for item in f:
        if not isinstance(item, dict):
            return _ValidatorPayload(verdict, counts, [], False)
        uid = str(item.get("uid", "")).strip()
        if not uid:
            return _ValidatorPayload(verdict, counts, [], False)
        # normalize path and line numbers if present
        norm: Dict[str, object] = {"uid": uid}
        if "path" in item:
            norm["path"] = str(item.get("path") or "").strip()
        for k in ("start", "end"):
            if k in item:
                try:
                    norm[k] = int(item.get(k))
                except Exception:
                    return _ValidatorPayload(verdict, counts, [], False)
        if "severity" in item:
            norm["severity"] = str(item.get("severity") or "").upper()
        if "title" in item:
            norm["title"] = str(item.get("title") or "").strip()
        findings.append(norm)
    return _ValidatorPayload(verdict, counts, findings, True)


def _extract_composer_selection(text: str) -> Tuple[str, List[str]]:
    """Parse the composer JSON: {intro, selected_ids}.

    Accepts either bare JSON or a fenced ```json block at the end.
    Returns (intro, selected_ids) or ("", []) when parsing fails.
    """
    if not text:
        return "", []
    # Try fenced JSON at end
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```\s*$", text, flags=re.I | re.M)
    raw = None
    if m:
        raw = m.group(1)
    else:
        # Try to parse as bare JSON
        t = text.strip()
        if t.startswith("{") and t.endswith("}"):
            raw = t
    if not raw:
        return "", []
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return "", []
        intro = str(obj.get("intro", "") or "")
        ids = obj.get("selected_ids")
        if isinstance(ids, list):
            sel = [str(x) for x in ids if isinstance(x, (str, int))]
        else:
            sel = []
        return intro, sel
    except Exception:
        return "", []


def _format_branch_context(
    *,
    compare_base: str,
    workspace_branch: str,
    include_branch_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Return prompt lines describing git context and ordered review branches."""

    include_branch_names = [
        b.strip()
        for b in (include_branch_names or [])
        if isinstance(b, str) and b.strip()
    ]

    review_branches: List[str] = []
    if (
        workspace_branch
        and workspace_branch.lower() != "head"
        and (workspace_branch != compare_base)
    ):
        review_branches.append(workspace_branch)

    for name in include_branch_names:
        if name.lower() == "head":
            continue
        if name != compare_base and name not in review_branches:
            review_branches.append(name)

    lines: List[str] = []
    if workspace_branch == compare_base:
        lines.append(
            f"Workspace checkout starts on `{workspace_branch}`, which is also the diff base."
        )
    else:
        lines += [
            f"Workspace checkout starts on review branch `{workspace_branch}` (the PR head).",
            f"Compare all changes against base branch `{compare_base}` using an explicit diff.",
        ]

    if review_branches:
        joined = ", ".join(f"`{b}`" for b in review_branches)
        lines += [
            "Review branch candidates available locally:",
            f"* {joined}",
            (
                "Use `git diff {compare_base}...<review-branch>` (default "
                f"`{review_branches[0]}`) to inspect the PR changes."
            ),
        ]
    else:
        lines.append(
            f"No alternate review branch detected; diff the working tree against `{compare_base}`."
        )

    return lines, review_branches


def _build_reviewer_prompt(
    *,
    base_branch: str,
    workspace_branch: str,
    extra_instructions: str = "",
    include_branch_names: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    branch_context_lines, review_branches = _format_branch_context(
        compare_base=base_branch,
        workspace_branch=workspace_branch,
        include_branch_names=include_branch_names,
    )

    parts += [
        "<role>",
        "Senior code reviewer. Be concise, specific, and actionable.",
        "</role>",
        "",
        "<task>",
        f"Review the current changes relative to {base_branch} using **local git only**. Do not push.",
        *branch_context_lines,
        (
            "List local branches (`git branch --list`), identify the review branch"
            + (
                f" (expected: {', '.join(f'`{b}`' for b in review_branches)})"
                if review_branches
                else ""
            )
            + ", then inspect:"
        ),
        (
            f"  git diff {base_branch}...{review_branches[0]}"
            if review_branches
            else f"  git diff {base_branch}...<target-branch>"
        ),
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
        "* Keep scope to the current diff: focus only on issues introduced by these changes.",
        "* Propose minimal, concrete fixes (filenames, line ranges, short snippets or unified diffs).",
        "* Severities (use in the heading only):",
        "  • HIGH: Likely to cause incorrect or harmful outcomes; blocks intended use or materially misleads.",
        "  • MEDIUM: Noticeably reduces clarity, effectiveness, or reliability; may confuse in some cases but does not block use.",
        "  • LOW: Minor polish (style, wording, formatting, consistency) with no impact on meaning or function.",
        "* **Scope (STRICT)**: Report only issues introduced by the diff between the base and review branches.",
        "  • You MAY cite out‑of‑diff lines as evidence in Description, but the PRIMARY Location MUST point into the diff.",
        "* **Location verification (MANDATORY)**:",
        "  • Validate every Location using read‑only commands, for example:",
        (
            f"      git show {review_branches[0] if review_branches else workspace_branch}:<path> | nl -ba | sed -n '<start>,<end>p'"
        ),
        "  • If content does not match your Description, correct the Location (or the finding) before outputting.",
        "* **Precision & duplication**:",
        "  • Normalize and merge duplicates; avoid repetition across findings.",
        "  • Reference exact files and **inclusive** line ranges (L<start>–L<end>).",
        "* **Formatting**:",
        "  • Use the <output_format> exactly. No checkboxes or extra headings. No “Severity:” bullet. No JSON.",
        "  • Links: avoid bare links in parentheses.",
        "    – Inline reference → use a Markdown link with text: `[file Ls](path?plain=1#Lx-Ly)`.",
        "    – Absolute GitHub links (commit/branch) that create a rich preview must be placed on their own line, with no surrounding punctuation.",
        "  • For formatting-only or textual edits (docs, comments, strings, spacing), prefer a minimal unified diff snippet using +/− lines (fenced as diff) when the change fits in a few lines; include only 1–3 lines of context.",
        "  • For larger changes that don’t fit cleanly in a short diff or span multiple files/blocks, explain the intent and approach in words and reference precise files/lines.",
        "  • When suggesting exact small changes, prefer a unified diff over descriptive prose.",
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
            "MANDATORY: <integrator_instructions> extend these rules and are required. If any guidance conflicts, treat <integrator_instructions> as the source of truth and resolve the conflict in their favor.",
            "",
        ]
    parts += INTERNAL_FINDINGS_OUTPUT_FORMAT_REVIEW
    return "\n".join(parts) + "\n"


def _build_validator_prompt(
    *,
    base_branch: str,
    workspace_branch: str,
    reviewer_report: str,
    extra_instructions: str = "",
    include_branch_names: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    branch_context_lines, review_branches = _format_branch_context(
        compare_base=base_branch,
        workspace_branch=workspace_branch,
        include_branch_names=include_branch_names,
    )

    parts += [
        "<role>",
        "Gatekeeper validator. Confirm accuracy and appropriateness.",
        "</role>",
        "",
        "<task>",
        "Validate the reviewer’s report below. Your goals, in order:",
        *branch_context_lines,
        (
            "* Before verification, ensure the review branch is checked out (for example: "
            + (
                f"`git checkout {review_branches[0]}`"
                if review_branches
                else "`git checkout <review-branch>`"
            )
            + ") and diff against the base with "
            + (
                f"`git diff {base_branch}...{review_branches[0]}`"
                if review_branches
                else f"`git diff {base_branch}...<review-branch>`"
            )
            + "."
        ),
        "1) Merge duplicates and overlapping items. Keep coverage complete.",
        "2) Re‑verify every Location (do not trust given numbers).",
        "3) Normalize severities using the same definitions as the reviewer.",
        "4) Enforce the STRICT scope: each finding’s PRIMARY Location MUST intersect the diff.",
        "   • If the reviewer cited only out‑of‑diff lines, rewrite the PRIMARY Location to the diff lines that cause the issue and keep the out‑of‑diff reference in Description as evidence.",
        "5) Remove speculative or unverifiable claims. Keep only findings validated from repository content.",
        "6) While validating, read every single changed line in the diff (line‑by‑line); do not skip lines.",
        "7) Assign a canonical UID per finding and insert it as a hidden HTML marker right AFTER each finding heading: <!-- fid:<uid> -->.",
        "   • UID construction (canonical, cross‑validator):",
        (
            "     - Inputs: <path>, <start>, <end>, and the exact line span bytes from the review branch for L<start>-L<end>.\n"
            "     - Compute span hash with git: git show <review-branch>:<path> | sed -n '<start>,<end>p' | git hash-object --stdin\n"
            "     - Then: uid = 'fid:' + sha1( path + '\n' + str(start) + '-' + str(end) + '\n' + span_hash )[0:12]"
        ),
        "7a) Convert suggestions to GitHub review suggestions when possible:",
        "   • If the change is a single contiguous range (L<start>–L<end> in one file), under ‘Suggestion:’ include exactly one fenced block using the GitHub suggestion language:",
        "     ```suggestion",
        "     <replacement lines only — NO +/- prefixes, NO diff headers>",
        "     ```",
        "   • Ensure Location.start/end equals the contiguous lines to be replaced (RIGHT side).",
        "   • If multiple non‑contiguous edits are needed, split into multiple findings; each gets its own Location and suggestion fence.",
        "   • Only when a GitHub suggestion cannot express the change, fall back to a minimal fenced diff for human context.",
        "8) Return the validated report as your final message in the exact <output_format>, then append ONE fenced JSON trailer (schema below). Nothing may appear after it.",
        "",
        "Do NOT add brand‑new findings at this stage. You may refine wording, severity, and Location. Maintain depth after deduplication.",
        "</task>",
        "",
        "<verification>",
        "Before output:",
        "* For each finding, run a read‑only check to confirm the Location lines match the described issue. Prefer commands such as:",
        (
            f"    git show {review_branches[0]}:<path> | nl -ba | sed -n '<start>,<end>p'"
            if review_branches
            else "    git show <review-branch>:<path> | nl -ba | sed -n '<start>,<end>p'"
        ),
        "* Fix mismatches or drop the finding if it cannot be supported.",
        "</verification>",
        "",
        "<formatting>",
        "* Output the validated **Findings** exactly per <output_format>.",
        "* Then append the JSON trailer as the **last** fenced block in the message. Nothing may appear after it.",
        "* Counts MUST be non‑negative integers and MUST equal the number of findings per severity in your validated report.",
        '* Verdict MUST be exactly one of: "PASS" or "NEEDS_CHANGES".',
        "  (Note: CI may treat other human words like “FAIL” in prose, but the JSON itself MUST use only PASS or NEEDS_CHANGES.)",
        "* Suggestion blocks (GitHub): prefer a single fenced ```suggestion with replacement text only (no +/- or headers).",
        "  • Location MUST match the exact contiguous lines to be replaced (RIGHT side).",
        "  • One suggestion fence per finding. No extra prose or code fences inside the suggestion fence.",
        "* Fallback when GH suggestion is not possible: include a minimal fenced ```diff for human context (1–3 lines).",
        "  • Keep diff minimal; do not emit large diffs. Do not mix +/- from diff into the suggestion fence.",
        "* For larger refactors or multi-file edits, keep the suggestion in words: describe the intent and exact places to update.",
        "</formatting>",
        "",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
            "MANDATORY: <integrator_instructions> extend these rules and are required. If any guidance conflicts, treat <integrator_instructions> as the source of truth and resolve the conflict in their favor.",
            "",
        ]
    parts += ["<reviewer_report>", reviewer_report, "</reviewer_report>", ""]
    parts += INTERNAL_FINDINGS_OUTPUT_FORMAT_VALIDATOR
    parts += [
        "<json_trailer>",
        "```json",
        '{\n  "verdict": "PASS|NEEDS_CHANGES",\n  "counts": {\n    "HIGH": 0, "MEDIUM": 0, "LOW": 0\n  },\n  "findings": [\n    {\n      "uid": "fid:xxxxxxxxxxxx",\n      "path": "path/to/file.py",\n      "start": 10,\n      "end": 20,\n      "severity": "HIGH|MEDIUM|LOW",\n      "title": "Short title"\n    }\n  ]\n}',
        "```",
        "</json_trailer>",
    ]
    return "\n".join(parts) + "\n"


def _build_composer_prompt(
    *,
    base_branch: str,
    workspace_branch: str,
    validated_reports: List[Tuple[int, str]],
    candidates: List[Dict[str, object]],
    extra_instructions: str = "",
    include_branch_names: Optional[List[str]] = None,
) -> str:
    joined_reports = "\n".join(
        f"* Reviewer #{i}:\n{msg}\n" for i, msg in validated_reports
    )
    parts: List[str] = []
    branch_context_lines, review_branches = _format_branch_context(
        compare_base=base_branch,
        workspace_branch=workspace_branch,
        include_branch_names=include_branch_names,
    )
    parts += [
        "<role>",
        "Selector composer. Perform semantic filtering and de‑duplication of validated findings; output JSON only.",
        "</role>",
        "",
        "<inputs>",
        "Validated reviewer reports (context only):",
        joined_reports,
        "",
        "Candidates (normalized; may include duplicates across validators):",
        json.dumps(candidates, ensure_ascii=False, indent=2),
        "</inputs>",
        "",
        "<task>",
        *branch_context_lines,
        "Goal: choose the minimal, non‑overlapping set of UIDs that best represents the diff’s issues, then write a concise author‑facing intro.",
        "Selection rules:",
        "- Use only UIDs from the given candidates. Do not invent IDs.",
        "- Merge near‑duplicates: when two UIDs refer to the same file and overlapping or adjacent lines (±2 lines) with similar titles, keep one.",
        "  • Prefer higher severity; then shorter span (more precise); then lower validator_index.",
        "- Exclude LOW unless no HIGH/MEDIUM exist.",
        "- Order selected_ids by severity (HIGH→MEDIUM→LOW), then path, then start line.",
        "- Do not include more than one UID for the same contiguous hunk unless they clearly address different problems.",
        "Intro rules (audience‑first):",
        "- 1–2 sentences. Thank the author and state the next step succinctly.",
        "- Mention scope lightly using candidates/selection (a key file or up to two top‑level directories).",
        "- Do not include links in the intro. When referencing a file, use a backticked path followed by a colon (e.g., ``tvm/runvm.mdx``:) rather than a parenthetical link.",
        "- Calibrate magnitude from the selection: say ‘one inline suggestion’, ‘a couple of suggestions’, or ‘several suggestions’.",
        "- Action orientation: say ‘please apply the inline suggestion(s)’.",
        "- Avoid meta/process words entirely: never say ‘finding(s)’, ‘severity’, ‘selected’, ‘validated’, ‘deduped’, ‘UID’, or ‘PR summary’.",
        "- Do not restate issue details or code in the intro; leave specifics to the inline suggestions.",
        "- Tone: appreciative, direct, and professional. No lists, no headings, no JSON in the intro.",
        "Return JSON only:",
        '{"intro": "1–2 sentences", "selected_ids": ["fid:..."]}',
        "</task>",
    ]
    if extra_instructions:
        parts += [
            "<integrator_instructions>",
            extra_instructions,
            "</integrator_instructions>",
            "",
            "MANDATORY: <integrator_instructions> extend these rules and are required. If any guidance conflicts, treat <integrator_instructions> as the source of truth and resolve the conflict in their favor.",
            "",
        ]
    return "\n".join(parts) + "\n"


# (No host HEAD auto-inclusion; include_branches must be provided by config/CLI.)


def _extract_blocks_by_uid(md_text: str) -> Dict[str, str]:
    """Extract finding blocks keyed by `fid:<uid>` from a validator message.

    Relies on the hidden marker `<!-- fid:... -->` inserted immediately after
    the finding heading. Returns a mapping of uid -> full block text starting at
    the `### [SEVERITY] ...` line up to (but not including) the next `### [`
    or the end of the Findings section.
    """
    if not md_text:
        return {}
    # Find the Findings section boundaries (after '## Findings')
    start_m = re.search(r"^##\s+Findings\b.*$", md_text, flags=re.MULTILINE)
    start_idx = start_m.start() if start_m else 0
    # Collect all markers
    blocks: Dict[str, str] = {}
    for m in re.finditer(r"<!--\s*fid:([A-Za-z0-9:_-]+)\s*-->", md_text):
        raw = m.group(1).strip()
        uid = raw if raw.startswith("fid:") else f"fid:{raw}"
        # Walk backwards to the preceding heading
        before = md_text[: m.start()]
        head_match = list(re.finditer(r"^###\s+\[.*?\].*$", before, flags=re.MULTILINE))
        if not head_match:
            continue
        head = head_match[-1]
        head_idx = head.start()
        if head_idx < start_idx:
            continue
        # End at the next '### [' after marker or end of text
        after = md_text[m.end() :]
        next_head = re.search(r"^###\s+\[.*?\].*$", after, flags=re.MULTILINE)
        end_idx = m.end() + (next_head.start() if next_head else len(after))
        block = md_text[head_idx:end_idx].strip()
        blocks[uid] = block
    return blocks


def _parse_block_detail(block: str) -> Tuple[str, str, str, Dict[str, str]]:
    """Parse a single finding block into (severity, title, desc, suggestion).

    Suggestion object schema: {"kind": "replacement"|"diff"|"none", "code": str}
    If no code fences found under Suggestion:, returns kind="none".
    """
    severity = ""
    title = ""
    desc = ""
    suggestion: Dict[str, str] = {"kind": "none", "code": ""}
    if not block:
        return severity, title, desc, suggestion
    # Heading line
    m_head = re.search(
        r"^###\s+\[(HIGH|MEDIUM|LOW)\]\s+(.*)$", block, flags=re.MULTILINE
    )
    if m_head:
        severity = m_head.group(1).upper().strip()
        title = m_head.group(2).strip()
    # Description section
    m_desc = re.search(
        r"^Description:\s*\n([\s\S]*?)(?:\n\s*Suggestion:\s*\n|\Z)",
        block,
        flags=re.MULTILINE,
    )
    if m_desc:
        desc = m_desc.group(1).strip()
    # Suggestion section
    m_sug = re.search(r"^Suggestion:\s*\n([\s\S]*)$", block, flags=re.MULTILINE)
    sug_text = m_sug.group(1).strip() if m_sug else ""
    if sug_text:
        # Prefer explicit GitHub suggestion fence if present (no +/− allowed)
        m_suggestion = re.search(
            r"```suggestion\n([\s\S]*?)\n```", sug_text, flags=re.IGNORECASE
        )
        if m_suggestion:
            code = m_suggestion.group(1).rstrip("\n")
            suggestion = {"kind": "gh", "code": code}
        else:
            # Fallback: fenced diff/replacement
            m_code = re.search(r"```(diff)?\n([\s\S]*?)\n```", sug_text)
            if m_code:
                lang = (m_code.group(1) or "").lower()
                # Preserve leading whitespace; only strip trailing newlines
                code = m_code.group(2).rstrip("\n")
                suggestion = {
                    "kind": "diff" if lang == "diff" else "replacement",
                    "code": code,
                }
            else:
                # No fence; keep raw text as code replacement if short
                raw = sug_text.strip()
                if raw:
                    suggestion = {"kind": "replacement", "code": raw}
    return severity, title, desc, suggestion


def _build_selected_details(
    *,
    selected_uids: List[str],
    validator_trailers: Dict[int, List[Dict[str, object]]],
    validator_texts: Dict[int, str],
) -> List[Dict[str, object]]:
    """Construct selected_details by stitching validator blocks.

    Chooses, for each uid, the highest-severity candidate across validators,
    tie-breaking by lowest validator index. Falls back to trailer metadata
    when parsing fails.
    """
    # Build candidate map: uid -> list of (idx, trailer_item, block)
    candidates: Dict[str, List[Tuple[int, Dict[str, object], str]]] = {}
    for idx, trailer in validator_trailers.items():
        text = validator_texts.get(idx, "")
        blocks = _extract_blocks_by_uid(text)
        for item in trailer:
            uid = str(item.get("uid", "")).strip()
            if not uid:
                continue
            block = blocks.get(uid, "")
            candidates.setdefault(uid, []).append((idx, item, block))

    sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    details: List[Dict[str, object]] = []
    for uid in selected_uids:
        opts = candidates.get(uid, [])
        if not opts:
            continue

        # Choose by severity (max), then earliest idx
        def _opt_key(t: Tuple[int, Dict[str, object], str]):
            _idx, _item, _block = t
            s = str(_item.get("severity", "")).upper()
            return (sev_order.get(s, 99), _idx)

        idx, item, block = sorted(opts, key=_opt_key)[0]
        sev, title, desc, suggestion = _parse_block_detail(block)
        # Fallbacks from trailer when parsing is incomplete
        path = str(item.get("path", ""))
        start = int(item.get("start", 0) or 0)
        end = int(item.get("end", 0) or 0)
        if not sev:
            sev = str(item.get("severity", "")).upper()
        if not title:
            title = str(item.get("title", ""))
        # Convert diff/replacement into GitHub suggestion content (replacement only)
        gh_code = ""
        kind = suggestion.get("kind", "none")
        code_text = suggestion.get("code", "")
        if kind == "gh":
            # Already a GitHub-ready suggestion block from validator
            gh_code = code_text
        elif kind == "diff":
            lines = []
            for ln in code_text.splitlines():
                # Skip diff headers/context; collect only added lines
                if ln.startswith("+++") or ln.startswith("---") or ln.startswith("@@"):
                    continue
                if ln.startswith("+") and not ln.startswith("+++"):
                    lines.append(ln[1:])
            gh_code = "\n".join(lines).rstrip("\n")
        elif kind == "replacement":
            gh_code = code_text
        # If we still have nothing, keep kind none
        if gh_code:
            # Final sanitation: remove leading diff markers accidentally left in
            gh_lines = []
            for _l in gh_code.splitlines():
                if _l.startswith("+") and not _l.startswith("+++"):
                    gh_lines.append(_l[1:])
                elif _l.startswith("-") and not _l.startswith("---"):
                    # drop accidental removed lines
                    continue
                else:
                    gh_lines.append(_l)
            gh_code = "\n".join(gh_lines).rstrip("\n")
            suggestion = {"kind": "gh", "code": gh_code}
        details.append(
            {
                "uid": uid,
                "path": path,
                "start": start,
                "end": end,
                "severity": sev,
                "title": title,
                "desc": desc,
                "suggestion": suggestion,
            }
        )
    return details


def _resolve_commit_id(
    *,
    repo_path: Path,
    base: str,
    workspace_branch: str,
    include_branch_names: List[str],
) -> str:
    """Return the HEAD commit id for the most likely review branch.

    Picks the first candidate returned by _format_branch_context; falls back to
    HEAD in the local repository if no candidates are found.
    """
    try:
        lines, review_branches = _format_branch_context(
            compare_base=base,
            workspace_branch=workspace_branch,
            include_branch_names=include_branch_names,
        )
        target = review_branches[0] if review_branches else "HEAD"
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", target],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _write_review_sidecar(
    *,
    run_id: str,
    intro: str,
    event: str,
    selected_ids: List[str],
    selected_details: List[Dict[str, object]],
    commit_id: str,
) -> str:
    """Write results/<run_id>/review/index.json and return its relative path."""
    try:
        base_dir = Path("./results") / run_id / "review"
        base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "intro": intro,
            "event": event,
            "selected_ids": selected_ids,
            "selected_details": selected_details,
            "commit_id": commit_id,
        }
        out_path = base_dir / "index.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Return path relative to run results dir
        return f"review/index.json"
    except Exception:
        return ""


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
