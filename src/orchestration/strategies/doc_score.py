"""
Documentation Scoring Strategy (doc-score)

Simplified, plugin-agnostic strategy that scores documentation pages using
N independent reviewers per page. No validators; results attach parsed scores
to metadata and complete.

Flow per page:
- Spawn reviewers_per_page scoring tasks in parallel (read-only; no commits)
- Each task uses the scoring prompt verbatim and returns a JSON array (0–4)
- Parse JSON, attach to result.metadata["doc_score"], compute simple aggregates
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig
from .doc_review import _load_pages_file, _slugify

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class DocScoreConfig(StrategyConfig):
    """Configuration for documentation scoring strategy."""

    pages_file: str = ""  # YAML or JSON list of pages
    reviewers_per_page: int = 1
    reviewer_max_retries: int = 2

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if not self.pages_file:
            raise ValueError("pages_file is required for doc-score strategy")
        if int(self.reviewers_per_page) < 1:
            raise ValueError("reviewers_per_page must be at least 1")
        if int(self.reviewer_max_retries) < 0:
            raise ValueError("reviewer_max_retries must be >= 0")


class DocScoreStrategy(Strategy):
    @property
    def name(self) -> str:
        return "doc-score"

    def get_config_class(self) -> type[StrategyConfig]:
        return DocScoreConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """Score pages with N reviewers each and attach scores to metadata."""
        cfg: DocScoreConfig = self.create_config()  # type: ignore

        # Parse pages
        repo_path: Path = getattr(ctx._orchestrator, "repo_path", Path.cwd())  # noqa
        pages = _load_pages_file(Path(cfg.pages_file), repo_path)
        if not pages:
            raise ValueError(
                "pages_file parsed but no pages found; expected list of {title, path}"
            )

        # Generate unique slugs
        slug_counts: Dict[str, int] = {}
        normalized_pages: List[Dict[str, Any]] = []
        for p in pages:
            title = str(p.get("title") or "").strip()
            path = str(p.get("path") or "").strip()
            slug = str(p.get("slug") or _slugify(title or path)).strip()
            if not title or not path:
                continue
            base_slug = slug
            i = slug_counts.get(base_slug, 0)
            if i > 0:
                slug = f"{base_slug}-{i+1}"
            slug_counts[base_slug] = i + 1
            normalized_pages.append({"title": title, "path": path, "slug": slug})

        async def _run_scorer_with_retries(
            page: Dict[str, Any], r_idx: int
        ) -> Optional[InstanceResult]:
            attempts = int(cfg.reviewer_max_retries) + 1
            last_result: Optional[InstanceResult] = None
            for i in range(attempts):
                # No corrective guidance — use verbatim prompt; retry if JSON invalid
                s_prompt = _build_scoring_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    page_slug=page["slug"],
                )
                task: Dict[str, Any] = {
                    "prompt": s_prompt,
                    "base_branch": base_branch,
                    "model": cfg.model,
                    # Read-only scoring: never import/commit
                    "import_policy": "never",
                    "skip_empty_import": True,
                }
                key = ctx.key("page", page["slug"], f"r{r_idx}", "score", f"attempt-{i+1}")
                try:
                    h = await ctx.run(task, key=key)
                    result = await ctx.wait(h)
                except Exception:
                    result = None
                if not result:
                    continue
                # Attach page metadata and parsed scores
                _attach_scores_metadata(result, page)
                last_result = result
                # If parse succeeded, return immediately
                if result.metadata and isinstance(result.metadata.get("doc_score"), dict):
                    return result
            return last_result

        # Fan out across all pages and reviewers
        tasks: List[asyncio.Task] = []
        for page in normalized_pages:
            for r_idx in range(1, int(cfg.reviewers_per_page) + 1):
                tasks.append(asyncio.create_task(_run_scorer_with_retries(page, r_idx)))

        results: List[InstanceResult] = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r is not None:
                results.append(r)
        return results


# ---------------------------
# Helpers
# ---------------------------


def _attach_scores_metadata(result: InstanceResult, page: Dict[str, Any]) -> None:
    """Parse final_message JSON and attach to result.metadata and metrics."""
    # Initialize metadata
    if result.metadata is None:
        result.metadata = {}
    result.metadata.setdefault("page_title", page.get("title"))
    result.metadata.setdefault("page_path", page.get("path"))
    result.metadata.setdefault("page_slug", page.get("slug"))

    text = (result.final_message or "").strip()
    scores = _parse_scores_json(text)
    if scores is None:
        # Leave without doc_score; parsing failed
        return
    # Attach structured scores
    result.metadata["doc_score"] = {"scores": scores}
    # Compute simple aggregates
    total = 0
    count = 0
    try:
        for item in scores:
            if isinstance(item, dict) and "score" in item and isinstance(item.get("score"), int):
                total += int(item.get("score"))
                count += 1
    except Exception:
        pass
    avg = (float(total) / float(count)) if count > 0 else None
    if result.metrics is None:
        result.metrics = {}
    result.metrics["doc_score_count"] = count
    if avg is not None:
        result.metrics["doc_score_avg"] = avg


def _parse_scores_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort parse of the scoring JSON array from model output."""
    if not text:
        return None
    # Try full text first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data  # trust schema enforcement in prompt
    except Exception:
        pass
    # Extract first JSON array-looking block
    import re

    # Code block extraction
    m = re.search(r"```(?:json)?\n(\[[\s\S]*?\])\n```", text, flags=re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    # Inline array
    m2 = re.search(r"(\[[\s\S]*\])", text)
    if m2:
        try:
            data = json.loads(m2.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _build_scoring_prompt(*, page_title: str, page_path: str, page_slug: str) -> str:
    target = (
        f"<target>\n"
        f"PAGE TITLE: {page_title}\n"
        f"PAGE PATH: {page_path}\n"
        f"SLUG: {page_slug}\n"
        f"</target>\n\n"
    )
    prompt = """<role>
    You are an AI reviewer for a documentation repository. Your ONLY task is to SCORE the PAGE named above.
    Output ONLY a JSON array that validates against the JSON Schema in &lt;json_schema&gt;, written to a file named:
    SCORING_&lt;slug&gt;.json  (slug = lowercase; spaces→“-”; drop non [a-z0-9-]).
</role>

<output_contract>
    - Write: SCORING_&lt;slug&gt;.json
    - Content: JSON array that VALIDATES against &lt;json_schema&gt;
    - No extra text, logs, or commentary
</output_contract>

<context_understanding>
    - FULL READ FIRST: read the ENTIRE page (headings, paragraphs, lists, tables, code blocks) BEFORE scoring or making any assumptions.
    - Do not form judgments or start checks until the full read is complete.
    - After reading, plan what to verify for each criterion, then perform the checks.
</context_understanding>

<evidence_policy>
    - Judge the CONTENT only.
    - You MAY check external links via HTTP (HEAD first; GET only to confirm a fragment #anchor or when HEAD is inconclusive).
    • Treat 2xx/3xx as reachable; 4xx/5xx as broken; timeouts/429 = unverified (do not penalize for unverified).
    • Limit: ≤10 HTTP requests per page; ≤5s timeout each.
    - You MAY use web search for GENERAL, non‑domain‑specific facts (e.g., math, common computing/web concepts, broadly known APIs). Limit: ≤5 queries; prefer reputable, independent sources; stop when ≥2 independent sources agree.
    - Domain‑specific facts (defined/owned by this documentation set): DO NOT web‑search; validate ONLY by consistency within this repo.
</evidence_policy>

<verifiability_and_style>
    - Reasons must be fully verifiable from repo content or allowed checks.
    - NO hedging words: do not use “likely”, “appears”, “seems”, “probably”, “maybe”, “might”, “could be”, “looks like”, “assume”, “guess”.
    - If something cannot be verified, either use NA (when allowed) or choose a conservative lower score and state the concrete uncertainty (e.g., “term undefined in repo”).
</verifiability_and_style>

<length_invariance>
    - Score quality per unit of content; long pages are NOT penalized for length.
    - Apply the same review depth to short and long pages.
    - When a criterion truly does not apply, set "na": true and provide "why_na".
</length_invariance>

<scoring_scale>
    - Use integers 0–5 only:
    0 = unusable / wrong
    1 = poor (material problems)
    2 = weak / mixed (several issues)
    3 = acceptable (minor issues)
    4 = good (small, non‑blocking nits)
    5 = excellent (no meaningful defects)
</scoring_scale>

<criteria>
    <criterion id="correctness_general">
    <what_it_is>
        Accuracy of GENERAL knowledge statements: mathematics; common computing and web concepts; broadly standardized protocols/APIs; conventional data formats/units.
        Examples: arithmetic, percentage math, HTTP status class meanings, JSON shape semantics (generic), file path conventions, timestamp formats.
        Excludes: project‑specific semantics, custom APIs/terms owned by this docs set (covered by correctness_domain_consistency).
    </what_it_is>
    <how_to_check>
        1) Extract explicit factual statements (definitions, formulas, standard behaviors).
        2) Validate from your general knowledge; when uncertain, use web search within limits until ≥2 independent reputable sources agree.
        3) Judge severity by whether a reasonable reader would be misled; evaluate density (quality per unit text), not raw counts.
    </how_to_check>
    <scoring_anchors>
        0: Multiple clear factual errors that would mislead usage or understanding.
        1: One severe or several material mistakes; reader likely to act incorrectly.
        2: Several unclear/unsupported claims or minor errors; usable with caution.
        3: Generally accurate; at most minor imprecision/phrasing nits.
        4: Accurate and precise; terminology and numbers correct; definitions clear.
        5: Flawless general accuracy; precise wording; nothing to correct.
    </scoring_anchors>
    <na_rules>NA only if the page presents no factual claims (e.g., a pure navigation index).</na_rules>
    </criterion>

    <criterion id="correctness_domain_consistency">
    <what_it_is>
        Accuracy of DOMAIN‑SPECIFIC claims defined by this documentation set (its concepts, APIs, parameters, formats, workflows, constraints).
        Examples: endpoint names/paths, parameter types, domain terms, canonical flows, versioning rules specific to this project.
        Validation source: the repository itself (canonical guides, references, tutorials). External web search is NOT allowed.
    </what_it_is>
    <how_to_check>
        1) Extract domain‑specific claims (terms/fields/flows unique to this project).
        2) Cross‑check against canonical/overview/reference pages and thematically related guides; ALSO sample a small broad set across the repo (≈10–15) to establish house norms.
        3) If the page has no domain‑specific claims OR no corroboration exists anywhere in the repo, set NA (do not guess).
    </how_to_check>
    <scoring_anchors>
        NA: No domain‑specific claims, or no corroboration anywhere in repo.
        0: Direct contradiction on critical behaviors/definitions.
        1: Multiple conflicts on key points; reader likely misled.
        2: Noticeable drift/weak support; some claims conflict or lack confirmations.
        3: Mostly aligned; minor wording/coverage deltas; no critical contradictions.
        4: Strong alignment across canonical/related pages; wording and semantics match.
        5: Fully aligned with canonical statements across the repo; precise, internally referenced.
    </scoring_anchors>
    </criterion>

    <criterion id="examples_quality">
    <what_it_is>
        Completeness and apparent runnability of examples/snippets/commands as SPECIFIED BY THE TEXT (without executing).
        The reader should be able to reproduce steps using stated prerequisites, inputs, and expected outputs.
        Includes: prerequisites/tools/versions; imports/flags; clear inputs; ordered steps; expected outputs or verification step; explanation of placeholders; code fences with language.
    </what_it_is>
    <how_to_check>
        1) Identify each example or command block and its surrounding instructions.
        2) Check that all prerequisites and inputs are stated; steps are logically ordered; expected outputs or verification steps are present.
        3) Judge on completeness and clarity (not on runtime).
    </how_to_check>
    <scoring_anchors>
        0: Examples absent or clearly wrong/misleading (missing core steps or contradictory instructions).
        1: Key examples incomplete; crucial prerequisites/inputs missing; reproduction unlikely.
        2: Runnable in principle but important steps/inputs/expected outputs are missing or ambiguous.
        3: Runnable with small edits (e.g., one env var implied or a minor install step); outputs mostly clear.
        4: Runnable as written; prerequisites and outputs are explicit; steps minimal and ordered.
        5: Exemplary: end‑to‑end flow, variants/edge notes, explicit verification; copy‑paste friendly.
    </scoring_anchors>
    <na_rules>
        NA is allowed ONLY when the page genre does not reasonably require examples (e.g., Concept/Overview/Hub).
        If examples are present OR the page is a Tutorial/Guide/Reference, you MUST score (do not mark NA).
    </na_rules>
    </criterion>

    <criterion id="writing_clarity_grammar">
    <what_it_is>
        Readability and grammatical correctness: clear sentences; proper punctuation; consistent person/voice; minimal jargon and all jargon defined on first use; coherent paragraph flow; headings reflect content.
    </what_it_is>
    <how_to_check>
        1) Scan for typos, agreement errors, punctuation misuse, run‑ons, and ambiguous pronouns.
        2) Check sentence length distribution (avoid long multi‑clause chains); prefer active voice where practical; maintain consistent second‑person (“you”) or neutral voice within a page.
        3) Ensure terms are defined on first use or linked to a definition.
    </how_to_check>
    <scoring_anchors>
        0: Very hard to follow; pervasive grammar/style issues; meaning frequently unclear.
        1: Frequent errors and awkward phrasing; reader struggles to extract steps/meaning.
        2: Mixed clarity with recurring issues (run‑ons, unexplained jargon); requires effort.
        3: Mostly clear; a few minor issues; paragraphs flow acceptably.
        4: Clear and concise; polished phrasing; definitions provided on first use; minimal nits.
        5: Exceptional clarity and polish; consistent, professional tone; no detectable errors.
    </scoring_anchors>
    <na_rules>Never NA (always applicable).</na_rules>
    </criterion>

    <criterion id="consistency_on_page">
    <what_it_is>
        Internal consistency WITHIN THIS PAGE: one term per concept; consistent casing (Title vs sentence case); uniform units/date formats; stable code style (fence language tags, indentation, quotes); consistent heading style and admonition format; consistent link text conventions.
    </what_it_is>
    <how_to_check>
        1) List the key terms and verify they are used uniformly.
        2) Check units (e.g., KB/MB), number formatting (decimal/grouping), and date style across sections.
        3) Confirm code blocks have language tags and consistent style; headings follow one casing style; the same type of note uses the same callout format.
    </how_to_check>
    <scoring_anchors>
        0: Conflicting terms/units/casing across sections; inconsistent code/heading styles.
        1: Many mismatches; conventions vary often; reader likely confused.
        2: Several drifts; noticeable flip‑flops in terms or style.
        3: Minor nits; overall consistent.
        4: Fully consistent with only trivial slips.
        5: Impeccably consistent; reads as one coherent voice/style.
    </scoring_anchors>
    <na_rules>Never NA (always applicable).</na_rules>
    </criterion>

    <criterion id="consistency_vs_other_pages">
    <what_it_is>
        Alignment with site‑wide norms across the repository: shared terms, tone, structure patterns, versioning conventions, and expected cross‑links to canonical pages.
        Comparison set: thematically related pages AND a small random sample across the repo (≈10–15) to avoid local bias. Do NOT list sampled pages in output.
    </what_it_is>
    <how_to_check>
        1) Compare term choices to glossary/overview pages; check tone (instructional second‑person vs narrative), and common structure patterns (Overview → Prereqs → Steps → Verify → Next).
        2) Check that versions/SDK names aren’t unexplained outliers; verify expected cross‑links to canonical topics are present when the topic is discussed.
        3) Evaluate alignment at the pattern level (not identical wording).
    </how_to_check>
    <scoring_anchors>
        0: Clear outlier on multiple dimensions; contradicts site norms.
        1: Many mismatches; tone/terms/structure diverge noticeably.
        2: Several drifts; partially aligned but uneven.
        3: Minor deltas; broadly aligned.
        4: Well aligned; reinforces established patterns.
        5: Model alignment; could serve as a template for others.
    </scoring_anchors>
    <na_rules>
        NA only if this repository effectively contains no other substantive pages to compare (rare). Otherwise, MUST score.
    </na_rules>
    </criterion>

    <criterion id="structure_ia">
    <what_it_is>
        Information architecture and logical flow: correct heading ladder (no level skips), ordered sections, presence of prerequisites (when relevant), verification/expected results (when relevant), and “Next steps/References”; page is not obviously orphaned in available nav/index files.
    </what_it_is>
    <how_to_check>
        1) Verify heading levels are not skipped (e.g., H2 must follow H1 before H3).
        2) Check that sections follow logical order (Intro → Prereqs → Steps → Verify → Next) when applicable to the genre.
        3) If nav manifests exist (e.g., SUMMARY.md, mkdocs.yml, sidebars.*), confirm inclusion; otherwise, do not penalize orphan status.
    </how_to_check>
    <scoring_anchors>
        0: Fragmented/illogical; ladder misused; no onward paths.
        1: Weak flow or repeated ladder problems; missing prereqs/verification where expected.
        2: Acceptable but choppy or one major section misplaced/missing.
        3: Good flow; small gaps only; headings mostly correct.
        4: Clear intro→steps→verify→next; headings correct throughout.
        5: Exemplary IA; highly scannable; robust onward paths; clearly placed in nav.
    </scoring_anchors>
    <na_rules>
        NA allowed for tiny notices/changelogs where a full ladder is not meaningful.
    </na_rules>
    </criterion>

    <criterion id="links_references">
    <what_it_is>
        Validity and quality of internal/external references and links. INTERNAL links must resolve to existing files/anchors. EXTERNAL links may be verified via HTTP.
        Evaluate link scheme (prefer HTTPS), anchor precision, and reference usefulness.
    </what_it_is>
    <how_to_check>
        1) Internal: confirm target file exists and fragment matches an anchor/heading id; relative paths resolve.
        2) External: HEAD first; GET only to validate a fragment or when HEAD is inconclusive. 2xx/3xx = reachable; 4xx/5xx = broken; timeouts/429 = unverified (neutral).
        3) Prefer HTTPS; count plain HTTP as a minor issue unless required by the target.
    </how_to_check>
    <scoring_anchors>
        0: Broken internal paths/anchors; many bad externals.
        1: Several malformed/HTTP/bad links; weak references.
        2: Some malformed/HTTP links or inconsistent citations; overall usable.
        3: Minor nits (e.g., occasional redirect or formatting quirk).
        4: All links valid/stable; references are helpful and relevant.
        5: Pristine: anchor‑precise; consistently HTTPS; references elevate comprehension.
    </scoring_anchors>
    <na_rules>
        NA allowed only if the page contains no links/references and its genre does not reasonably require them (e.g., a very short concept definition).
        If links exist OR genre implies they are expected (tutorial/reference), you MUST score (not NA).
    </na_rules>
    </criterion>
</criteria>

<procedure>
    1) Perform FULL READ of the page.
    2) Plan checks per criterion; then perform repo reads, allowed HTTP checks, and (for general facts only) web search within budgets.
    3) Assign 0–5 for each criterion using anchors; use NA strictly per &lt;na_rules&gt;.
    4) Produce ONLY the JSON array defined in &lt;json_schema&gt; and write SCORING_&lt;slug&gt;.json. Stop.
</procedure>

<json_schema>
{
    "type": "array",
    "minItems": 8,
    "maxItems": 8,
    "items": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
        "id": {
        "type": "string",
        "enum": [
            "correctness_general",
            "correctness_domain_consistency",
            "examples_quality",
            "writing_clarity_grammar",
            "consistency_on_page",
            "consistency_vs_other_pages",
            "structure_ia",
            "links_references"
        ]
        },
        "score": { "type": "integer", "minimum": 0, "maximum": 5 },
        "na": { "type": "boolean" },
        "why_na": { "type": "string", "minLength": 5, "maxLength": 400 },
        "reason": { "type": "string", "minLength": 8, "maxLength": 400 }
    },
    "required": ["id", "reason"]
    }
}
</json_schema>"""

    return target + prompt
