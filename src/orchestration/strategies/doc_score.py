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
    prompt = """ROLE
You are an AI reviewer for a documentation repository. Your ONLY task is to SCORE the PAGE named above. Do not list issues or fixes. Do not compute any overall score. Output ONLY the JSON array defined by the JSON Schema at the end, written to a file named SCORING_<slug>.json (slug = lowercase; spaces→“-”; drop non [a‑z0‑9‑]).

OUTPUT CONTRACT
- File: SCORING_<slug>.json
- Content: a JSON array that VALIDATES against the schema at the end.
- Absolutely no extra text, logs, or commentary.

WORKFLOW (deterministic)
1) Full read FIRST: read the ENTIRE page end‑to‑end (all headings, paragraphs, lists, tables, code blocks) before scoring or calling any tools. Do not score from partial reads.
2) Plan checks: for each criterion below, plan what you must verify from CONTENT only (and the allowed tools).
3) Tool policy (allowed + budgets):
    • External link checks: you MAY use HTTP requests (curl). Use HEAD first; use GET only when needed to verify a fragment (#anchor). Timeouts ≤5s; ≤10 total requests per page.
    • General fact checks (non‑domain‑specific): you MAY use web search. ≤5 queries per page; prefer reputable sources; stop when ≥2 independent sources agree.
    • Domain‑specific facts (defined by/unique to this documentation set): DO NOT web‑search; validate ONLY by consistency within this repo.
4) Score: assign an integer 0–5 per criterion using the anchors. Write ONE concise, content‑based reason. If a criterion truly does not apply, set "na": true and add "why_na".
5) Determinism: avoid randomness. For “consistency vs other pages,” compare with closely related pages AND a small broad sample across the repo; keep total sampled pages reasonable (≈10–15). Do NOT list sampled pages in output.

STRICT RULES — DO
- Read the whole file before judging anything.
- Keep depth the same regardless of page length; judge quality, not raw counts.
- Use integer scores 0–5 only.
- Keep reasons short, specific, and verifiable from repo content and allowed tools.

STRICT RULES — DO NOT
- Do NOT guess or speculate. If you cannot verify, choose a conservative lower score or use NA when allowed.
- Do NOT use hedging words: “likely”, “appears”, “seems”, “probably”, “maybe”, “might”, “could be”, “looks like”, “assume”, “guess”.
- Do NOT reason about runtime/site renderer quirks; judge the CONTENT and what your allowed tools can check.
- Do NOT execute code or commands beyond web search and HTTP checks.
- Do NOT output anything besides the JSON array.

SCORING SCALE (0–5)
0 = unusable/wrong; 1 = poor; 2 = weak/mixed; 3 = acceptable (minor issues); 4 = good (small nits only); 5 = excellent (no meaningful defects)

CRITERIA (each 0–5; reasons = one sentence, no hedging)
1) correctness_general
What it is: factual accuracy of GENERAL knowledge (math, common computing/web concepts, broadly known APIs/protocols). You MAY use web search here.
0: multiple clear factual errors; 1: several errors/misstatements; 2: some unclear or unsupported claims; 3: minor imprecision only; 4: accurate with clear phrasing; 5: fully accurate, precise, and unambiguous.

2) correctness_domain_consistency
What it is: accuracy of DOMAIN‑SPECIFIC claims defined by this documentation set (its APIs, terms, formats). Validate ONLY by consistency across the docs (no web search).
NA if: the page makes no domain‑specific claims OR such claims cannot be corroborated anywhere in the docs.
0: direct contradiction with other pages; 1: several conflicts; 2: noticeable drift/weak support; 3: mostly aligned, minor deltas; 4: strongly consistent across related pages; 5: fully aligned with canonical/overview pages with zero deltas.

3) examples_quality
What it is: completeness and apparent runnability of examples/snippets/commands from the text (without executing). Inputs/env/expected output are stated; steps are ordered.
0: examples absent or clearly wrong; 1: key examples incomplete or misleading; 2: runnable in principle but important steps/inputs missing; 3: runnable with small edits; 4: runnable as written with clear inputs and outputs; 5: exemplary: minimal setup, end‑to‑end clarity, edge cases noted.

4) writing_clarity_grammar
What it is: readability, grammar, and avoidance of unexplained jargon.
0: very hard to follow; 1: frequent grammar/style issues; 2: mixed clarity with recurring issues; 3: mostly clear; few minor issues; 4: clear and concise; 5: exceptionally clear, consistent tone, no errors.

5) consistency_on_page
What it is: internal consistency of terms, units, casing, code style, headings. One term per concept.
0: conflicting terms/units/casing; 1: many mismatches; 2: several drifts; 3: minor nits; 4: consistent with rare trivial nits; 5: fully uniform and polished.

6) consistency_vs_other_pages
What it is: alignment with site‑wide norms (terms, tone, structure patterns, versioning, expected cross‑links). Compare with related pages AND a broad sample.
0: clear outlier; 1: many mismatches; 2: several drifts; 3: minor deltas; 4: well aligned overall; 5: exemplary alignment and reinforces shared patterns.

7) structure_ia
What it is: logical flow, correct heading ladder, presence of prerequisites/next steps/reference links; page is not an orphan in nav.
0: fragmented/illogical; 1: weak flow or heading misuse; 2: acceptable but choppy or missing key sections; 3: good flow with minor gaps; 4: clear intro→steps→next with solid nav; 5: exemplary structure, scannable with robust onward paths.

8) links_references
What it is: validity and quality of references and links. INTERNAL links: verify paths and fragments. EXTERNAL links: you MAY use curl HEAD/GET.
Rules: internal = must resolve to existing files/anchors; external = prefer HTTPS; treat 2xx/3xx as reachable; 4xx/5xx as broken; timeouts/429 = do not penalize (state “unverified”).
0: broken internals or many bad externals; 1: several malformed/HTTP/bad links; 2: some malformed/HTTP or inconsistent citations; 3: minor nits (e.g., occasional redirect); 4: all valid and stable; 5: pristine, with helpful, well‑chosen references.

JSON SCHEMA (validate the file content against this schema)
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
        "why_na": { "type": "string", "minLength": 5, "maxLength": 240 },
        "reason": { "type": "string", "minLength": 5, "maxLength": 300 }
    },
    "required": ["id", "reason"],
    "allOf": [
        {
        "if": { "properties": { "na": { "const": true } }, "required": ["na"] },
        "then": { "required": ["why_na"], "not": { "required": ["score"] } }
        },
        {
        "if": { "not": { "properties": { "na": { "const": true } }, "required": ["na"] } },
        "then": { "required": ["score"] }
        }
    ]
    }
}"""

    return target + prompt
