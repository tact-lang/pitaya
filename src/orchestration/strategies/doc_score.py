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
    # Verbatim prompt from PROMPT.txt
    prompt = (
        "ROLE\n"
        "You are an AI reviewer for TON Docs with repo read access. Your ONLY task is to SCORE the PAGE named above. Do not list issues or fixes. Do not compute any overall score. Output ONLY the JSON array that validates against the JSON Schema at the end.\n\n"
        "RUNTIME HINTS (model)\n"
        "- If available, set reasoning_effort = \"high\" and ensure ample token budget to avoid truncation. Follow the JSON Schema strictly. (No extra text.)\n\n"
        "STRICT RULES — DO\n"
        "- Read the ENTIRE PAGE first (all headings, paragraphs, lists, tables, code blocks) before scoring. Do not start scoring until you’ve completed a full read.\n"
        "- Apply the SAME depth regardless of length; judge quality, not raw defect counts.\n"
        "- Validate non‑TON facts with general knowledge; you MAY use web search/tools for non‑TON fact checks.\n"
        "- Validate TON‑specific claims ONLY by checking consistency across the docs; do NOT use external web for TON specifics.\n"
        "- For cross‑page judgments (consistency_vs_other_pages and TON consistency), compare with related pages AND also a broad random sample (~10) across the docs to gauge site‑wide norms. Do NOT list sampled pages in output.\n"
        "- Use integer scores 0–4 only.\n"
        "- Provide exactly one short, concrete reason per criterion (plain language, content‑based).\n"
        "- For links_references: you MAY use HTTP HEAD/GET (curl) to external URLs. Consider 2xx/3xx \“reachable,\” 4xx/5xx/timeout = \“broken.\” Internal links must match existing files/anchors.\n\n"
        "STRICT RULES — DO NOT\n"
        "- Do NOT speculate or hedge. Ban: “likely”, “appears”, “seems”, “probably”, “maybe”, “might”, “could be”, “looks like”, “assume”, “guess”.\n"
        "- Do NOT use external web for TON‑specific validation.\n"
        "- Do NOT reason about renderer/deployment behavior beyond verifiable content and link reachability.\n"
        "- Do NOT execute code examples; judge completeness/presence only.\n"
        "- Do NOT output anything besides the JSON array.\n\n"
        "LENGTH‑INVARIANCE & SPARSE‑SIGNAL POLICY\n"
        "- Scores should reflect quality per unit of content. Long pages aren’t penalized for size; short pages aren’t inflated for brevity.\n"
        "- If a criterion truly doesn’t apply, set \"na\": true with a short \"why_na\".\n"
        "- Tutorials/Guides/References: examples are expected. Concept/Overview/Hub: examples may be NA.\n\n"
        "SCORING SCALE (0–4)\n"
        "0 = unusable/wrong; 1 = poor; 2 = mixed; 3 = good (minor nits); 4 = excellent (no meaningful defects)\n\n"
        "CRITERIA (score each 0–4; one-sentence reason)\n"
        "1) correctness_general\n"
        "Non‑TON facts (math, generic APIs/conventions).\n"
        "0: clearly wrong; 1: multiple errors; 2: some unclear/shaky claims; 3: minor imprecision; 4: solid & unambiguous.\n"
        "NA only if the page makes no factual claims.\n\n"
        "2) correctness_ton_consistency\n"
        "TON‑specific claims validated ONLY against other TON Docs pages.\n"
        "NA if: (a) no TON‑specific claims, OR (b) no corroboration found anywhere in the docs.\n"
        "0: direct contradiction elsewhere; 1: several conflicts; 2: drift/weak support; 3: aligns with most of the docs; 4: strongly consistent across the docs or matches a canonical page with no deltas.\n\n"
        "3) examples_quality\n"
        "Snippets/commands appear runnable as written; inputs/env/expected outputs are clear.\n"
        "0: absent/wrong; 1: key snippets unlikely to run as written; 2: runnable but notable steps missing; 3: runnable with minor edits; 4: runnable as written with expected outputs.\n\n"
        "4) writing_clarity_grammar\n"
        "Plain language, correct grammar, avoids unexplained jargon.\n"
        "0: hard to follow; 1: frequent issues; 2: mixed clarity; 3: mostly clear; 4: concise & clean.\n\n"
        "5) consistency_on_page\n"
        "One term per concept; consistent casing/units; uniform code/heading style.\n"
        "0: conflicting terms/units; 1: many mismatches; 2: noticeable drift; 3: minor nits; 4: fully consistent.\n\n"
        "6) consistency_vs_other_pages\n"
        "Alignment with site‑wide norms (terms, tone, structure, versions; expected cross‑links). Sample broadly (include random pages) in addition to related ones; do not list them.\n"
        "0: clear outlier; 1: many mismatches; 2: several drifts; 3: minor deltas; 4: well‑aligned with the broader docs.\n\n"
        "7) structure_ia\n"
        "Logical flow; correct heading ladder; “next steps/refs” present; not an orphan.\n"
        "0: fragmented; 1: weak flow/ladder; 2: OK but choppy; 3: good flow; 4: clear intro→steps→next.\n\n"
        "8) links_references\n"
        "Internal: valid paths/fragments in repo. External: HTTPS recommended; HEAD/GET 2xx/3xx = reachable; 4xx/5xx/timeout = broken.\n"
        "0: broken internal path/fragment or multiple unreachable externals; 1: many malformed/HTTP/unreachable; 2: some malformed/HTTP or inconsistent refs; 3: minor nits; 4: all valid and reachable by checks.\n\n"
        "PROCEDURE\n"
        "1) Fully read the PAGE; then sample across the docs to judge cross‑page criteria.\n"
        "2) Assign 0–4 for each criterion with one‑sentence reasons; mark NA where appropriate.\n"
        "3) Write a file named SCORING_<slug>.json (slug = lowercase, spaces→“‑”, drop non [a‑z0‑9‑]) containing ONLY the JSON array that validates against the schema below. Output the file content only.\n\n"
        "JSON SCHEMA\n"
        "{\n"
        "  \"type\": \"array\",\n"
        "  \"minItems\": 8,\n"
        "  \"maxItems\": 8,\n"
        "  \"items\": {\n"
        "    \"type\": \"object\",\n"
        "    \"additionalProperties\": false,\n"
        "    \"properties\": {\n"
        "      \"id\": {\n"
        "        \"type\": \"string\",\n"
        "        \"enum\": [\n"
        "          \"correctness_general\",\n"
        "          \"correctness_ton_consistency\",\n"
        "          \"examples_quality\",\n"
        "          \"writing_clarity_grammar\",\n"
        "          \"consistency_on_page\",\n"
        "          \"consistency_vs_other_pages\",\n"
        "          \"structure_ia\",\n"
        "          \"links_references\"\n"
        "        ]\n"
        "      },\n"
        "      \"score\": { \"type\": \"integer\", \"minimum\": 0, \"maximum\": 4 },\n"
        "      \"reason\": { \"type\": \"string\", \"minLength\": 5, \"maxLength\": 240 },\n"
        "      \"na\": { \"type\": \"boolean\" },\n"
        "      \"why_na\": { \"type\": \"string\", \"minLength\": 5, \"maxLength\": 200 }\n"
        "    },\n"
        "    \"required\": [\"id\", \"reason\"],\n"
        "    \"allOf\": [\n"
        "      {\n"
        "        \"if\": { \"properties\": { \"na\": { \"const\": true } }, \"required\": [\"na\"] },\n"
        "        \"then\": { \"required\": [\"why_na\"], \"not\": { \"required\": [\"score\"] } }\n"
        "      },\n"
        "      {\n"
        "        \"if\": { \"not\": { \"properties\": { \"na\": { \"const\": true } }, \"required\": [\"na\"] } },\n"
        "        \"then\": { \"required\": [\"score\"] }\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n"
    )
    return target + prompt

