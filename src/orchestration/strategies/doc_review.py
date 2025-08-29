"""
Documentation Review Strategy

Minimal, plugin-agnostic multi-stage strategy for technical documentation review.

Stages:
1) Reviewers (N per page): each produces a simple structured report at
   reports/doc-review/raw/REPORT_{slug}__r{n}.md and commits once.
2) Validators (streamed, 1:1): run on each reviewer branch, strictly validate
   and refine the report in place, committing once.
Note: The composer stage has been removed. This strategy returns the
successful validator results; downstream tooling can aggregate reports if needed.

Minimal configuration:
- pages_file (YAML or JSON): list of {title, path, slug?}
- reviewers_per_page: default 1
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import yaml

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class DocReviewConfig(StrategyConfig):
    """Configuration for documentation review strategy."""

    pages_file: str = ""  # YAML or JSON list of pages
    reviewers_per_page: int = 1
    report_dir: str = "reports/doc-review"
    # Unified retry controls (retries = additional attempts after the first)
    reviewer_max_retries: int = 3
    validator_max_retries: int = 3

    def validate(self) -> None:
        super().validate()
        if not self.pages_file:
            raise ValueError("pages_file is required for doc-review strategy")
        if self.reviewers_per_page < 1:
            raise ValueError("reviewers_per_page must be at least 1")
        if self.reviewer_max_retries < 0:
            raise ValueError("reviewer_max_retries must be >= 0")
        if self.validator_max_retries < 0:
            raise ValueError("validator_max_retries must be >= 0")
        # no composer


class DocReviewStrategy(Strategy):
    @property
    def name(self) -> str:
        return "doc-review"

    def get_config_class(self) -> type[StrategyConfig]:
        return DocReviewConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        cfg: DocReviewConfig = self.create_config()  # type: ignore

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
            # ensure uniqueness
            base_slug = slug
            i = slug_counts.get(base_slug, 0)
            if i > 0:
                slug = f"{base_slug}-{i+1}"
            slug_counts[base_slug] = i + 1
            normalized_pages.append({"title": title, "path": path, "slug": slug})

        # Make sure report_dir/raw exists in repo (not required, but helps agents)
        try:
            (repo_path / cfg.report_dir / "raw").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Unified helpers ---------------------------------------------------

        async def _run_reviewer_with_retries(
            page: Dict[str, Any], r_idx: int
        ) -> Optional[InstanceResult]:
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            attempts = cfg.reviewer_max_retries + 1
            last_result: Optional[InstanceResult] = None
            for i in range(attempts):
                corrective = i > 0
                review_prompt = _build_reviewer_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    report_path=report_rel,
                )
                task = {
                    "prompt": review_prompt,
                    "base_branch": base_branch,
                    "model": cfg.model,
                }
                key_parts = ["page", page["slug"], f"r{r_idx}", "review"]
                if corrective:
                    key_parts.append(f"attempt-{i}")
                key = ctx.key(*key_parts)
                try:
                    h = await ctx.run(task, key=key)
                    result = await ctx.wait(h)
                except Exception:
                    result = None  # Treat as failed attempt, continue
                if result:
                    last_result = result
                    ok = _verify_file_in_branch(
                        repo_path, result.branch_name, report_rel
                    )
                    if result.success and result.branch_name and ok:
                        return result
            return last_result if (last_result and last_result.success) else None

        async def _run_validator_with_retries(
            page: Dict[str, Any], r_idx: int, reviewer_branch: str
        ) -> Optional[InstanceResult]:
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            attempts = cfg.validator_max_retries + 1
            last_vres: Optional[InstanceResult] = None
            for i in range(attempts):
                corrective = i > 0
                validator_prompt = _build_validator_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    report_path=report_rel,
                )
                v_task = {
                    "prompt": validator_prompt,
                    "base_branch": reviewer_branch,
                    "model": cfg.model,
                }
                v_key_parts = ["page", page["slug"], f"r{r_idx}", "validate"]
                if corrective:
                    v_key_parts.append(f"attempt-{i}")
                v_key = ctx.key(*v_key_parts)
                try:
                    vh = await ctx.run(v_task, key=v_key)
                    vres = await ctx.wait(vh)
                except Exception:
                    vres = None
                if vres:
                    last_vres = vres
                    ok = _verify_file_in_branch(repo_path, vres.branch_name, report_rel)
                    if vres.success and vres.branch_name and ok:
                        return vres
            return last_vres if (last_vres and last_vres.success) else None

        # (composer stage removed)

        # Stage 1+2: Run reviewers with retries and chain validators with retries
        async def _review_then_validate(
            page: Dict[str, Any], r_idx: int
        ) -> Optional[Tuple[Dict[str, Any], int, InstanceResult]]:
            rres = await _run_reviewer_with_retries(page, r_idx)
            if not (rres and rres.success and rres.branch_name):
                return None
            vres = await _run_validator_with_retries(page, r_idx, rres.branch_name)
            if not (vres and vres.success and vres.branch_name):
                return None
            return (page, r_idx, vres)

        tasks = []
        for page in normalized_pages:
            for r_idx in range(1, int(cfg.reviewers_per_page) + 1):
                tasks.append(asyncio.create_task(_review_then_validate(page, r_idx)))

        validated_reports: List[Tuple[Dict[str, Any], int, InstanceResult]] = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r is not None:
                validated_reports.append(r)

        # Composer removed: return successful validator results directly
        return [vres for (_page, _idx, vres) in validated_reports]


# ---------------------------
# Helpers
# ---------------------------


def _load_pages_file(p: Path, repo_path: Path) -> List[Dict[str, Any]]:
    """Load pages from YAML or JSON file.

    Expands '~' and environment variables in the pages_file path. If the path is
    relative and the file exists under the repository root, that location is used.

    Returns a list of dicts with at least {title, path}. Relative paths are
    expected to be repo-relative.
    """
    # Resolve user/env expansions and prefer repo-relative when applicable
    try:
        import os

        p_expanded = Path(os.path.expanduser(os.path.expandvars(str(p))))
        if not p_expanded.is_absolute():
            candidate = (repo_path / p_expanded).resolve()
            if candidate.exists():
                p_expanded = candidate
        raw = p_expanded.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Could not read pages_file: {p}: {e}")

    pages: List[Dict[str, Any]] = []
    try:
        if p_expanded.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        elif p_expanded.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            # Try YAML first, then JSON
            try:
                data = yaml.safe_load(raw)
            except Exception:
                data = json.loads(raw)
        if isinstance(data, dict) and "pages" in data:
            data = data["pages"]
        if not isinstance(data, list):
            raise ValueError("pages_file must contain a list of pages")
        for item in data:
            if not isinstance(item, dict):
                # Skip non-dict entries for minimal version
                continue
            title = str(item.get("title") or "").strip()
            path = str(item.get("path") or "").strip()
            if not title or not path:
                continue
            # Normalize to repo relative
            path_obj = Path(path)
            if path_obj.is_absolute():
                try:
                    path = str(path_obj.relative_to(repo_path))
                except Exception:
                    path = str(path_obj)
            pages.append(
                {
                    "title": title,
                    "path": path,
                    **({"slug": item.get("slug")} if item.get("slug") else {}),
                }
            )
    except Exception as e:
        raise ValueError(f"Failed parsing pages_file: {e}")
    return pages


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "page"


def _verify_file_in_branch(repo: Path, branch: Optional[str], rel_path: str) -> bool:
    if not branch:
        return False
    try:
        cmd = ["git", "-C", str(repo), "show", f"{branch}:{rel_path}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout)
    except Exception:
        return False


def _build_reviewer_prompt(
    *,
    page_title: str,
    page_path: str,
    report_path: str,
) -> str:

    return (
        f"<role>\n"
        f"You are a meticulous technical documentation reviewer with access to this repository's documentation. "
        f"Audit the target page with extreme care: read it end-to-end with line-by-line attention. "
        f"Your job is to surface user-facing defects and propose surgical, minimal fixes without inventing domain facts.\n"
        f"</role>\n\n"
        f"<target>\n"
        f"TARGET PAGE TITLE: {page_title}\n"
        f"TARGET PAGE PATH: {page_path}\n"
        f"OUTPUT FILE (required): {report_path}\n"
        f"</target>\n\n"
        f"<scope>\n"
        f"Only consider user-facing content that appears to readers on the rendered page: prose, headings, lists, tables, notes/callouts, code blocks and examples, commands, API shapes shown in examples, configuration snippets, parameters and defaults, return codes, images/diagrams/captions/alt text, links/anchors/cross-references, and inline UI labels.\n"
        f"Out of scope: repository internals, build/CI/tooling, docs engine internals, hidden frontmatter not rendered, source code quality of the product, and anything else not visible to readers. Do not comment on these.\n"
        f"</scope>\n\n"
        f"<truth_sources>\n"
        f"Domain-specific fact checks must use only this documentation corpus. Never assume or import domain facts from memory or the web. "
        f"The glossary is canonical where it exists; if absent or silent, prefer consistent usage across other pages of the docs. "
        f"General knowledge (outside the domain) may be used for basics like HTTP semantics, JSON/YAML well-formedness, Markdown rules, shell quoting rules, and language syntax for static validation; when used, say so briefly in the description.\n"
        f"</truth_sources>\n\n"
        f"<style_of_issues>\n"
        f"Focus on the style of issues that degrade a reader’s ability to succeed:\n"
        f"- Truthfulness and internal consistency: every claim, parameter, default, value range, return code, and constraint must match the documentation corpus.\n"
        f"- Terminology discipline: terms, entity names, and capitalization must align with the glossary or prevailing usage; avoid synonyms that create ambiguity.\n"
        f"- Reproducibility of procedures: steps must be complete, ordered, and free of hidden prerequisites; commands and examples must be runnable with placeholders clearly marked.\n"
        f"- Example viability: code blocks must be labeled with the correct language; configurations must be syntactically valid; HTTP methods/status codes must be plausible.\n"
        f"- Navigability: headings hierarchy must be coherent; links/anchors must resolve; cross-references must point to the most relevant canonical location.\n"
        f"- Clarity and tone: concise, imperative, unambiguous instructions; avoid weasel words, undefined acronyms, and speculative language.\n"
        f"- Formatting semantics: Markdown constructs must be used correctly; avoid broken lists, malformed tables, mis-nesting, or inconsistent callout styles.\n"
        f"</style_of_issues>\n\n"
        f"<workflow>\n"
        f"1) Full-read: Read the entire target page end-to-end before judging anything. Identify all claims, terms, procedures, examples, and links.\n"
        f"2) Context pass: Skim user-facing overview materials inside the docs (README/Introduction/Getting Started/Index and the Glossary). "
        f"Use them only to understand the project and to check terminology; do not import external sources for domain facts.\n"
        f"3) Plan: Draft a concise internal plan for this review (do not output the plan): what you will validate on this page - terminology against glossary/usage, cross-page consistency of claims, examples/commands, link/anchor integrity, structure/order of steps, clarity/scope boundaries.\n"
        f"4) Execute checks:\n"
        f"   - Cross-doc verification: for each factual claim, confirm alignment with the corpus. If the corpus is silent, mark the item as needing a domain decision; do not invent facts.\n"
        f"   - Examples/commands/configs: run a static sanity check (language tag correctness, basic syntax/format validity, placeholder conventions, flags/keys present, HTTP method/status plausibility).\n"
        f"   - Links/anchors: verify existence and that link text matches the target heading; ensure See also items are relevant and canonical.\n"
        f"   - Language/tone/structure: ensure procedural voice, remove ambiguity, verify heading levels are logical and consistent.\n"
        f"5) Fix proposals: For each defect, propose the minimal change to make the page correct and clear. Do not introduce new domain facts; when a new fact would be required, write that the item needs confirmation from the domain owner.\n"
        f"6) Report: Write findings to the required file using the exact report format, then commit that single file once.\n"
        f"</workflow>\n\n"
        f"<evidence_and_citations>\n"
        f"When citing support or conflicts, reference other documentation pages by repo-relative path with anchors to headings. "
        f"If a finding rests on general knowledge, state this in one short sentence within the description. "
        f"Do not fabricate citations or external references.\n"
        f"</evidence_and_citations>\n\n"
        f"<report_format>\n"
        f"Keep this exact format for every finding. No headers, no preamble, no sections beyond the list items. Number sequentially.\n"
        f"- [ ] **<number>. <Short Precise Title>**\n\n"
        f"<repo-relative path or stable URL with optional line anchors>\n\n"
        f"<Detailed description of the issue and the correct version or fix. Include cross-doc citations as repo-relative paths with anchors. "
        f"If you relied on general knowledge for a basic rule, say so in one short sentence. If a domain decision is required, state it explicitly.>\n\n"
        f"---\n\n"
        f"</report_format>\n\n"
        f"<constraints>\n"
        f"- Produce only the report at the specified path.\n"
        f"- Stage and commit exactly once. Do not modify the target page or any other files.\n"
        f"- Do not expand the scope beyond the single target page.\n"
        f"</constraints>\n\n"
        f"<persistence>\n"
        f"You are an agent. Keep going until the page is fully reviewed, the report is written in the exact format, and the single commit is made. "
        f"Do not ask for clarification; proceed with the most reasonable conservative interpretation and document any assumptions in the report items.\n"
        f"</persistence>\n\n"
        f"<tool_preambles>\n"
        f"Begin by restating your goal succinctly to yourself and outlining the steps you will take. "
        f"As you read and verify, avoid redundant searches; prefer targeted queries within the docs. "
        f"Finish by summarizing actions completed before you write and commit the report file.\n"
        f"</tool_preambles>\n\n"
        f"<stop_conditions>\n"
        f"Stop only after: the full page has been read; checks executed; all user-visible defects captured; the report file exists with the exact format; "
        f"numbering is continuous; the file is added; and exactly one commit has been made.\n"
        f"</stop_conditions>\n\n"
        f"<commit_requirements>\n"
        f"CRITICAL: This run is considered failed unless you add and commit the report file. "
        f"Uncommitted workspace changes are ignored by orchestration.\n"
        f"- Required file: {report_path}\n"
        f'- Commit exactly once after writing: git add {report_path} && git commit -m "doc-review: add {Path(report_path).name}"\n'
        f"</commit_requirements>\n"
    )


def _build_validator_prompt(
    *, page_title: str, page_path: str, report_path: str
) -> str:

    return (
        f"<role>\n"
        f"You are validating and refining a documentation review report for a single target page. "
        f"Your job is to verify every finding cautiously against the documentation corpus, remove anything unsubstantiated or out of scope, and improve clarity while preserving the required output format.\n"
        f"</role>\n\n"
        f"<target>\n"
        f"TARGET PAGE TITLE: {page_title}\n"
        f"TARGET PAGE PATH: {page_path}\n"
        f"REPORT TO VALIDATE (required): {report_path}\n"
        f"</target>\n\n"
        f"<scope>\n"
        f"Keep scope user-facing only. Do not add or validate findings about repository internals, build/CI, docs engine internals, or anything not visible to readers.\n"
        f"</scope>\n\n"
        f"<truth_sources>\n"
        f"Domain-specific verification must use only this documentation corpus. The glossary is canonical where it exists; "
        f"if a term is not in the glossary or no glossary exists, compare usage across the docs and prefer the dominant/most recent convention. "
        f"General knowledge is permitted only for basics like HTTP semantics, JSON/YAML validity, Markdown rules, shell quoting, and language syntax for static checks; "
        f"when a finding relies on this, ensure the description states it briefly.\n"
        f"</truth_sources>\n\n"
        f"<workflow>\n"
        f"1) Read the entire target page fully. Then skim user-facing overview materials (README/Introduction/Getting Started/Index and the Glossary) to refresh context.\n"
        f"2) Open the report file and iterate through each finding in order.\n"
        f"3) For each finding:\n"
        f"   - Verify correctness against the target page and the documentation corpus.\n"
        f"   - Remove findings that lack sufficient evidence, are speculative, introduce new domain facts, or are out of scope.\n"
        f"   - Merge duplicates; keep the clearest, most conservative version.\n"
        f"   - Refine language for precision and actionability while remaining concise.\n"
        f"   - Ensure cross-doc citations are repo-relative with anchors when applicable.\n"
        f"   - If the issue hinges on a domain decision, state that explicitly in the description; do not invent a decision.\n"
        f"4) Maintain the exact output format and fix numbering to be continuous.\n"
        f"5) Save the refined report, stage it, and make a single commit.\n"
        f"</workflow>\n\n"
        f"<report_format>\n"
        f"Keep this exact format:\n"
        f"- [ ] **<number>. <Short Precise Title>**\n\n"
        f"<repo-relative path or stable URL with optional line anchors>\n\n"
        f"<Detailed description of the issue and the correct version or fix. Include cross-doc citations as repo-relative paths with anchors. "
        f"If the entry relies on general knowledge for a basic rule, say so in one short sentence. If a domain decision is required, state it explicitly.>\n\n"
        f"---\n\n"
        f"</report_format>\n\n"
        f"<constraints>\n"
        f"- Validate and refine only the report file.\n"
        f"- Do not change the target page or any other file.\n"
        f"- Commit exactly once.\n"
        f"</constraints>\n\n"
        f"<persistence>\n"
        f"Keep going until all findings are validated or removed, the format is exact, numbering is correct, and the single commit is made.\n"
        f"</persistence>\n\n"
        f"<commit_requirements>\n"
        f"CRITICAL: This run is considered failed unless you add and commit the refined report file. "
        f"Uncommitted workspace changes are ignored by orchestration.\n"
        f"- Required file: {report_path}\n"
        f'- Commit exactly once after refining: git add {report_path} && git commit -m "doc-review: refine {Path(report_path).name}"\n'
        f"</commit_requirements>\n"
    )

    # composer removed
