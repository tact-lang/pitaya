"""
Documentation Review Strategy

Minimal, plugin-agnostic multi-stage strategy for technical documentation review.

Stages:
1) Reviewers (N per page): each produces a simple structured report at
   reports/doc-review/raw/REPORT_{slug}__r{n}.md and commits once.
2) Validators (streamed, 1:1): run on each reviewer branch, strictly validate
   and refine the report in place, committing once.
3) Composer (global): after all validators complete, combine all validated
   reports into a single final reports/doc-review/REPORT.md on one branch.

Merging model: no git merges; the composer reads validated reports via git show
from their branches. To keep runner isolation, we inject the file contents into
the composer prompt (no workspace injection required).

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
    from ..strategy_context import StrategyContext, Handle


@dataclass
class DocReviewConfig(StrategyConfig):
    """Configuration for documentation review strategy."""

    pages_file: str = ""  # YAML or JSON list of pages
    reviewers_per_page: int = 1
    report_dir: str = "reports/doc-review"

    def validate(self) -> None:
        super().validate()
        if not self.pages_file:
            raise ValueError("pages_file is required for doc-review strategy")
        if self.reviewers_per_page < 1:
            raise ValueError("reviewers_per_page must be at least 1")


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

        # Stage 1: reviewers
        reviewer_handles: List[Tuple[Dict[str, Any], int, Handle]] = (
            []
        )  # (page, r_idx, handle)
        for page in normalized_pages:
            for r_idx in range(1, int(cfg.reviewers_per_page) + 1):
                report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
                review_prompt = _build_reviewer_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    report_path=report_rel,
                )
                task = {
                    "prompt": review_prompt,
                    "base_branch": base_branch,
                    "model": cfg.model,
                    # The reviewer should create the report file and commit it
                }
                key = ctx.key("page", page["slug"], f"r{r_idx}", "review")
                handle = await ctx.run(task, key=key)
                reviewer_handles.append((page, r_idx, handle))

        # As reviewers finish, schedule validators immediately
        validator_handles: List[Tuple[Dict[str, Any], int, Handle]] = []
        # Map instance_id -> (page, r_idx)
        reviewer_map: Dict[str, Tuple[Dict[str, Any], int]] = {
            h.instance_id: (pg, idx) for (pg, idx, h) in reviewer_handles
        }

        async def _wait_and_validate(
            handle: Handle,
        ) -> Optional[Tuple[Dict[str, Any], int, Handle, InstanceResult]]:
            page, r_idx = reviewer_map[handle.instance_id]
            result: InstanceResult
            try:
                result = await ctx.wait(handle)
            except Exception:
                # Reviewer failed
                return None

            # Guardrail: ensure report exists in branch
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            ok = _verify_file_in_branch(repo_path, result.branch_name, report_rel)
            if not (result.success and result.branch_name and ok):
                # One corrective retry with explicit instruction
                retry_prompt = _build_reviewer_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    report_path=report_rel,
                    corrective=True,
                )
                retry_task = {
                    "prompt": retry_prompt,
                    "base_branch": base_branch,
                    "model": cfg.model,
                }
                retry_key = ctx.key(
                    "page", page["slug"], f"r{r_idx}", "review", "attempt-2"
                )
                retry_handle = await ctx.run(retry_task, key=retry_key)
                try:
                    result = await ctx.wait(retry_handle)
                except Exception:
                    return None
                ok = _verify_file_in_branch(repo_path, result.branch_name, report_rel)
                if not (result.success and result.branch_name and ok):
                    return None

            # Schedule validator on the reviewer branch
            validator_prompt = _build_validator_prompt(
                page_title=page["title"],
                page_path=page["path"],
                report_path=report_rel,
            )
            v_task = {
                "prompt": validator_prompt,
                "base_branch": result.branch_name or base_branch,
                "model": cfg.model,
            }
            v_key = ctx.key("page", page["slug"], f"r{r_idx}", "validate")
            v_handle = await ctx.run(v_task, key=v_key)
            return (page, r_idx, v_handle, result)

        # Kick off waits concurrently
        wait_tasks = [
            asyncio.create_task(_wait_and_validate(h)) for (_, _, h) in reviewer_handles
        ]
        for coro in asyncio.as_completed(wait_tasks):
            res = await coro
            if res is None:
                continue
            page, r_idx, v_handle, _ = res
            validator_handles.append((page, r_idx, v_handle))

        # Stage 2: wait for all validators and verify outputs
        validated_reports: List[Tuple[Dict[str, Any], int, InstanceResult]] = []

        async def _await_validator(
            entry: Tuple[Dict[str, Any], int, Handle],
        ) -> Optional[Tuple[Dict[str, Any], int, InstanceResult]]:
            page, r_idx, handle = entry
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            try:
                vres = await ctx.wait(handle)
            except Exception:
                # One corrective retry prompting to ensure file commit
                corrective_prompt = _build_validator_prompt(
                    page_title=page["title"],
                    page_path=page["path"],
                    report_path=report_rel,
                    corrective=True,
                )
                vt = {
                    "prompt": corrective_prompt,
                    "base_branch": base_branch,
                    "model": cfg.model,
                }
                vkey = ctx.key(
                    "page", page["slug"], f"r{r_idx}", "validate", "attempt-2"
                )
                h2 = await ctx.run(vt, key=vkey)
                try:
                    vres = await ctx.wait(h2)
                except Exception:
                    return None

            ok = _verify_file_in_branch(repo_path, vres.branch_name, report_rel)
            if not (vres.success and vres.branch_name and ok):
                return None
            return (page, r_idx, vres)

        v_waits = [asyncio.create_task(_await_validator(v)) for v in validator_handles]
        for coro in asyncio.as_completed(v_waits):
            r = await coro
            if r is not None:
                validated_reports.append(r)

        # Stage 3: composer
        # Collect all validated report contents via git show and inject into composer prompt
        inputs: List[Tuple[str, str, str]] = []  # (slug, title, content)
        for page, r_idx, vres in validated_reports:
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            content = _read_file_from_branch(repo_path, vres.branch_name, report_rel)
            if content:
                inputs.append((page["slug"], page["title"], content))

        composer_prompt = _build_composer_prompt(inputs, cfg.report_dir)
        compose_task = {
            "prompt": composer_prompt,
            "base_branch": base_branch,
            "model": cfg.model,
        }
        c_key = ctx.key("compose")
        c_handle = await ctx.run(compose_task, key=c_key)
        c_result = await ctx.wait(c_handle)

        # Verify final report exists
        final_rel = f"{cfg.report_dir}/REPORT.md"
        c_ok = _verify_file_in_branch(repo_path, c_result.branch_name, final_rel)
        if not (c_result.success and c_result.branch_name and c_ok):
            # Best effort: return result but mark as failed
            c_result.success = False
            c_result.status = "failed"
            c_result.error = c_result.error or "Final report missing or not committed"

        # Return only the final composer result for strategy output
        return [c_result]


# ---------------------------
# Helpers
# ---------------------------


def _load_pages_file(p: Path, repo_path: Path) -> List[Dict[str, Any]]:
    """Load pages from YAML or JSON file.

    Returns a list of dicts with at least {title, path}. Relative paths are
    normalized relative to the repo root.
    """
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Could not read pages_file: {p}: {e}")

    pages: List[Dict[str, Any]] = []
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        elif p.suffix.lower() == ".json":
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


def _read_file_from_branch(
    repo: Path, branch: Optional[str], rel_path: str
) -> Optional[str]:
    if not branch:
        return None
    try:
        cmd = ["git", "-C", str(repo), "show", f"{branch}:{rel_path}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    except Exception:
        return None
    return None


def _build_reviewer_prompt(
    *,
    page_title: str,
    page_path: str,
    report_path: str,
    corrective: bool = False,
) -> str:
    corrective_text = (
        "IMPORTANT: The required report file was not created previously. Create it at the specified path and commit it in a single commit.\n\n"
        if corrective
        else ""
    )

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
        f"{corrective_text}"
    )


def _build_validator_prompt(
    *, page_title: str, page_path: str, report_path: str, corrective: bool = False
) -> str:
    corrective_text = (
        "IMPORTANT: Ensure the refined report file exists at the specified path and commit it in a single commit.\n\n"
        if corrective
        else ""
    )

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
        f"{corrective_text}"
    )


def _build_composer_prompt(inputs: List[Tuple[str, str, str]], report_dir: str) -> str:
    """
    inputs: List of tuples (slug, title, content) where content is the FULL validated report text for that page.
    """
    header = (
        f"<role>\n"
        f"You are composing a single, final user-facing documentation defect report by merging multiple validated per-page reports. "
        f"Your job is to deduplicate, standardize, and sequence findings while preserving precision and the required format.\n"
        f"</role>\n\n"
        f"<target>\n"
        f"OUTPUT FILE (required): {report_dir}/REPORT.md\n"
        f"</target>\n\n"
        f"<scope>\n"
        f"Include only user-facing issues. Exclude any items about repository internals, build/CI, docs engine internals, or anything not visible to readers.\n"
        f"</scope>\n\n"
        f"<truth_sources>\n"
        f"Do not invent new domain facts while merging. When findings conflict, prefer the most conservative reading or mark that a domain decision is required within the description. "
        f"The glossary remains canonical; if absent or silent, prefer the dominant/most recent usage across the docs.\n"
        f"</truth_sources>\n\n"
        f"<workflow>\n"
        f"1) Read all source reports in full.\n"
        f"2) Deduplicate: if two items are materially the same, keep one with the clearest description and best citation; fold any unique details from the duplicate into the survivor.\n"
        f"3) Normalize: align terminology, citation style, tone, and fix proposals across items.\n"
        f"4) Sequence: produce a single continuous numbering from 1..N. Grouping is not allowed in the output; only a flat list.\n"
        f"5) Output a single file at the required path and commit it once.\n"
        f"</workflow>\n\n"
        f"<report_format>\n"
        f"Keep this exact format for every finding:\n"
        f"- [ ] **<number>. <Short Precise Title>**\n\n"
        f"<repo-relative path or stable URL with optional line anchors>\n\n"
        f"<Detailed description of the issue and the correct version or fix. Include cross-doc citations as repo-relative paths with anchors. "
        f"If the entry relies on general knowledge for a basic rule, say so in one short sentence. If a domain decision is required, state it explicitly.>\n\n"
        f"---\n\n"
        f"</report_format>\n\n"
        f"<constraints>\n"
        f"- Produce exactly one file: REPORT.md at the specified path.\n"
        f"- Do not modify any other files.\n"
        f"- Make a single commit.\n"
        f"</constraints>\n\n"
        f"<persistence>\n"
        f"Keep going until the final merged report exists at the required path with continuous numbering and the single commit is made.\n"
        f"</persistence>\n\n"
    )

    # Concatenate inputs; include a clear, minimal provenance marker before each source
    body_parts = []
    for slug, title, content in inputs:
        body_parts.append(f"# Source Report: {slug} - {title}\n\n{content}\n")
    sources = (
        "".join(body_parts) if body_parts else "No validated reports were provided.\n"
    )

    return header + sources
