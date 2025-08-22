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
- reviewers_per_page: default 2
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
            raise ValueError("pages_file parsed but no pages found; expected list of {title, path}")

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
        reviewer_handles: List[Tuple[Dict[str, Any], int, Handle]] = []  # (page, r_idx, handle)
        for page in normalized_pages:
            for r_idx in range(1, int(cfg.reviewers_per_page) + 1):
                report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
                review_prompt = _build_reviewer_prompt(
                    original_prompt=prompt,
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

        async def _wait_and_validate(handle: Handle) -> Optional[Tuple[Dict[str, Any], int, Handle, InstanceResult]]:
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
                    original_prompt=prompt,
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
                retry_key = ctx.key("page", page["slug"], f"r{r_idx}", "review", "attempt-2")
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
        wait_tasks = [asyncio.create_task(_wait_and_validate(h)) for (_, _, h) in reviewer_handles]
        for coro in asyncio.as_completed(wait_tasks):
            res = await coro
            if res is None:
                continue
            page, r_idx, v_handle, _ = res
            validator_handles.append((page, r_idx, v_handle))

        # Stage 2: wait for all validators and verify outputs
        validated_reports: List[Tuple[Dict[str, Any], int, InstanceResult]] = []

        async def _await_validator(entry: Tuple[Dict[str, Any], int, Handle]) -> Optional[Tuple[Dict[str, Any], int, InstanceResult]]:
            page, r_idx, handle = entry
            report_rel = f"{cfg.report_dir}/raw/REPORT_{page['slug']}__r{r_idx}.md"
            try:
                vres = await ctx.wait(handle)
            except Exception:
                # One corrective retry prompting to ensure file commit
                corrective_prompt = _build_validator_prompt(
                    page_title=page["title"], page_path=page["path"], report_path=report_rel, corrective=True
                )
                vt = {"prompt": corrective_prompt, "base_branch": base_branch, "model": cfg.model}
                vkey = ctx.key("page", page["slug"], f"r{r_idx}", "validate", "attempt-2")
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
            pages.append({"title": title, "path": path, **({"slug": item.get("slug")} if item.get("slug") else {})})
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


def _read_file_from_branch(repo: Path, branch: Optional[str], rel_path: str) -> Optional[str]:
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
    *, original_prompt: str, page_title: str, page_path: str, report_path: str, corrective: bool = False
) -> str:
    guidance = (
        "You are a meticulous technical documentation reviewer. Analyze ONLY the specified page.\n"
        "Find any formatting issues, grammar errors, typos, inconsistencies, broken links, incorrect code block languages, incorrect facts, or mismatched terminology.\n"
        "Write your findings in a simple, strict format in the specified report file, then commit that file exactly once.\n"
    )
    format_rules = (
        "Report format (strict, keep exactly):\n"
        "- [ ] **<number>. <Short Precise Title>**\n\n"
        "<repo-relative path or stable URL with optional line anchors>\n\n"
        "<Detailed description of the issue and the correct version or fix.>\n\n"
        "---\n\n"
        "Repeat the list item structure for each finding. Do not include any other sections. Do not include frontmatter.\n"
    )
    corrective_text = (
        "\nIMPORTANT: Your previous attempt did not produce the required report file.\n"
        "You MUST create the file at the given path and commit it in a single commit.\n"
    ) if corrective else ""
    return (
        f"ORIGINAL REQUEST (context for scope only):\n{original_prompt}\n\n"
        f"PAGE TO REVIEW:\n- Title: {page_title}\n- Path: {page_path}\n\n"
        f"OUTPUT FILE (REQUIRED):\n- Create: {report_path}\n\n"
        + guidance
        + format_rules
        + corrective_text
        + (
            "\nInstructions:\n"
            "1. Read the page and identify issues.\n"
            "2. Produce the report in the exact format.\n"
            "3. 'git add' the report file and make ONE commit.\n"
            "4. Do NOT modify other files. Do NOT change the page itself.\n"
        )
    )


def _build_validator_prompt(*, page_title: str, page_path: str, report_path: str, corrective: bool = False) -> str:
    corrective_text = (
        "\nIMPORTANT: Ensure the report file exists, is well-formed, and commit the refined file.\n"
    ) if corrective else ""
    return (
        f"VALIDATION TASK:\n"
        f"- Open the report at: {report_path}\n"
        f"- Page: {page_title} ({page_path})\n\n"
        "For EACH finding in the report:\n"
        "- Extremely cautiously verify correctness against the page content and general tech writing best practices.\n"
        "- Delete invalid or unsubstantiated findings.\n"
        "- Refine descriptions to be clear, specific, and actionable.\n"
        "- Keep the exact simple format (checkbox line, link/path line, description paragraph, '---' separators).\n"
        "- Correct any numbering if out of order.\n\n"
        "Finally: commit the updated report file in a single commit. Do NOT change other files." + corrective_text
    )


def _build_composer_prompt(inputs: List[Tuple[str, str, str]], report_dir: str) -> str:
    header = (
        "FINAL REPORT COMPOSITION TASK:\n"
        f"- Create a single report at: {report_dir}/REPORT.md\n"
        "- Combine ALL validated reports provided below.\n"
        "- Deduplicate similar findings; if two entries are the same, keep one with the clearest description.\n"
        "- Keep the exact simple format shown below; produce a single numbered list of all findings across all pages.\n"
        "- Descriptions must be specific and more detailed than the minimal samples.\n"
        "- Commit exactly one file: the final REPORT.md.\n\n"
        "Format (strict):\n"
        "- [ ] **<number>. <Short Precise Title>**\n\n"
        "<repo-relative path or stable URL with optional line anchors>\n\n"
        "<Detailed description of the issue and the correct version or fix.>\n\n"
        "---\n\n"
    )
    # Concatenate inputs; for safety, add a separator header per source
    body_parts = []
    for slug, title, content in inputs:
        body_parts.append(f"# Source Report: {slug} — {title}\n\n{content}\n\n")
    sources = "".join(body_parts) if body_parts else "(no validated reports found)\n"
    return header + sources
