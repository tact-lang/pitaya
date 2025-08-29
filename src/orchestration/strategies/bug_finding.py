"""
Bug Finding Strategy — discover, then validate and document a real bug.

Two phases with clear contracts:
1) Discovery: scan a target area, confirm one real bug with a minimal reproduction,
   and produce a structured report in the final message (read‑only — no commits).
2) Validation: independently reproduce the bug from base_branch, write BUG_REPORT.md,
   and commit exactly once if valid; otherwise explain why not reproducible.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext

logger = logging.getLogger(__name__)


@dataclass
class BugFindingConfig(StrategyConfig):
    """Configuration for bug finding strategy."""

    # Target area to search for bugs (e.g., "src/parser", "authentication module")
    target_area: str = ""
    # Additional context about what to look for
    bug_focus: str = ""
    # Validation artifacts
    report_path: str = "BUG_REPORT.md"
    # Attempts
    discovery_max_retries: int = 1
    validation_max_retries: int = 1
    # Read-only discovery (no commits during discovery)
    read_only_discovery: bool = True
    # Optional model override for validation
    validator_model: str = ""
    # Optional durable key prefix
    key_prefix: str = ""

    def validate(self) -> None:  # type: ignore[override]
        super().validate()
        if not self.target_area:
            raise ValueError("target_area must be specified for bug finding")
        if self.discovery_max_retries < 0:
            raise ValueError("discovery_max_retries must be >= 0")
        if self.validation_max_retries < 0:
            raise ValueError("validation_max_retries must be >= 0")


class BugFindingStrategy(Strategy):
    """
    Two-phase bug finding strategy.

    Phase 1: Discovery - Find and document a bug
    Phase 2: Validation - Reproduce and commit bug report
    """

    @property
    def name(self) -> str:
        return "bug-finding"

    def get_config_class(self) -> type[BugFindingConfig]:
        return BugFindingConfig

    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """Execute bug finding strategy."""
        cfg: BugFindingConfig = self.create_config()  # type: ignore

        # Phase 1 — Discovery (read-only)
        logger.info(f"bug-finding: discovery in area '{cfg.target_area}'")
        disc_prompt = _build_discovery_prompt(
            target_area=cfg.target_area, bug_focus=cfg.bug_focus, original_prompt=prompt
        )
        disc_task: Dict[str, Any] = {
            "prompt": disc_prompt,
            "base_branch": base_branch,
            "model": cfg.model,
            "import_policy": "never" if cfg.read_only_discovery else "auto",
            "skip_empty_import": True,
        }
        attempts = int(cfg.discovery_max_retries) + 1
        disc_result: Optional[InstanceResult] = None
        prefix = [cfg.key_prefix] if cfg.key_prefix else []
        for i in range(1, attempts + 1):
            key = ctx.key(*prefix, "discovery", cfg.target_area, f"attempt-{i}")
            try:
                h = await ctx.run(disc_task, key=key)
                r = await ctx.wait(h)
            except Exception:
                r = None
            if r and r.success and (r.final_message or "").strip():
                disc_result = r
                break
            disc_result = r or disc_result

        if not (disc_result and disc_result.success):
            return [disc_result] if disc_result else []

        bug_report = (disc_result.final_message or "").strip()
        if not bug_report:
            # Treat empty report as failure
            disc_result.success = False
            disc_result.status = "failed"
            disc_result.error = "discovery did not produce a bug report"
            return [disc_result]

        # Phase 2 — Validation (reproduce independently from base_branch)
        logger.info("bug-finding: validation starting")
        validator_model = (cfg.validator_model or cfg.model or "sonnet").strip()
        val_prompt = _build_validation_prompt(
            bug_report=bug_report, report_path=cfg.report_path
        )
        val_task: Dict[str, Any] = {
            "prompt": val_prompt,
            "base_branch": base_branch,
            "model": validator_model,
        }
        v_attempts = int(cfg.validation_max_retries) + 1
        val_result: Optional[InstanceResult] = None
        for i in range(1, v_attempts + 1):
            vkey = ctx.key(*prefix, "validation", cfg.target_area, f"attempt-{i}")
            try:
                vh = await ctx.run(val_task, key=vkey)
                vr = await ctx.wait(vh)
            except Exception:
                vr = None
            if vr and vr.success:
                val_result = vr
                break
            val_result = vr or val_result

        if not val_result:
            # Surface validation failure explicitly
            try:
                from ...shared import InstanceResult as _IR

                failed_val = _IR(
                    success=False,
                    error="validation did not complete",
                    error_type="validation_failed",
                    status="failed",
                )
            except Exception:
                failed_val = None
            if failed_val is not None:
                failed_val.metadata["bug_confirmed"] = False
                return [disc_result, failed_val]
            return [disc_result]

        # Confirm bug via commit and artifact presence
        confirmed = False
        try:
            commit_count = (
                val_result.commit_statistics.get("commit_count", 0)
                if val_result.commit_statistics
                else 0
            )
        except Exception:
            commit_count = 0

        if commit_count > 0 and _verify_file_in_branch(
            Path(getattr(ctx._orchestrator, "repo_path", ".")),
            val_result.branch_name,
            cfg.report_path,
        ):
            confirmed = True

        if confirmed:
            val_result.metadata["bug_confirmed"] = True
            val_result.metadata["bug_report_branch"] = val_result.branch_name
            logger.info(
                f"bug-finding: confirmed bug; report {cfg.report_path} committed on {val_result.branch_name}"
            )
        else:
            val_result.success = False
            val_result.status = "failed"
            val_result.error = (
                "Bug could not be reproduced or was invalid (no report commit found)"
            )
            val_result.metadata["bug_confirmed"] = False
            logger.info("bug-finding: validation failed (no report commit)")

        return [disc_result, val_result]


# ---------------------------
# Prompt + helper utilities
# ---------------------------


def _build_discovery_prompt(
    *, target_area: str, bug_focus: str, original_prompt: str
) -> str:
    focus = bug_focus.strip()
    focus_block = f"Specific focus: {focus}\n" if focus else ""
    return (
        f"<role>\n"
        f"You are a seasoned bug hunter. Search for a real, reproducible bug in the target area below.\n"
        f"Confirm the issue with a minimal reproduction and stop after documenting one confirmed bug.\n"
        f"</role>\n\n"
        f"<target_area>\n{target_area}\n</target_area>\n\n"
        f"<context>\n{focus_block}Original request: {original_prompt}\n</context>\n\n"
        f"<scope>\n"
        f"Consider logic errors, edge cases, race conditions, error handling, input validation, and security issues.\n"
        f"Prefer issues that can be demonstrated with a concise, local change or clear reproduction steps.\n"
        f"</scope>\n\n"
        f"<constraints>\n"
        f"- Do not modify files in this phase.\n"
        f"- Provide a single confirmed bug only.\n"
        f"</constraints>\n\n"
        f"<output>\n"
        f"Provide a concise structured report in plain text including:\n"
        f"- Title\n- Impact/Severity\n- Root Cause\n- Minimal Reproduction Steps\n- Affected Files/Paths\n"
        f"</output>\n"
    )


def _build_validation_prompt(*, bug_report: str, report_path: str) -> str:
    name = Path(report_path).name
    return (
        f"<role>\n"
        f"You are validating a bug report independently. Reproduce the issue on this branch and document it.\n"
        f"</role>\n\n"
        f"<bug_report>\n{bug_report}\n</bug_report>\n\n"
        f"<workflow>\n"
        f"1) Read the report fully and understand the reproduction.\n"
        f"2) Attempt the reproduction exactly; adjust paths/commands with care.\n"
        f"3) If the bug is real, write a {name} with: summary, severity, reproduction steps, root cause, affected components.\n"
        f"4) If helpful, add a minimal test/PoC file.\n"
        f"</workflow>\n\n"
        f"<constraints>\n"
        f"- Create only the report file and minimal reproduction assets.\n"
        f"- Make exactly one commit when valid; otherwise make no changes.\n"
        f"</constraints>\n\n"
        f"<commit_requirements>\n"
        f"- Required file: {report_path}\n"
        f'- Commit exactly once: git add {report_path} && git commit -m "bug: add {name}"\n'
        f"</commit_requirements>\n"
    )


def _verify_file_in_branch(repo: Path, branch: Optional[str], rel_path: str) -> bool:
    if not branch:
        return False
    try:
        cmd = ["git", "-C", str(repo), "show", f"{branch}:{rel_path}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout)
    except Exception:
        return False
