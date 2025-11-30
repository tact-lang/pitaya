"""Import workspace changes and compute commit statistics."""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from pitaya.exceptions import GitError
from pitaya.runner.workspace.git_operations import GitOperations
from pitaya.runner.runner_params import RunnerParams

logger = logging.getLogger(__name__)


class ImportManager:
    """Handles branch import, optional force-commit, and statistics gathering."""

    def __init__(
        self,
        *,
        git_ops: GitOperations,
        params: RunnerParams,
        workspace_dir,
        emit_event,
    ) -> None:
        self.git_ops = git_ops
        self.params = params
        self.workspace_dir = workspace_dir
        self.emit_event = emit_event

    async def import_branch(self) -> Dict[str, Optional[str]]:
        return await self.git_ops.import_branch(
            repo_path=self.params.repo_path,
            workspace_dir=self.workspace_dir,
            branch_name=self.params.branch_name,
            import_policy=self.params.import_policy,
            import_conflict_policy=self.params.import_conflict_policy,
            skip_empty_import=self.params.skip_empty_import,
            task_key=self.params.task_key,
            run_id=self.params.run_id,
            strategy_execution_id=self.params.strategy_execution_id,
            allow_overwrite_protected_refs=self.params.allow_overwrite_protected_refs,
        )

    async def force_commit_if_needed(self) -> None:
        if not self.params.force_commit:
            return
        try:
            status_cmd = ["git", "-C", str(self.workspace_dir), "status", "--porcelain"]
            rc, out = await self.git_ops._run_command(status_cmd)  # type: ignore[attr-defined]
            if rc != 0 or not (out or "").strip():
                return
            await self.git_ops._run_command(["git", "-C", str(self.workspace_dir), "add", "-A"])  # type: ignore[attr-defined]
            msg = f"pitaya: force commit (run_id={self.params.run_id or ''} iid={self.params.instance_id})"
            commit_cmd = ["git", "-C", str(self.workspace_dir), "commit", "-m", msg]
            await self.git_ops._run_command(commit_cmd)  # type: ignore[attr-defined]
            logger.info("force-commit created a commit in workspace")
        except Exception as exc:  # pragma: no cover - best effort only
            logger.debug("force-commit error ignored: %s", exc)

    async def gather_stats(self, has_changes: bool) -> Optional[Dict[str, int]]:
        if not has_changes:
            return None
        commit_statistics: Dict[str, int] = {}
        try:
            stats_cmd = [
                "git",
                "-C",
                str(self.params.repo_path),
                "diff",
                "--stat",
                f"{self.params.base_branch}..{self.params.branch_name}",
            ]
            _, stats_output = await self.git_ops._run_command(stats_cmd)  # type: ignore[attr-defined]

            stats_match = re.search(
                r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(\-\))?",
                stats_output,
            )
            if stats_match:
                commit_statistics = {
                    "files_changed": int(stats_match.group(1) or 0),
                    "insertions": int(stats_match.group(2) or 0),
                    "deletions": int(stats_match.group(3) or 0),
                }

            count_cmd = [
                "git",
                "-C",
                str(self.params.repo_path),
                "rev-list",
                "--count",
                f"{self.params.base_branch}..{self.params.branch_name}",
            ]
            _, count_output = await self.git_ops._run_command(count_cmd)  # type: ignore[attr-defined]
            if count_output.strip().isdigit():
                commit_statistics["commit_count"] = int(count_output.strip())
        except (GitError, OSError) as exc:
            logger.warning("Failed to get commit statistics: %s", exc)
        return commit_statistics or None

    def emit_after_import(
        self,
        import_info: Dict[str, Optional[str]],
        final_branch: Optional[str],
        has_changes: bool,
    ) -> None:
        if has_changes:
            self.emit_event("instance.branch_imported", {"branch_name": final_branch})
        else:
            self.emit_event(
                "instance.no_changes",
                {"branch_name": import_info.get("target_branch")},
            )
        self.emit_event("instance.phase_completed", {"phase": "result_collection"})
