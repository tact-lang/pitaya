"""Implementation of importing workspace commits into the host repository."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import GitError
from .git_common import is_protected_ref, is_valid_branch_name, run_command
from .git_import_dedupe import dedupe_matching_branch, dedupe_suffixed_branch
from .git_import_types import (
    ImportBranchRequest,
    ImportConflictPolicy,
    ImportContext,
    ImportPolicy,
    ImportResult,
)
from .git_import_utils import (
    branch_exists,
    create_branch_at_sha,
    has_new_commits as compute_has_new_commits,
    read_base_branch,
    read_base_commit,
    read_workspace_head,
    resolve_base_sha,
    resolve_git_dir,
)
from .git_lock import RepoLock

logger = logging.getLogger(__name__)

MAX_SUFFIX_ATTEMPTS = 50


class BranchImporter:
    """Import commits from a workspace into the host repository."""

    async def import_branch(self, request: ImportBranchRequest) -> ImportResult:
        base_branch = read_base_branch(request.workspace_dir)
        git_dir = await resolve_git_dir(request.repo_path)
        lock_path = git_dir / ".pitaya_import.lock"
        metadata = {
            "action": "import_branch",
            "run_id": request.run_id,
            "strategy_execution_id": request.strategy_execution_id,
        }

        async with RepoLock(lock_path, metadata=metadata):
            return await self._import_with_lock(request, base_branch)

    async def _import_with_lock(
        self, request: ImportBranchRequest, base_branch: str
    ) -> ImportResult:
        self._validate_branch_name(request.branch_name)
        self._guard_protected_ref(
            request.branch_name, request.allow_overwrite_protected_refs
        )

        context = await self._build_context(request)
        note_writer = self._note_writer(
            request.repo_path,
            context.workspace_head,
            request.task_key,
            request.run_id,
            request.strategy_execution_id,
        )

        dedup = await self._dedupe(request, context, note_writer)
        if dedup:
            return dedup

        policy_result = self._apply_import_policy(
            request.import_policy,
            request.skip_empty_import,
            context.has_new_commits,
            context.base_commit,
            context.workspace_head,
        )
        if policy_result:
            return policy_result

        target_branch = await self._resolve_target_branch(
            request, context.branch_exists
        )

        if context.has_new_commits:
            await self._fetch_workspace_commits(
                request.repo_path,
                request.workspace_dir,
                target_branch,
                request.conflict_policy,
            )
            await note_writer(target_branch)
            return ImportResult(
                has_changes=True,
                target_branch=target_branch,
                commit=context.workspace_head,
                duplicate_of_branch=None,
                dedupe_reason=None,
            )

        base_sha = await self._resolve_base_sha(request.repo_path, base_branch)
        await self._create_branch_at_sha(request.repo_path, target_branch, base_sha)
        return ImportResult(
            has_changes=False,
            target_branch=target_branch,
            commit=base_sha,
            duplicate_of_branch=None,
            dedupe_reason="by_commit_no_changes",
        )

    async def _dedupe(
        self,
        request: ImportBranchRequest,
        context: ImportContext,
        note_writer,
    ) -> Optional[ImportResult]:
        if context.branch_exists:
            return await dedupe_matching_branch(
                request.repo_path,
                request.branch_name,
                context.workspace_head,
                request.task_key,
                note_writer,
            )
        if request.conflict_policy == ImportConflictPolicy.SUFFIX:
            return await dedupe_suffixed_branch(
                request.repo_path,
                request.branch_name,
                context.workspace_head,
                request.task_key,
                note_writer,
            )
        return None

    async def _build_context(self, request: ImportBranchRequest) -> ImportContext:
        existing_branch = await branch_exists(request.repo_path, request.branch_name)
        base_commit = read_base_commit(request.workspace_dir)
        workspace_head = await read_workspace_head(request.workspace_dir)
        new_commits = await compute_has_new_commits(base_commit, request.workspace_dir)
        return ImportContext(
            branch_exists=existing_branch,
            base_commit=base_commit,
            workspace_head=workspace_head,
            has_new_commits=new_commits,
        )

    def _validate_branch_name(self, name: str) -> None:
        if not is_valid_branch_name(name):
            raise GitError(f"Invalid target branch name: {name}")

    def _guard_protected_ref(self, name: str, allow_overwrite: bool) -> None:
        if is_protected_ref(name) and not allow_overwrite:
            raise GitError(
                f"Refusing to update protected ref '{name}'. "
                "Use allow_overwrite_protected_refs to allow."
            )

    def _apply_import_policy(
        self,
        policy: ImportPolicy,
        skip_empty: bool,
        has_new_commits: bool,
        base_commit: str,
        workspace_head: str,
    ) -> Optional[ImportResult]:
        if policy == ImportPolicy.NEVER:
            logger.info("Import policy 'never'; skipping branch import")
            return ImportResult(
                has_changes=False,
                target_branch=None,
                commit=base_commit or workspace_head,
            )
        if policy == ImportPolicy.AUTO and skip_empty and not has_new_commits:
            logger.info("No changes; skipping branch creation per policy")
            return ImportResult(
                has_changes=False,
                target_branch=None,
                commit=base_commit or workspace_head,
                dedupe_reason="by_commit_no_changes",
            )
        return None

    async def _resolve_target_branch(
        self, request: ImportBranchRequest, branch_exists: bool
    ) -> str:
        if not branch_exists:
            return request.branch_name
        if request.conflict_policy == ImportConflictPolicy.FAIL:
            raise GitError(f"Branch {request.branch_name} already exists")
        if request.conflict_policy == ImportConflictPolicy.OVERWRITE:
            if (
                is_protected_ref(request.branch_name)
                and not request.allow_overwrite_protected_refs
            ):
                raise GitError(
                    f"Refusing to overwrite protected ref '{request.branch_name}'. "
                    "Use allow_overwrite_protected_refs to allow."
                )
            return request.branch_name

        suffix = 2
        while suffix <= MAX_SUFFIX_ATTEMPTS:
            candidate = f"{request.branch_name}_{suffix}"
            if not await branch_exists(request.repo_path, candidate):
                return candidate
            suffix += 1
        raise GitError(
            f"Unable to allocate suffixed branch after {MAX_SUFFIX_ATTEMPTS} attempts"
        )

    async def _fetch_workspace_commits(
        self,
        repo_path: Path,
        workspace_dir: Path,
        target_branch: str,
        conflict_policy: ImportConflictPolicy,
    ) -> None:
        refspec = (
            f"+HEAD:{target_branch}"
            if conflict_policy == ImportConflictPolicy.OVERWRITE
            else f"HEAD:{target_branch}"
        )
        result = await run_command(
            ["git", "-C", str(repo_path), "fetch", str(workspace_dir), refspec]
        )
        if result.code != 0:
            raise GitError(f"Failed to import branch: {result.output}")

    def _note_writer(
        self,
        repo_path: Path,
        workspace_head: str,
        task_key: Optional[str],
        run_id: Optional[str],
        strategy_execution_id: Optional[str],
    ):
        async def _write(branch: str) -> None:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            note_parts = [
                f"task_key={task_key}",
                f"run_id={run_id}",
                f"strategy_execution_id={strategy_execution_id}",
                f"branch={branch}",
                f"ts={ts}",
            ]
            try:
                await run_command(
                    [
                        "git",
                        "-C",
                        str(repo_path),
                        "notes",
                        "--ref=pitaya",
                        "add",
                        "-f",
                        "-m",
                        "; ".join(note_parts),
                        workspace_head,
                    ]
                )
            except Exception:
                logger.debug("Failed to write provenance note for %s", branch)

        return _write

    async def _resolve_base_sha(self, repo_path: Path, base_branch: str) -> str:
        return await resolve_base_sha(repo_path, base_branch)

    async def _create_branch_at_sha(
        self, repo_path: Path, branch_name: str, sha: str
    ) -> None:
        await create_branch_at_sha(repo_path, branch_name, sha)
