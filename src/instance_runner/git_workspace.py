"""Workspace preparation utilities for isolated git clones."""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import GitError
from .git_common import is_valid_branch_name, relax_workspace_permissions, run_command
from ..utils.platform_utils import get_temp_dir

logger = logging.getLogger(__name__)

WINDOWS_MAX_PATH = 200


@dataclass(frozen=True, slots=True)
class WorkspaceRequest:
    """Input parameters required to prepare an isolated workspace."""

    repo_path: Path
    base_branch: str
    instance_id: str
    run_id: str | None = None
    strategy_execution_id: str | None = None
    container_name: str | None = None
    reuse_if_exists: bool = False
    include_branches: tuple[str, ...] = ()


class WorkspacePreparer:
    """Prepare isolated workspaces for containerized runs."""

    async def prepare(self, request: WorkspaceRequest) -> Path:
        workspace_dir = self._compute_workspace_dir(request)
        self._validate_branches(request)
        await asyncio.to_thread(workspace_dir.parent.mkdir, parents=True, exist_ok=True)

        if request.reuse_if_exists and workspace_dir.exists():
            reused = await self._reuse_if_possible(workspace_dir, request.base_branch)
            if reused:
                return workspace_dir

        try:
            await self._remove_if_exists(workspace_dir)
            await self._clone_repository(request.repo_path, workspace_dir)
            await self._record_base_branch(workspace_dir, request.base_branch)
            await self._materialize_branches(
                workspace_dir, request.base_branch, request.include_branches
            )
            await self._prune_local_branches(
                workspace_dir, keep={request.base_branch, *request.include_branches}
            )
            await self._record_base_commit(workspace_dir)
            await self._detach_origin(workspace_dir)
            await asyncio.to_thread(relax_workspace_permissions, workspace_dir)
            logger.info("Workspace prepared successfully at %s", workspace_dir)
            return workspace_dir
        except Exception:
            await self._remove_if_exists(workspace_dir)
            raise

    def _compute_workspace_dir(self, request: WorkspaceRequest) -> Path:
        base_dir = (
            Path.home() / ".pitaya" / "workspaces"
            if platform.system() == "Darwin"
            else get_temp_dir()
        )
        if request.run_id and request.container_name:
            workspace_dir = base_dir / self._derive_durable_path(request)
        else:
            workspace_dir = base_dir / f"pitaya/instance_{request.instance_id}"

        if (
            platform.system() == "Windows"
            and len(str(workspace_dir)) > WINDOWS_MAX_PATH
        ):
            from uuid import uuid4

            workspace_dir = base_dir / f"pit/{uuid4().hex[:8]}"
        return workspace_dir

    def _derive_durable_path(self, request: WorkspaceRequest) -> Path:
        tokens = (request.container_name or "").split("_")
        sidx = next(
            (t[1:] for t in tokens if t.startswith("s") and t[1:].isdigit()), "0"
        )
        khash = next((t[1:] for t in tokens if t.startswith("k") and len(t) > 1), "")
        iid8 = (request.instance_id or "")[:8] or "x"
        khash = khash or iid8
        return Path(f"pitaya/{request.run_id}/i_{sidx}_{khash}_{iid8}")

    def _validate_branches(self, request: WorkspaceRequest) -> None:
        branches = (request.base_branch,) + request.include_branches
        for name in branches:
            if not is_valid_branch_name(name):
                raise GitError(f"Invalid branch name: {name}")
            if name.startswith("origin/") or name.startswith("refs/"):
                raise GitError(
                    f"Branch must be unqualified (got '{name}'); use 'main' not 'origin/main'."
                )

    async def _reuse_if_possible(self, workspace_dir: Path, base_branch: str) -> bool:
        git_dir = workspace_dir / ".git"
        if not git_dir.exists():
            logger.info("Existing workspace %s missing .git; recreating", workspace_dir)
            return False
        logger.info("Reusing existing workspace at %s", workspace_dir)
        base_branch_file = git_dir / "BASE_BRANCH"
        if not base_branch_file.exists():
            try:
                base_branch_file.write_text(base_branch)
            except Exception:
                logger.debug("Failed to rewrite BASE_BRANCH for reused workspace")
        return True

    async def _remove_if_exists(self, workspace_dir: Path) -> None:
        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_dir)

    async def _clone_repository(self, repo_path: Path, workspace_dir: Path) -> None:
        result = await run_command(
            ["git", "clone", "--no-hardlinks", str(repo_path), str(workspace_dir)]
        )
        if result.code != 0:
            raise GitError(f"Failed to clone repository: {result.output}")

    async def _record_base_branch(self, workspace_dir: Path, base_branch: str) -> None:
        base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
        base_branch_file.write_text(base_branch.strip())

    async def _materialize_branches(
        self, workspace_dir: Path, base_branch: str, include_branches: Iterable[str]
    ) -> None:
        await self._verify_remote_branch(workspace_dir, base_branch)
        await self._checkout_branch(workspace_dir, base_branch)
        for name in include_branches:
            await self._verify_remote_branch(workspace_dir, name)
            await self._checkout_branch(workspace_dir, name)
        await run_command(["git", "-C", str(workspace_dir), "checkout", base_branch])

    async def _verify_remote_branch(self, workspace_dir: Path, name: str) -> None:
        rc = await run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{name}",
            ]
        )
        if rc.code != 0:
            raise GitError(f"Branch not found in source repo: {name}")

    async def _checkout_branch(self, workspace_dir: Path, name: str) -> None:
        await run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "checkout",
                "-B",
                name,
                f"origin/{name}",
            ]
        )

    async def _prune_local_branches(self, workspace_dir: Path, keep: set[str]) -> None:
        refs = await run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads",
            ]
        )
        if refs.code != 0:
            return
        for ref in (refs.output or "").splitlines():
            ref = ref.strip()
            if ref and ref not in keep:
                await run_command(
                    ["git", "-C", str(workspace_dir), "branch", "-D", ref]
                )

    async def _record_base_commit(self, workspace_dir: Path) -> None:
        rc = await run_command(["git", "-C", str(workspace_dir), "rev-parse", "HEAD"])
        if rc.code == 0:
            (workspace_dir / ".git" / "BASE_COMMIT").write_text(rc.output.strip())

    async def _detach_origin(self, workspace_dir: Path) -> None:
        remove_result = await run_command(
            ["git", "-C", str(workspace_dir), "remote", "remove", "origin"]
        )
        if remove_result.code != 0:
            logger.debug("Could not remove origin: %s", remove_result.output)

        refs = await run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "for-each-ref",
                "--format=%(refname)",
                "refs/remotes",
            ]
        )
        if refs.code != 0:
            return
        for ref in (refs.output or "").splitlines():
            ref = ref.strip()
            if ref:
                await run_command(
                    ["git", "-C", str(workspace_dir), "update-ref", "-d", ref]
                )
