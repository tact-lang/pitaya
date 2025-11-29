"""Workspace preparation implementation (extracted from original GitOperations)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from . import GitError
from .git_utils import is_valid_branch_name as _is_valid_branch_name
from .git_utils import relax_workspace_permissions as _relax_workspace_permissions
from ..utils.platform_utils import get_temp_dir

logger = logging.getLogger(__name__)


async def prepare_workspace(
    git_ops,
    repo_path: Path,
    base_branch: str,
    instance_id: str,
    run_id: Optional[str] = None,
    strategy_execution_id: Optional[str] = None,
    container_name: Optional[str] = None,
    reuse_if_exists: bool = False,
    *,
    include_branches: Optional[list[str]] = None,
) -> Path:
    """Create isolated git workspace before container starts."""

    try:
        import platform

        if platform.system() == "Darwin":
            base_dir = Path.home() / ".pitaya" / "workspaces"
        else:
            base_dir = get_temp_dir()
    except Exception:
        base_dir = get_temp_dir()

    if run_id and container_name:
        tokens = (container_name or "").split("_")
        sidx = "0"
        khash = ""
        try:
            for t in tokens:
                if t.startswith("s") and t[1:].isdigit():
                    sidx = t[1:]
                    break
            for t in tokens:
                if t.startswith("k") and len(t) > 1:
                    khash = t[1:]
                    break
            if not khash:
                khash = (instance_id or "")[0:8] or "x"
        except Exception:
            sidx = "0"
            khash = (instance_id or "")[0:8] or "x"

        iid8 = (instance_id or "")[:8] or "x"
        workspace_dir = base_dir / f"pitaya/{run_id}/i_{sidx}_{khash}_{iid8}"
    else:
        workspace_dir = base_dir / f"pitaya/instance_{instance_id}"

    try:
        if not _is_valid_branch_name(base_branch):
            raise GitError(f"Invalid base branch name: {base_branch}")
        import platform

        if platform.system() == "Windows":
            if len(str(workspace_dir)) > 200:
                from uuid import uuid4

                workspace_dir = base_dir / f"pit/{uuid4().hex[:8]}"
    except Exception:
        pass

    try:
        await asyncio.to_thread(workspace_dir.parent.mkdir, parents=True, exist_ok=True)

        if reuse_if_exists and workspace_dir.exists():
            if (workspace_dir / ".git").exists():
                logger.info(f"Reusing existing workspace at {workspace_dir}")
                try:
                    base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
                    if not base_branch_file.exists():
                        base_branch_file.write_text(base_branch)
                except Exception:
                    pass
                return workspace_dir
            logger.info(
                f"Existing workspace at {workspace_dir} is not a git repo; recreating"
            )
            await asyncio.to_thread(shutil.rmtree, workspace_dir)

        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_dir)

        logger.info(f"Creating isolated workspace at {workspace_dir}")
        base_name = base_branch.strip()
        if not _is_valid_branch_name(base_name):
            raise GitError(f"Invalid base branch name: {base_branch}")
        if base_name.startswith("origin/") or base_name.startswith("refs/"):
            raise GitError(
                "base_branch must be an unqualified local branch name (e.g., 'main'), not 'origin/main' or 'refs/...'."
            )

        clone_cmd = [
            "git",
            "clone",
            "--no-hardlinks",
            str(repo_path),
            str(workspace_dir),
        ]

        result = await git_ops._run_command(clone_cmd)
        if result[0] != 0:
            raise GitError(f"Failed to clone repository: {result[1]}")

        base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
        base_branch_file.write_text(base_name)
        logger.info(f"Preserved base branch reference: {base_name}")

        rc_base, _ = await git_ops._run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{base_name}",
            ]
        )
        if rc_base != 0:
            raise GitError(f"Base branch not found in source repo: {base_name}")
        await git_ops._run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "checkout",
                "-B",
                base_name,
                f"origin/{base_name}",
            ]
        )

        created_locals: list[str] = []
        include_branches = list(include_branches or [])
        for name in include_branches:
            if not _is_valid_branch_name(name):
                raise GitError(f"Invalid include branch name: {name}")
            if name.startswith("origin/") or name.startswith("refs/"):
                raise GitError(
                    f"Include branch must be an unqualified branch name (got '{name}')"
                )
            rc_chk, _ = await git_ops._run_command(
                [
                    "git",
                    "-C",
                    str(workspace_dir),
                    "rev-parse",
                    "--verify",
                    f"refs/remotes/origin/{name}",
                ]
            )
            if rc_chk != 0:
                raise GitError(f"Include branch not found in source repo: {name}")
            await git_ops._run_command(
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
            created_locals.append(name)

        rc_ls, out_ls = await git_ops._run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads",
            ]
        )
        if rc_ls == 0:
            keep = {base_name, *created_locals}
            for ref in (out_ls or "").splitlines():
                ref = ref.strip()
                if ref and ref not in keep:
                    await git_ops._run_command(
                        [
                            "git",
                            "-C",
                            str(workspace_dir),
                            "branch",
                            "-D",
                            ref,
                        ]
                    )

        await git_ops._run_command(
            [
                "git",
                "-C",
                str(workspace_dir),
                "checkout",
                base_name,
            ]
        )

        try:
            base_commit_cmd = [
                "git",
                "-C",
                str(workspace_dir),
                "rev-parse",
                "HEAD",
            ]
            rc, out = await git_ops._run_command(base_commit_cmd)
            if rc == 0:
                (workspace_dir / ".git" / "BASE_COMMIT").write_text(out.strip())
        except Exception:
            pass

        remove_origin_cmd = [
            "git",
            "-C",
            str(workspace_dir),
            "remote",
            "remove",
            "origin",
        ]
        result = await git_ops._run_command(remove_origin_cmd)
        if result[0] != 0:
            logger.debug(f"Could not remove origin (may not exist): {result[1]}")

        try:
            rc_r, out_r = await git_ops._run_command(
                [
                    "git",
                    "-C",
                    str(workspace_dir),
                    "for-each-ref",
                    "--format=%(refname)",
                    "refs/remotes",
                ]
            )
            if rc_r == 0:
                for ref in (out_r or "").splitlines():
                    ref = (ref or "").strip()
                    if ref:
                        await git_ops._run_command(
                            [
                                "git",
                                "-C",
                                str(workspace_dir),
                                "update-ref",
                                "-d",
                                ref,
                            ]
                        )
        except Exception:
            pass

        try:
            await asyncio.to_thread(_relax_workspace_permissions, workspace_dir)
        except Exception as exc:
            logger.debug(f"Failed to relax workspace permissions: {exc}")
        logger.info(f"Workspace prepared successfully at {workspace_dir}")
        return workspace_dir

    except (OSError, IOError) as e:
        if workspace_dir.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
            except (OSError, IOError) as cleanup_error:
                logger.warning(f"Failed to cleanup workspace: {cleanup_error}")

        raise GitError(f"Failed to prepare workspace: {e}")
    except GitError:
        if workspace_dir.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
            except (OSError, IOError) as cleanup_error:
                logger.warning(f"Failed to cleanup workspace: {cleanup_error}")
        raise


__all__ = ["prepare_workspace"]
