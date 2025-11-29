"""Branch import orchestrator using helper modules."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from . import GitError
from .git_import_lock import acquire_import_lock, release_import_lock
from .git_import_logic import (
    compute_has_changes,
    handle_existing_branch,
    handle_suffix_match,
    resolve_branch_heads,
)
from .git_utils import is_protected_ref as _is_protected_ref

logger = logging.getLogger(__name__)


async def import_branch(
    git_ops,
    repo_path: Path,
    workspace_dir: Path,
    branch_name: str,
    import_policy: str = "auto",
    import_conflict_policy: str = "fail",
    skip_empty_import: bool = True,
    *,
    task_key: Optional[str] = None,
    run_id: Optional[str] = None,
    strategy_execution_id: Optional[str] = None,
    allow_overwrite_protected_refs: bool = False,
) -> Dict[str, Optional[str]]:
    lock_fh = None
    try:
        base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
        if not base_branch_file.exists():
            raise GitError(
                "BASE_BRANCH file not found - workspace not properly prepared"
            )
        base_branch = base_branch_file.read_text().strip()

        workspace_head, base_commit, branch_exists = await resolve_branch_heads(
            git_ops, repo_path, workspace_dir, branch_name
        )

        lock_path, lock_fh = await acquire_import_lock(
            repo_path,
            git_ops,
            run_id=run_id,
            strategy_execution_id=strategy_execution_id,
        )

        existing = None
        if branch_exists:
            existing = await handle_existing_branch(
                git_ops,
                repo_path,
                branch_name,
                workspace_head,
                task_key,
                run_id,
                strategy_execution_id,
            )
            if existing:
                return existing

        if import_conflict_policy == "suffix" and not branch_exists:
            suffix_match = await handle_suffix_match(
                git_ops,
                repo_path,
                branch_name,
                workspace_head,
                task_key,
                run_id,
                strategy_execution_id,
            )
            if suffix_match:
                return suffix_match

        has_new_commits, base_commit = await compute_has_changes(
            git_ops, workspace_dir, base_commit
        )

        if import_policy == "never":
            logger.info("Import policy 'never'; skipping branch import")
            return {
                "has_changes": "false",
                "target_branch": None,
                "commit": base_commit or workspace_head,
                "duplicate_of_branch": None,
                "dedupe_reason": None,
            }
        if import_policy == "auto" and skip_empty_import and not has_new_commits:
            logger.info("No changes; skipping branch creation per policy")
            return {
                "has_changes": "false",
                "target_branch": None,
                "commit": base_commit or workspace_head,
                "duplicate_of_branch": None,
                "dedupe_reason": "by_commit_no_changes",
            }

        target_branch = branch_name
        if branch_exists:
            if import_conflict_policy == "fail":
                raise GitError(f"Branch {branch_name} already exists")
            if import_conflict_policy == "overwrite":
                if (
                    _is_protected_ref(target_branch)
                    and not allow_overwrite_protected_refs
                ):
                    raise GitError(
                        f"Refusing to overwrite protected ref '{target_branch}'. Use --allow-overwrite-protected-refs to allow."
                    )
            if import_conflict_policy == "suffix":
                i = 2
                while True:
                    check_cmd = [
                        "git",
                        "-C",
                        str(repo_path),
                        "show-ref",
                        "--verify",
                        "--quiet",
                        f"refs/heads/{branch_name}_{i}",
                    ]
                    rc, _ = await git_ops._run_command(check_cmd)
                    if rc != 0:
                        target_branch = f"{branch_name}_{i}"
                        break
                    i += 1

        if has_new_commits:
            fetch_cmd = [
                "git",
                "-C",
                str(repo_path),
                "fetch",
                str(workspace_dir),
                (
                    f"+HEAD:{target_branch}"
                    if import_conflict_policy == "overwrite"
                    else f"HEAD:{target_branch}"
                ),
            ]

            result = await git_ops._run_command(fetch_cmd)
            if result[0] != 0:
                raise GitError(f"Failed to import branch: {result[1]}")

            logger.info(f"Successfully imported branch {target_branch}")
            try:
                _ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                note_lines = [
                    f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={target_branch}; ts={_ts}"
                ]
                await git_ops._run_command(
                    [
                        "git",
                        "-C",
                        str(repo_path),
                        "notes",
                        "--ref=pitaya",
                        "add",
                        "-f",
                        "-m",
                        "\n".join(note_lines),
                        workspace_head,
                    ]
                )
            except Exception:
                pass
            return {
                "has_changes": "true",
                "target_branch": target_branch,
                "commit": workspace_head,
                "duplicate_of_branch": None,
                "dedupe_reason": None,
            }

        base_sha_cmd = ["git", "-C", str(repo_path), "rev-parse", base_branch]
        result = await git_ops._run_command(base_sha_cmd)
        if result[0] != 0:
            raise GitError(f"Failed to get base branch SHA: {result[1]}")
        base_sha = result[1].strip()

        create_branch_cmd = [
            "git",
            "-C",
            str(repo_path),
            "branch",
            target_branch,
            base_sha,
        ]
        result = await git_ops._run_command(create_branch_cmd)
        if result[0] != 0:
            raise GitError(f"Failed to create branch: {result[1]}")

        logger.info(f"Created branch {target_branch} pointing to {base_branch}")
        return {
            "has_changes": "false",
            "target_branch": target_branch,
            "commit": base_sha,
            "duplicate_of_branch": None,
            "dedupe_reason": "by_commit_no_changes",
        }

    except GitError:
        raise
    except (OSError, IOError) as e:
        raise GitError(f"Failed to import branch: {e}")
    finally:
        release_import_lock(lock_fh)


__all__ = ["import_branch"]
