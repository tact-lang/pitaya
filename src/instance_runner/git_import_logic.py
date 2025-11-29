"""Core import logic helpers for git branch import."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import GitError
from .git_utils import is_valid_branch_name as _is_valid_branch_name

logger = logging.getLogger(__name__)


async def resolve_branch_heads(
    git_ops, repo_path: Path, workspace_dir: Path, branch_name: str
) -> Tuple[str, str, bool]:
    if not _is_valid_branch_name(branch_name):
        raise GitError(f"Invalid target branch name: {branch_name}")

    check_branch_cmd = [
        "git",
        "-C",
        str(repo_path),
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{branch_name}",
    ]
    branch_exists = (await git_ops._run_command(check_branch_cmd))[0] == 0

    ws_head_cmd = ["git", "-C", str(workspace_dir), "rev-parse", "HEAD"]
    rc, ws_head_out = await git_ops._run_command(ws_head_cmd)
    if rc != 0:
        raise GitError(f"Failed to get workspace HEAD: {ws_head_out}")
    workspace_head = ws_head_out.strip()

    base_commit_file = workspace_dir / ".git" / "BASE_COMMIT"
    base_commit = (
        base_commit_file.read_text().strip() if base_commit_file.exists() else ""
    )
    return workspace_head, base_commit, branch_exists


async def handle_existing_branch(
    git_ops,
    repo_path: Path,
    branch_name: str,
    workspace_head: str,
    task_key: Optional[str],
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
) -> Optional[Dict[str, Optional[str]]]:
    br_head_cmd = [
        "git",
        "-C",
        str(repo_path),
        "rev-parse",
        f"refs/heads/{branch_name}",
    ]
    rc, br_head_out = await git_ops._run_command(br_head_cmd)
    if rc != 0 or br_head_out.strip() != workspace_head:
        return None

    note_rc, note_out = await git_ops._run_command(
        [
            "git",
            "-C",
            str(repo_path),
            "notes",
            "--ref=pitaya",
            "show",
            workspace_head,
        ]
    )
    if note_rc == 0 and task_key and (f"task_key={task_key}" in note_out):
        dedupe_reason = "same_task_provenance"
    else:
        try:
            _ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            note_lines = [
                f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={branch_name}; ts={_ts}"
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
        dedupe_reason = "by_commit"
    return {
        "has_changes": "true",
        "target_branch": branch_name,
        "commit": workspace_head,
        "duplicate_of_branch": None,
        "dedupe_reason": dedupe_reason,
    }


async def handle_suffix_match(
    git_ops,
    repo_path: Path,
    branch_name: str,
    workspace_head: str,
    task_key: Optional[str],
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
) -> Optional[Dict[str, Optional[str]]]:
    list_refs = [
        "git",
        "-C",
        str(repo_path),
        "for-each-ref",
        "--format=%(refname:short) %(objectname)",
        "refs/heads",
    ]
    rc, refs_out = await git_ops._run_command(list_refs)
    if rc != 0:
        return None
    import re

    for line in refs_out.splitlines():
        try:
            ref, obj = line.strip().split(" ")
        except ValueError:
            continue
        if re.match(rf"^{branch_name}(_[0-9]+)?$", ref) and obj == workspace_head:
            note_rc, note_out = await git_ops._run_command(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "notes",
                    "--ref=pitaya",
                    "show",
                    workspace_head,
                ]
            )
            if note_rc == 0 and task_key and (f"task_key={task_key}" in note_out):
                return {
                    "has_changes": "true",
                    "target_branch": ref,
                    "commit": workspace_head,
                    "duplicate_of_branch": ref,
                    "dedupe_reason": "same_task_provenance",
                }
            try:
                _ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                note_lines = [
                    f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={ref}; ts={_ts}"
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
                "target_branch": ref,
                "commit": workspace_head,
                "duplicate_of_branch": ref,
                "dedupe_reason": "crash_window",
            }
    return None


async def compute_has_changes(
    git_ops, workspace_dir: Path, base_commit: str
) -> Tuple[bool, str]:
    has_new_commits = True
    if base_commit:
        count_cmd = [
            "git",
            "-C",
            str(workspace_dir),
            "rev-list",
            "--count",
            f"{base_commit}..HEAD",
        ]
        rc, count_out = await git_ops._run_command(count_cmd)
        if rc == 0:
            try:
                has_new_commits = int(count_out.strip()) > 0
            except Exception:
                has_new_commits = True
    return has_new_commits, base_commit


__all__ = [
    "resolve_branch_heads",
    "handle_existing_branch",
    "handle_suffix_match",
    "compute_has_changes",
]
