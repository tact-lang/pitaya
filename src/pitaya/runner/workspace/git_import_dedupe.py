"""Deduplication helpers for branch import operations."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .git_common import run_command
from .git_import_types import ImportResult

logger = logging.getLogger(__name__)


async def dedupe_matching_branch(
    repo_path: Path,
    target_branch: str,
    workspace_head: str,
    task_key: Optional[str],
    note_writer: Callable[[str], Awaitable[None]],
) -> Optional[ImportResult]:
    """Handle the case where the target branch already points at workspace HEAD."""
    head_cmd = [
        "git",
        "-C",
        str(repo_path),
        "rev-parse",
        f"refs/heads/{target_branch}",
    ]
    head = await run_command(head_cmd)
    if head.code != 0 or head.output.strip() != workspace_head:
        return None

    notes = await run_command(
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
    if notes.code == 0 and task_key and (f"task_key={task_key}" in notes.output):
        reason = "same_task_provenance"
    else:
        await note_writer(target_branch)
        reason = "by_commit"

    logger.info("Branch already matches workspace HEAD; idempotent success")
    return ImportResult(
        has_changes=True,
        target_branch=target_branch,
        commit=workspace_head,
        duplicate_of_branch=None,
        dedupe_reason=reason,
    )


async def dedupe_suffixed_branch(
    repo_path: Path,
    base_branch: str,
    workspace_head: str,
    task_key: Optional[str],
    note_writer: Callable[[str], Awaitable[None]],
) -> Optional[ImportResult]:
    """Check for an existing suffixed branch pointing to the same commit."""
    refs = await run_command(
        [
            "git",
            "-C",
            str(repo_path),
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            "refs/heads",
        ]
    )
    if refs.code != 0:
        return None

    pattern = re.compile(rf"^{re.escape(base_branch)}(_[0-9]+)?$")
    for line in refs.output.splitlines():
        try:
            ref, obj = line.strip().split(" ")
        except ValueError:
            continue
        if not pattern.match(ref) or obj != workspace_head:
            continue

        notes = await run_command(
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
        if notes.code == 0 and task_key and (f"task_key={task_key}" in notes.output):
            reason = "same_task_provenance"
        else:
            await note_writer(ref)
            reason = "crash_window"

        return ImportResult(
            has_changes=True,
            target_branch=ref,
            commit=workspace_head,
            duplicate_of_branch=ref,
            dedupe_reason=reason,
        )
    return None
