"""Aggregates git operations for the instance runner."""

from __future__ import annotations

from pathlib import Path

from .git_cleanup_impl import cleanup_workspace as _cleanup_workspace
from .git_import_impl import import_branch as _import_branch
from .git_run_cmd_impl import run_command as _run_command
from .git_workspace_impl import prepare_workspace as _prepare_workspace


class GitOperations:
    """Delegates git operations to helper modules (semantics unchanged)."""

    async def prepare_workspace(
        self, *args, **kwargs
    ):  # pragma: no cover - thin wrapper
        return await _prepare_workspace(self, *args, **kwargs)

    async def import_branch(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        return await _import_branch(self, *args, **kwargs)

    async def cleanup_workspace(self, workspace_dir: Path) -> None:  # pragma: no cover
        return await _cleanup_workspace(workspace_dir)

    async def _run_command(self, cmd: list[str]):  # pragma: no cover - thin wrapper
        return await _run_command(cmd)


__all__ = ["GitOperations"]
