"""Workspace preparation helpers for instance attempts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from .git_operations import GitOperations
from .runner_params import RunnerParams

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Prepare and clean isolated workspaces."""

    def __init__(
        self,
        *,
        git_ops: GitOperations,
        params: RunnerParams,
        emit_event: Callable[[str, Dict], None],
        session_id: Optional[str],
    ) -> None:
        self.git_ops = git_ops
        self.params = params
        self.emit_event = emit_event
        self.session_id = session_id
        self.workspace_dir: Optional[Path] = None

    async def prepare(self) -> Path:
        reuse_existing = bool(self.params.operator_resume and self.session_id)
        if self.params.startup_semaphore is not None:
            return await self._prepare_with_semaphore(reuse_existing)
        self.emit_event(
            "instance.workspace_preparing", {"base_branch": self.params.base_branch}
        )
        self.workspace_dir = await self.git_ops.prepare_workspace(
            repo_path=self.params.repo_path,
            base_branch=self.params.base_branch,
            instance_id=self.params.instance_id,
            run_id=self.params.run_id,
            strategy_execution_id=self.params.strategy_execution_id,
            container_name=self.params.container_name,
            reuse_if_exists=reuse_existing,
            include_branches=self.params.workspace_include_branches,
        )
        self._emit_workspace_prepared()
        return self.workspace_dir

    async def cleanup(self) -> None:
        if self.workspace_dir:
            await self.git_ops.cleanup_workspace(self.workspace_dir)
            self.emit_event(
                "instance.workspace_cleaned", {"workspace_dir": str(self.workspace_dir)}
            )

    async def _prepare_with_semaphore(self, reuse_existing: bool) -> Path:
        try:
            self.emit_event(
                "instance.startup_waiting",
                {"base_branch": self.params.base_branch},
            )
        except Exception:
            pass
        async with self.params.startup_semaphore:
            self.emit_event(
                "instance.workspace_preparing", {"base_branch": self.params.base_branch}
            )
            self.workspace_dir = await self.git_ops.prepare_workspace(
                repo_path=self.params.repo_path,
                base_branch=self.params.base_branch,
                instance_id=self.params.instance_id,
                run_id=self.params.run_id,
                strategy_execution_id=self.params.strategy_execution_id,
                container_name=self.params.container_name,
                reuse_if_exists=reuse_existing,
                include_branches=self.params.workspace_include_branches,
            )
        self._emit_workspace_prepared()
        return self.workspace_dir

    def _emit_workspace_prepared(self) -> None:
        if not self.workspace_dir:
            return
        self.emit_event(
            "instance.workspace_prepared", {"workspace_dir": str(self.workspace_dir)}
        )
        self.emit_event("instance.phase_completed", {"phase": "workspace_preparation"})
        logger.info("Workspace prepared at: %s", self.workspace_dir)
