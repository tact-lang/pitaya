"""Facade for git workspace isolation and import routines."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

from .git_importer import (
    BranchImporter,
    ImportBranchRequest,
    coerce_conflict_policy,
    coerce_import_policy,
)
from .git_workspace import WorkspacePreparer, WorkspaceRequest

logger = logging.getLogger(__name__)

__all__ = ["GitOperations"]


class GitOperations:
    """Public API consumed by the instance runner."""

    def __init__(self) -> None:
        self._workspace_preparer = WorkspacePreparer()
        self._branch_importer = BranchImporter()

    async def prepare_workspace(
        self,
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
        """Create an isolated workspace before the container starts."""
        request = WorkspaceRequest(
            repo_path=repo_path,
            base_branch=base_branch,
            instance_id=instance_id,
            run_id=run_id,
            strategy_execution_id=strategy_execution_id,
            container_name=container_name,
            reuse_if_exists=reuse_if_exists,
            include_branches=tuple(include_branches or ()),
        )
        return await self._workspace_preparer.prepare(request)

    async def import_branch(
        self,
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
        """Import workspace commits back to the host repository."""
        request = ImportBranchRequest(
            repo_path=repo_path,
            workspace_dir=workspace_dir,
            branch_name=branch_name,
            import_policy=coerce_import_policy(import_policy),
            conflict_policy=coerce_conflict_policy(import_conflict_policy),
            skip_empty_import=skip_empty_import,
            task_key=task_key,
            run_id=run_id,
            strategy_execution_id=strategy_execution_id,
            allow_overwrite_protected_refs=allow_overwrite_protected_refs,
        )
        result = await self._branch_importer.import_branch(request)
        return result.to_dict()

    async def cleanup_workspace(self, workspace_dir: Path) -> None:
        """Remove a temporary workspace directory."""
        try:
            if workspace_dir.exists():
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
                logger.info("Cleaned up workspace %s", workspace_dir)
        except (OSError, IOError, shutil.Error) as exc:
            logger.warning("Failed to cleanup workspace %s: %s", workspace_dir, exc)
