"""Workspace cleanup helper (extracted from original GitOperations)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


async def cleanup_workspace(workspace_dir: Path) -> None:
    """Remove temporary workspace directory."""
    try:
        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_dir)
            logger.info(f"Cleaned up workspace {workspace_dir}")
    except (OSError, IOError, shutil.Error) as e:
        logger.warning(f"Failed to cleanup workspace {workspace_dir}: {e}")


__all__ = ["cleanup_workspace"]
