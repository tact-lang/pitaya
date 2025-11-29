"""Workspace cleanup helper."""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def cleanup_workspace(workspace_dir: Path) -> None:
    """Remove temporary workspace directory."""
    try:
        if workspace_dir and workspace_dir.exists():
            await asyncio.to_thread(
                workspace_dir.unlink if workspace_dir.is_file() else workspace_dir.rmdir
            )
    except Exception as e:
        logger.warning(f"Failed to cleanup workspace {workspace_dir}: {e}")


__all__ = ["cleanup_workspace"]
