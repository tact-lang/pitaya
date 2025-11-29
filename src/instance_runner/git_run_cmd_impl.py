"""Run git command helper (extracted from original GitOperations)."""

from __future__ import annotations

import asyncio
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


async def run_command(cmd: list[str]) -> Tuple[int, str]:
    """Run a git command and return exit code and output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        return proc.returncode, output
    except FileNotFoundError:
        return 1, "git command not found"
    except Exception as exc:
        logger.error(f"Error running command {' '.join(cmd)}: {exc}")
        return 1, str(exc)


__all__ = ["run_command"]
