"""Shared git helpers used by workspace preparation and import routines."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_BRANCH_RE = re.compile(r"^[A-Za-z0-9._\-/]{1,200}$")


def is_valid_branch_name(name: str) -> bool:
    """Return True when the branch name is syntactically safe."""
    if not isinstance(name, str) or not name:
        return False
    if name.startswith("-") or name.endswith("/"):
        return False
    if ".." in name or ".lock" in name or "@{" in name or "//" in name:
        return False
    return bool(_BRANCH_RE.match(name))


def is_protected_ref(name: str) -> bool:
    """Return True for protected refs (main/master/develop/stable/release/*/hotfix/*)."""
    if not isinstance(name, str):
        return False
    if name in ("main", "master", "develop", "stable"):
        return True
    return name.startswith("release/") or name.startswith("hotfix/")


def relax_workspace_permissions(root: Path) -> None:
    """Ensure the cloned workspace is writable by non-owner users."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        try:
            dir_path.chmod(0o777)
        except Exception:  # pragma: no cover - best effort only
            logger.debug("chmod failed for directory %s", dir_path)
        for name in filenames:
            file_path = dir_path / name
            try:
                if file_path.is_symlink():
                    continue
                file_path.chmod(0o666)
            except Exception:  # pragma: no cover - best effort only
                logger.debug("chmod failed for file %s", file_path)


def read_lock_holder(lock_path: Path) -> str:
    """Best-effort helper to decode lock holder metadata."""
    try:
        with lock_path.open("r") as fh:
            data = fh.read().strip()
    except Exception:
        return "<unavailable>"

    if not data:
        return "{}"

    try:
        obj = json.loads(data)
    except Exception:
        return data[:120]

    pid = obj.get("pid")
    ts = obj.get("ts")
    act = obj.get("action")
    return f"pid={pid} ts={ts} action={act}"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a subprocess invocation."""

    code: int
    output: str


async def run_command(cmd: Sequence[str]) -> CommandResult:
    """Run a git command and capture combined output."""
    logger.debug("Running command: %s", " ".join(cmd))
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (OSError, asyncio.SubprocessError) as exc:
        logger.error("Command failed to start: %s", exc)
        return CommandResult(1, str(exc))

    stdout, _ = await process.communicate()
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    logger.debug("Command completed with code %s", process.returncode)
    return CommandResult(process.returncode or 0, output)
