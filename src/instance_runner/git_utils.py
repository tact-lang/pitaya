"""Utility helpers for git operations in the instance runner."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_BRANCH_RE = None


def relax_workspace_permissions(root: Path) -> None:
    """Ensure the cloned workspace is writable by non-owner users."""
    try:
        if not root.exists():
            return
        for dirpath, _, filenames in os.walk(root):
            dir_path = Path(dirpath)
            try:
                dir_path.chmod(0o777)
            except Exception as exc:  # pragma: no cover - best effort only
                logger.debug(f"chmod failed for directory {dir_path}: {exc}")
            for name in filenames:
                file_path = dir_path / name
                try:
                    if file_path.is_symlink():
                        continue
                    file_path.chmod(0o666)
                except Exception as exc:  # pragma: no cover - best effort only
                    logger.debug(f"chmod failed for file {file_path}: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.debug(f"Failed to relax permissions under {root}: {exc}")


def is_valid_branch_name(name: str) -> bool:
    """Validate branch name against a strict regex and forbidden substrings."""
    try:
        import re

        global _BRANCH_RE
        if _BRANCH_RE is None:
            _BRANCH_RE = re.compile(r"^[A-Za-z0-9._\-/]{1,200}$")
        if not isinstance(name, str) or not name:
            return False
        if name.startswith("-") or name.endswith("/"):
            return False
        if ".." in name or ".lock" in name or "@{" in name:
            return False
        if "//" in name:
            return False
        return bool(_BRANCH_RE.match(name))
    except Exception:
        return False


def is_protected_ref(name: str) -> bool:
    """Return True if the ref name is protected."""
    if not isinstance(name, str):
        return False
    if name in ("main", "master", "develop", "stable"):
        return True
    return name.startswith("release/") or name.startswith("hotfix/")


def read_lock_holder(lock_file: Path) -> Tuple[bool, Optional[str]]:
    """Return whether lock exists and its holder if present."""
    try:
        if not lock_file.exists():
            return False, None
        content = lock_file.read_text(encoding="utf-8", errors="ignore")
        content = content.strip()
        return True, content or None
    except Exception:
        return False, None


__all__ = [
    "relax_workspace_permissions",
    "is_valid_branch_name",
    "is_protected_ref",
    "read_lock_holder",
]
