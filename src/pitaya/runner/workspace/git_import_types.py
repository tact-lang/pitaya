"""Type definitions and enums for git import operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from . import GitError


class ImportPolicy(Enum):
    """Allowed import policies."""

    AUTO = "auto"
    NEVER = "never"
    ALWAYS = "always"


class ImportConflictPolicy(Enum):
    """Conflict handling for existing branches."""

    FAIL = "fail"
    OVERWRITE = "overwrite"
    SUFFIX = "suffix"


@dataclass(frozen=True, slots=True)
class ImportBranchRequest:
    """Inputs required to import a workspace branch."""

    repo_path: Path
    workspace_dir: Path
    branch_name: str
    import_policy: ImportPolicy
    conflict_policy: ImportConflictPolicy
    skip_empty_import: bool
    task_key: Optional[str]
    run_id: Optional[str]
    strategy_execution_id: Optional[str]
    allow_overwrite_protected_refs: bool


@dataclass(frozen=True, slots=True)
class ImportContext:
    """Computed state about the pending import."""

    branch_exists: bool
    base_commit: str
    workspace_head: str
    has_new_commits: bool


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Structured import result for callers."""

    has_changes: bool
    target_branch: Optional[str]
    commit: Optional[str]
    duplicate_of_branch: Optional[str] = None
    dedupe_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "has_changes": "true" if self.has_changes else "false",
            "target_branch": self.target_branch,
            "commit": self.commit,
            "duplicate_of_branch": self.duplicate_of_branch,
            "dedupe_reason": self.dedupe_reason,
        }


def coerce_import_policy(raw: str) -> ImportPolicy:
    """Convert user-facing string policy into the Enum."""
    try:
        return ImportPolicy(raw)
    except ValueError:
        raise GitError(f"Unsupported import policy '{raw}'")


def coerce_conflict_policy(raw: str) -> ImportConflictPolicy:
    """Convert conflict policy string into Enum."""
    try:
        return ImportConflictPolicy(raw)
    except ValueError:
        raise GitError(f"Unsupported import conflict policy '{raw}'")
