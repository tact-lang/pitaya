"""Validation helpers for instance attempts."""

from __future__ import annotations

from pathlib import Path

from . import ValidationError


def validate_inputs(repo_path: Path, container_name: str, branch_name: str) -> None:
    if not repo_path.exists():
        raise ValidationError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise ValidationError(f"Repository path is not a directory: {repo_path}")
    if not (repo_path / ".git").exists():
        raise ValidationError(f"Not a git repository: {repo_path}")
    if not container_name:
        raise ValidationError("Container name must be provided by orchestration")
    if not branch_name:
        raise ValidationError("Branch name must be provided by orchestration")


__all__ = ["validate_inputs"]
