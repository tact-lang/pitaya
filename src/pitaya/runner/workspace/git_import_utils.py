"""Shared helpers for git import routines."""

from __future__ import annotations

from pathlib import Path

from . import GitError
from .git_common import run_command


def read_base_branch(workspace_dir: Path) -> str:
    """Read the base branch marker written during workspace preparation."""
    path = workspace_dir / ".git" / "BASE_BRANCH"
    if not path.exists():
        raise GitError("BASE_BRANCH file not found - workspace not properly prepared")
    return path.read_text().strip()


def read_base_commit(workspace_dir: Path) -> str:
    """Read the base commit marker if present."""
    path = workspace_dir / ".git" / "BASE_COMMIT"
    return path.read_text().strip() if path.exists() else ""


async def resolve_git_dir(repo_path: Path) -> Path:
    """Return the absolute path to the repository's .git directory."""
    result = await run_command(["git", "-C", str(repo_path), "rev-parse", "--git-dir"])
    if result.code != 0:
        raise GitError(f"Failed to resolve git dir: {result.output}")
    raw = Path(result.output.strip())
    return raw if raw.is_absolute() else (repo_path / raw).resolve()


async def branch_exists(repo_path: Path, name: str) -> bool:
    """Check whether a branch already exists in the repo."""
    cmd = [
        "git",
        "-C",
        str(repo_path),
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{name}",
    ]
    return (await run_command(cmd)).code == 0


async def read_workspace_head(workspace_dir: Path) -> str:
    """Return the HEAD commit in the workspace clone."""
    result = await run_command(["git", "-C", str(workspace_dir), "rev-parse", "HEAD"])
    if result.code != 0:
        raise GitError(f"Failed to get workspace HEAD: {result.output}")
    return result.output.strip()


async def has_new_commits(base_commit: str, workspace_dir: Path) -> bool:
    """Determine whether workspace has commits beyond base."""
    if not base_commit:
        return True
    count = await run_command(
        [
            "git",
            "-C",
            str(workspace_dir),
            "rev-list",
            "--count",
            f"{base_commit}..HEAD",
        ]
    )
    if count.code != 0:
        return True
    try:
        return int(count.output.strip()) > 0
    except Exception:
        return True


async def resolve_base_sha(repo_path: Path, base_branch: str) -> str:
    """Resolve the SHA of the base branch in the host repo."""
    result = await run_command(["git", "-C", str(repo_path), "rev-parse", base_branch])
    if result.code != 0:
        raise GitError(f"Failed to get base branch SHA: {result.output}")
    return result.output.strip()


async def create_branch_at_sha(repo_path: Path, branch_name: str, sha: str) -> None:
    """Create a branch pointing at the provided SHA."""
    result = await run_command(
        ["git", "-C", str(repo_path), "branch", branch_name, sha]
    )
    if result.code != 0:
        raise GitError(f"Failed to create branch: {result.output}")
