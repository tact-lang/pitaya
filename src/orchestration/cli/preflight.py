"""Preflight checks before running a new orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path
import argparse

from rich.console import Console

__all__ = ["perform_preflight_checks"]


def _hint(console: Console, message: str, bullets: list[str]) -> None:
    console.print(f"[red]{message}[/red]")
    if bullets:
        console.print("Try:")
        for b in bullets[:3]:
            console.print(f"  • {b}")


def _check_repo(console: Console, repo_path: Path, base_branch: str) -> bool:
    if not repo_path.exists():
        _hint(
            console,
            f"repository not found: {repo_path}",
            ["create repo: git init", f"verify path: {repo_path}"],
        )
        return False
    if not repo_path.is_dir():
        _hint(
            console,
            f"path is not a directory: {repo_path}",
            ["choose a git repo directory", f"ls -la {repo_path}"],
        )
        return False
    if not (repo_path / ".git").exists():
        console.print(f"[red]Error: Not a git repository: {repo_path}[/red]")
        console.print("Initialize with: git init")
        return False
    # Disallow remote-qualified base names to match runner's strict rules
    if str(base_branch).startswith("origin/") or str(base_branch).startswith("refs/"):
        _hint(
            console,
            f"base branch must be an unqualified local name (got '{base_branch}')",
            ["use 'main' not 'origin/main'", "git branch --list"],
        )
        return False

    rc = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--verify", base_branch],
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        _hint(
            console,
            f"base branch not found: '{base_branch}'",
            ["git fetch origin --prune", "git branch --all"],
        )
        return False
    return True


def perform_preflight_checks(console: Console, args: argparse.Namespace) -> bool:
    repo_path = Path(getattr(args, "repo"))
    if not _check_repo(console, repo_path, getattr(args, "base_branch", "main")):
        return False
    # Enforce clean working tree if requested; warn otherwise
    try:
        dirty = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            if getattr(args, "require_clean_wt", False):
                console.print(
                    "[red]Working tree has uncommitted changes. Use --require-clean-wt to enforce cleanliness.[/red]"
                )
                return False
            console.print(
                "[yellow]Warning: Working tree has uncommitted changes.[/yellow]"
            )
    except (subprocess.SubprocessError, OSError):
        # Non-fatal; orchestrator will surface more specific errors later
        return True
    # Docker image access check is non-blocking; leave to orchestrator
    return True
