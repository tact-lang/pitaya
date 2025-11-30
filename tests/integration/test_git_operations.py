import subprocess
from pathlib import Path

import pytest

from pitaya.runner.workspace.git_operations import GitOperations


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed with {result.returncode}: {result.stderr}"
        )
    return result.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    run_git(repo_path, "init", "-b", "main")
    run_git(repo_path, "config", "user.email", "dev@example.com")
    run_git(repo_path, "config", "user.name", "Dev User")
    run_git(repo_path, "config", "commit.gpgsign", "false")
    (repo_path / "README.md").write_text("base\n")
    run_git(repo_path, "add", "README.md")
    run_git(repo_path, "commit", "-m", "initial")
    run_git(repo_path, "checkout", "-b", "feature")
    (repo_path / "feature.txt").write_text("feature branch\n")
    run_git(repo_path, "add", "feature.txt")
    run_git(repo_path, "commit", "-m", "feature")
    run_git(repo_path, "checkout", "main")
    return repo_path


@pytest.fixture()
def workspace_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Place workspaces under a writable tmp home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_prepare_workspace_sets_metadata_and_prunes_branches(
    repo: Path, workspace_home: Path
) -> None:
    git_ops = GitOperations()
    workspace = await git_ops.prepare_workspace(
        repo_path=repo,
        base_branch="main",
        instance_id="abc12345",
        include_branches=["feature"],
    )

    try:
        base_branch_marker = (workspace / ".git" / "BASE_BRANCH").read_text().strip()
        assert base_branch_marker == "main"

        branches = run_git(
            workspace, "for-each-ref", "--format=%(refname:short)", "refs/heads"
        ).splitlines()
        assert set(branches) == {"main", "feature"}

        remotes = run_git(workspace, "remote")
        assert remotes == ""

        assert (workspace / ".git" / "BASE_COMMIT").exists()
    finally:
        await git_ops.cleanup_workspace(workspace)


@pytest.mark.asyncio
async def test_import_branch_fetches_new_commit(
    repo: Path, workspace_home: Path
) -> None:
    git_ops = GitOperations()
    workspace = await git_ops.prepare_workspace(
        repo_path=repo,
        base_branch="main",
        instance_id="def67890",
    )

    try:
        run_git(workspace, "config", "user.email", "ws@example.com")
        run_git(workspace, "config", "user.name", "WS User")
        run_git(workspace, "config", "commit.gpgsign", "false")
        (workspace / "change.txt").write_text("workspace change\n")
        run_git(workspace, "add", "change.txt")
        run_git(workspace, "commit", "-m", "workspace change")

        result = await git_ops.import_branch(
            repo_path=repo,
            workspace_dir=workspace,
            branch_name="imported_branch",
        )

        workspace_head = run_git(workspace, "rev-parse", "HEAD")
        imported_head = run_git(repo, "rev-parse", "imported_branch")
        assert result["has_changes"] == "true"
        assert result["target_branch"] == "imported_branch"
        assert imported_head == workspace_head
    finally:
        await git_ops.cleanup_workspace(workspace)


@pytest.mark.asyncio
async def test_import_branch_skips_empty_import_when_policy_auto(
    repo: Path, workspace_home: Path
) -> None:
    git_ops = GitOperations()
    workspace = await git_ops.prepare_workspace(
        repo_path=repo,
        base_branch="main",
        instance_id="ghi90123",
    )

    try:
        result = await git_ops.import_branch(
            repo_path=repo,
            workspace_dir=workspace,
            branch_name="no_changes_branch",
            import_policy="auto",
            skip_empty_import=True,
        )
        assert result["has_changes"] == "false"
        assert result["target_branch"] is None
        assert result["dedupe_reason"] == "by_commit_no_changes"
    finally:
        await git_ops.cleanup_workspace(workspace)
