"""
Git operations for instance workspace isolation.

Implements the isolation strategy from specification section 3.5:
- Pre-container: Create completely isolated git clone
- Post-container: Import work back via git fetch
"""

import asyncio
import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

from . import GitError
from ..utils.platform_utils import get_temp_dir

logger = logging.getLogger(__name__)


class GitOperations:
    """Handles git workspace isolation for instances."""

    async def prepare_workspace(
        self,
        repo_path: Path,
        base_branch: str,
        instance_id: str,
        run_id: Optional[str] = None,
        strategy_execution_id: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> Path:
        """
        Create isolated git workspace before container starts.

        This implements the isolation strategy from the spec:
        - Clone only the target branch with --single-branch
        - Use --no-hardlinks to prevent any inode-level interference
        - Preserve base branch reference in .git/BASE_BRANCH
        - Remove origin to complete disconnection

        Args:
            repo_path: Host repository path
            base_branch: Branch to start from
            instance_id: Unique instance identifier

        Returns:
            Path to the prepared workspace directory
        """
        # Choose a workspace base path that Docker Desktop can mount reliably.
        # Prefer user-configured base via env, then a per-user directory under $HOME on macOS,
        # and fall back to system temp on other platforms.
        custom_base = os.environ.get("ORCHESTRATOR_WORKSPACE_BASE")
        if custom_base:
            base_dir = Path(custom_base)
        else:
            try:
                import platform
                if platform.system() == "Darwin":
                    # macOS: /Users is shared by default with Docker Desktop
                    base_dir = Path.home() / ".orchestrator" / "workspaces"
                else:
                    base_dir = get_temp_dir()
            except Exception:
                base_dir = get_temp_dir()

        if run_id and container_name:
            # Extract indices from container name for workspace naming
            # Container name format: orchestrator_{timestamp}_s{sidx}_i{iidx}
            name_parts = container_name.split("_")
            if len(name_parts) >= 4:
                sidx = name_parts[-2].lstrip("s")
                iidx = name_parts[-1].lstrip("i")
                # Path: {base}/orchestrator/${run_id}/i_${sidx}_${iidx}
                workspace_dir = base_dir / f"orchestrator/{run_id}/i_{sidx}_{iidx}"
            else:
                # Fallback if container name doesn't match expected format
                workspace_dir = base_dir / f"orchestrator/{run_id}/{instance_id}"
        else:
            # Fallback for standalone usage
            workspace_dir = base_dir / f"orchestrator/instance_{instance_id}"

        # On Windows, ensure path length remains below typical limits
        try:
            import platform
            if platform.system() == "Windows":
                # If the final path would be too long, use a shorter form
                if len(str(workspace_dir)) > 200:
                    from uuid import uuid4
                    workspace_dir = base_dir / f"orc/{uuid4().hex[:8]}"
        except Exception:
            pass

        try:
            # Ensure parent directory exists
            await asyncio.to_thread(
                workspace_dir.parent.mkdir, parents=True, exist_ok=True
            )

            # Remove any existing workspace (shouldn't happen but be safe)
            if workspace_dir.exists():
                await asyncio.to_thread(shutil.rmtree, workspace_dir)

            logger.info(f"Creating isolated workspace at {workspace_dir}")

            use_shared_clone = bool(os.environ.get("ORCHESTRATOR_GIT_SHARED_CLONE"))

            # Prepare optional shared mirror for faster clones
            reference_arg = []
            if use_shared_clone:
                try:
                    mirror_base = base_dir / "orchestrator/cache"
                    mirror_base.mkdir(parents=True, exist_ok=True)
                    # Use a simple hash of repo path for cache key
                    import hashlib
                    key = hashlib.sha256(str(repo_path).encode("utf-8")).hexdigest()[:12]
                    mirror = mirror_base / f"{key}.mirror"
                    if not mirror.exists():
                        # Create mirror
                        init_cmd = [
                            "git",
                            "clone",
                            "--mirror",
                            str(repo_path),
                            str(mirror),
                        ]
                        result = await self._run_command(init_cmd)
                        if result[0] != 0:
                            logger.warning(f"Failed to create mirror: {result[1]}")
                    else:
                        # Update mirror
                        fetch_cmd = [
                            "git",
                            "-C",
                            str(mirror),
                            "fetch",
                            "-p",
                        ]
                        await self._run_command(fetch_cmd)
                    if mirror.exists():
                        reference_arg = ["--reference-if-able", str(mirror)]
                except Exception as e:
                    logger.debug(f"Shared clone setup failed: {e}")

            # Clone with complete isolation (optionally using shared reference)
            # --branch: Start from specific branch
            # --single-branch: Only clone that branch, no other refs
            # --no-hardlinks: Force physical copy of all objects
            clone_cmd = [
                "git",
                "clone",
                "--branch",
                base_branch,
                "--single-branch",
                "--no-hardlinks",
                *reference_arg,
                str(repo_path),
                str(workspace_dir),
            ]

            result = await self._run_command(clone_cmd)
            if result[0] != 0:
                raise GitError(f"Failed to clone repository: {result[1]}")

            # Preserve base branch reference as per spec
            base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
            base_branch_file.write_text(base_branch)
            logger.info(f"Preserved base branch reference: {base_branch}")

            # Record base commit for idempotency checks (BASE_COMMIT)
            try:
                base_commit_cmd = [
                    "git",
                    "-C",
                    str(workspace_dir),
                    "rev-parse",
                    "HEAD",
                ]
                rc, out = await self._run_command(base_commit_cmd)
                if rc == 0:
                    (workspace_dir / ".git" / "BASE_COMMIT").write_text(out.strip())
            except Exception:
                pass

            # Remove origin to complete isolation
            remove_origin_cmd = [
                "git",
                "-C",
                str(workspace_dir),
                "remote",
                "remove",
                "origin",
            ]
            result = await self._run_command(remove_origin_cmd)
            if result[0] != 0:
                # This might fail if no origin exists, which is fine
                logger.debug(f"Could not remove origin (may not exist): {result[1]}")

            logger.info(f"Workspace prepared successfully at {workspace_dir}")
            return workspace_dir

        except (OSError, IOError) as e:
            # Clean up on failure
            if workspace_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, workspace_dir)
                except (OSError, IOError) as cleanup_error:
                    logger.warning(f"Failed to cleanup workspace: {cleanup_error}")

            raise GitError(f"Failed to prepare workspace: {e}")
        except GitError:
            # Clean up on failure
            if workspace_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, workspace_dir)
                except (OSError, IOError) as cleanup_error:
                    logger.warning(f"Failed to cleanup workspace: {cleanup_error}")
            raise

    async def import_branch(
        self,
        repo_path: Path,
        workspace_dir: Path,
        branch_name: str,
        import_policy: str = "auto",
        import_conflict_policy: str = "fail",
        skip_empty_import: bool = True,
    ) -> bool:
        """
        Import work from isolated workspace back to host repository.

        Uses git fetch to import all commits from the workspace.
        This creates the branch atomically in the host repository.

        Args:
            repo_path: Host repository path
            workspace_dir: Instance workspace directory
            branch_name: Name for the imported branch
            force: Force overwrite if branch already exists

        Returns:
            True if changes were imported, False if no changes
        """
        try:
            # Note: We do NOT create commits here. Claude Code handles its own commits.
            # We just import whatever commits exist in the workspace.

            # Read the base branch name from the preserved file
            base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
            if not base_branch_file.exists():
                raise GitError(
                    "BASE_BRANCH file not found - workspace not properly prepared"
                )

            base_branch = base_branch_file.read_text().strip()

            # Idempotent pre-check and conflict policy handling happen under a repo lock
            # Acquire per-repo import lock
            git_dir_cmd = ["git", "-C", str(repo_path), "rev-parse", "--git-dir"]
            rc, out = await self._run_command(git_dir_cmd)
            if rc != 0:
                raise GitError(f"Failed to resolve git dir: {out}")
            from pathlib import Path as _P
            git_dir = _P(out.strip()) if _P(out.strip()).is_absolute() else (repo_path / out.strip())
            lock_path = git_dir / ".orc_import.lock"

            # Simple cross-platform exclusive lock (best-effort)
            import os, sys
            lock_fh = None
            lock_fh = open(lock_path, "a+")
            try:
                if sys.platform.startswith("win"):
                    import msvcrt
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            # Check if target branch already exists
            check_branch_cmd = [
                "git",
                "-C",
                str(repo_path),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{branch_name}",
            ]
            branch_exists_result = await self._run_command(check_branch_cmd)
            branch_exists = branch_exists_result[0] == 0

            # Read preserved base commit
            base_commit_file = workspace_dir / ".git" / "BASE_COMMIT"
            base_commit = base_commit_file.read_text().strip() if base_commit_file.exists() else ""

            # Determine if workspace has new commits relative to base
            ws_head_cmd = ["git", "-C", str(workspace_dir), "rev-parse", "HEAD"]
            rc, ws_head_out = await self._run_command(ws_head_cmd)
            if rc != 0:
                raise GitError(f"Failed to get workspace HEAD: {ws_head_out}")
            workspace_head = ws_head_out.strip()

            # If branch exists, idempotent pre-check: identical heads → treat as success
            if branch_exists:
                br_head_cmd = ["git", "-C", str(repo_path), "rev-parse", f"refs/heads/{branch_name}"]
                rc, br_head_out = await self._run_command(br_head_cmd)
                if rc == 0 and br_head_out.strip() == workspace_head:
                    logger.info("Branch already matches workspace HEAD; idempotent success")
                    return True

            # Suffix idempotency: if suffix policy and base name missing, but a suffixed branch points to ws HEAD
            if import_conflict_policy == "suffix" and not branch_exists:
                try:
                    list_refs = [
                        "git",
                        "-C",
                        str(repo_path),
                        "for-each-ref",
                        "--format=%(refname:short) %(objectname)",
                        "refs/heads",
                    ]
                    rc, refs_out = await self._run_command(list_refs)
                    if rc == 0:
                        for line in refs_out.splitlines():
                            try:
                                ref, obj = line.strip().split(" ")
                            except ValueError:
                                continue
                            import re
                            if re.match(rf"^{branch_name}(_[0-9]+)?$", ref) and obj == workspace_head:
                                logger.info(f"Found existing suffixed branch {ref} with matching HEAD; idempotent success")
                                return True
                except Exception:
                    pass

            # Compute if there are new commits relative to base
            has_new_commits = True
            if base_commit:
                count_cmd = [
                    "git",
                    "-C",
                    str(workspace_dir),
                    "rev-list",
                    "--count",
                    f"{base_commit}..HEAD",
                ]
                rc, count_out = await self._run_command(count_cmd)
                if rc == 0:
                    try:
                        has_new_commits = int(count_out.strip()) > 0
                    except Exception:
                        has_new_commits = True

            # Apply import policy
            if import_policy == "never":
                logger.info("Import policy 'never'; skipping branch import")
                return False
            if import_policy == "auto" and skip_empty_import and not has_new_commits:
                logger.info("No changes; skipping branch creation per policy")
                return False

            # Import commits from workspace
            logger.info(
                f"Importing branch {branch_name} from {workspace_dir} (base: {base_branch})"
            )

            # Resolve conflict policy (fail|overwrite|suffix)
            target_branch = branch_name
            if branch_exists:
                if import_conflict_policy == "fail":
                    raise GitError(f"Branch {branch_name} already exists")
                elif import_conflict_policy == "overwrite":
                    pass  # will force update via +HEAD
                elif import_conflict_policy == "suffix":
                    # Find next available suffix
                    i = 2
                    while True:
                        check_cmd = ["git", "-C", str(repo_path), "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}_{i}"]
                        rc, _ = await self._run_command(check_cmd)
                        if rc != 0:
                            target_branch = f"{branch_name}_{i}"
                            break
                        i += 1

            if has_new_commits:
                # There are new commits, fetch them
                fetch_cmd = [
                    "git",
                    "-C",
                    str(repo_path),
                    "fetch",
                    str(workspace_dir),
                    (f"+HEAD:{target_branch}" if import_conflict_policy == "overwrite" else f"HEAD:{target_branch}"),
                ]

                result = await self._run_command(fetch_cmd)
                if result[0] != 0:
                    raise GitError(f"Failed to import branch: {result[1]}")

                logger.info(f"Successfully imported branch {target_branch}")
                return True
            else:
                # No new commits, create branch pointing to base
                logger.info(
                    f"No new commits found, creating branch {branch_name} pointing to {base_branch}"
                )

                # Get the base branch commit SHA in the host repo
                base_sha_cmd = ["git", "-C", str(repo_path), "rev-parse", base_branch]
                result = await self._run_command(base_sha_cmd)
                if result[0] != 0:
                    raise GitError(f"Failed to get base branch SHA: {result[1]}")
                base_sha = result[1].strip()

                # Create branch pointing to base
                create_branch_cmd = [
                    "git",
                    "-C",
                    str(repo_path),
                    "branch",
                    target_branch,
                    base_sha,
                ]
                result = await self._run_command(create_branch_cmd)
                if result[0] != 0:
                    raise GitError(f"Failed to create branch: {result[1]}")

                logger.info(f"Created branch {target_branch} pointing to {base_branch}")
                return False  # No changes imported

        except GitError:
            raise
        except (OSError, IOError) as e:
            raise GitError(f"Failed to import branch: {e}")
        finally:
            try:
                if lock_fh:
                    # Release lock
                    if sys.platform.startswith("win"):
                        try:
                            import msvcrt
                            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                        except Exception:
                            pass
                    else:
                        try:
                            import fcntl
                            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
                    lock_fh.close()
            except Exception:
                pass

    async def cleanup_workspace(self, workspace_dir: Path) -> None:
        """
        Remove temporary workspace directory.

        Args:
            workspace_dir: Directory to remove
        """
        try:
            if workspace_dir.exists():
                await asyncio.to_thread(shutil.rmtree, workspace_dir)
                logger.info(f"Cleaned up workspace {workspace_dir}")
        except (OSError, IOError, shutil.Error) as e:
            logger.warning(f"Failed to cleanup workspace {workspace_dir}: {e}")

    async def _run_command(self, cmd: list[str]) -> Tuple[int, str]:
        """
        Run a git command and return exit code and output.

        Args:
            cmd: Command parts to execute

        Returns:
            Tuple of (exit_code, output)
        """
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            logger.debug("Process created, waiting for completion...")
            stdout, _ = await process.communicate()
            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            logger.debug(
                f"Command completed with code {process.returncode}: {output[:200]}"
            )
            return process.returncode or 0, output

        except (OSError, asyncio.SubprocessError) as e:
            logger.error(f"Command failed: {e}")
            return 1, str(e)
