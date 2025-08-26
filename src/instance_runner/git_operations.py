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
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

from . import GitError
from ..utils.platform_utils import get_temp_dir

logger = logging.getLogger(__name__)

_BRANCH_RE = None


def _is_valid_branch_name(name: str) -> bool:
    """Validate branch name against a strict regex and forbidden substrings.

    Disallows dangerous patterns: '..', '.lock', '@{', leading '-', trailing '/', and invalid chars.
    """
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


def _is_protected_ref(name: str) -> bool:
    """Return True if the ref name is protected (main/master/develop/stable/release/*/hotfix/*)."""
    if not isinstance(name, str):
        return False
    if name in ("main", "master", "develop", "stable"):
        return True
    return name.startswith("release/") or name.startswith("hotfix/")


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
        reuse_if_exists: bool = False,
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
        # Per-user directory under $HOME on macOS; fall back to system temp elsewhere.
        try:
            import platform

            if platform.system() == "Darwin":
                # macOS: /Users is shared by default with Docker Desktop
                base_dir = Path.home() / ".pitaya" / "workspaces"
            else:
                base_dir = get_temp_dir()
        except Exception:
            base_dir = get_temp_dir()

        if run_id and container_name:
            # Derive a stable workspace path per run + strategy + durable key
            # Expected format: pitaya_{run_id}_s{sidx}_k{khash}[_rXXXX...]
            tokens = (container_name or "").split("_")
            sidx = "0"
            khash = ""
            try:
                # Find s{number}
                for t in tokens:
                    if t.startswith("s") and t[1:].isdigit():
                        sidx = t[1:]
                        break
                # Extract k{hash}
                for t in tokens:
                    if t.startswith("k") and len(t) > 1:
                        khash = t[1:]
                        break
                if not khash:
                    # Fallback: short instance_id
                    khash = (instance_id or "")[0:8] or "x"
            except Exception:
                sidx = "0"
                khash = (instance_id or "")[0:8] or "x"

            # Path: {base}/pitaya/${run_id}/i_${sidx}_${khash}_${iid8}
            # Include a short instance_id suffix to avoid collisions when many
            # parallel instances share the same durable key (khash).
            iid8 = (instance_id or "")[:8] or "x"
            workspace_dir = base_dir / f"pitaya/{run_id}/i_{sidx}_{khash}_{iid8}"
        else:
            # Fallback for standalone usage
            workspace_dir = base_dir / f"pitaya/instance_{instance_id}"

        # On Windows, ensure path length remains below typical limits
        try:
            # Validate base branch name (defensive; base must already exist in repo)
            if not _is_valid_branch_name(base_branch):
                raise GitError(f"Invalid base branch name: {base_branch}")
            import platform

            if platform.system() == "Windows":
                # If the final path would be too long, use a shorter form
                if len(str(workspace_dir)) > 200:
                    from uuid import uuid4

                    workspace_dir = base_dir / f"pit/{uuid4().hex[:8]}"
        except Exception:
            pass

        try:
            # Ensure parent directory exists
            await asyncio.to_thread(
                workspace_dir.parent.mkdir, parents=True, exist_ok=True
            )

            # Reuse existing workspace on resume when requested
            if reuse_if_exists and workspace_dir.exists():
                # Validate it's a git repo; if not, fall back to fresh clone
                if (workspace_dir / ".git").exists():
                    logger.info(f"Reusing existing workspace at {workspace_dir}")
                    # Best-effort: ensure BASE_BRANCH marker exists; if missing, write it
                    try:
                        base_branch_file = workspace_dir / ".git" / "BASE_BRANCH"
                        if not base_branch_file.exists():
                            base_branch_file.write_text(base_branch)
                    except Exception:
                        pass
                    return workspace_dir
                else:
                    logger.info(
                        f"Existing workspace at {workspace_dir} is not a git repo; recreating"
                    )
                    await asyncio.to_thread(shutil.rmtree, workspace_dir)

            # Remove any existing workspace when not reusing (fresh prepare)
            if workspace_dir.exists():
                await asyncio.to_thread(shutil.rmtree, workspace_dir)

            logger.info(f"Creating isolated workspace at {workspace_dir}")

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
        *,
        task_key: Optional[str] = None,
        run_id: Optional[str] = None,
        strategy_execution_id: Optional[str] = None,
        allow_overwrite_protected_refs: bool = False,
    ) -> Dict[str, Optional[str]]:
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
            Dict with keys:
              - has_changes (bool as string 'true'/'false' for JSON friendliness handled by caller)
              - target_branch (str)
              - commit (str)
              - duplicate_of_branch (str|None)
              - dedupe_reason (str|None)
        """
        try:
            # Note: We do NOT create commits here. The agent tool handles its own commits.
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

            _raw = (
                _P(out.strip())
                if _P(out.strip()).is_absolute()
                else (repo_path / out.strip())
            )
            # Use realpath (resolve) per spec to avoid duplicate locks on symlinked repos
            git_dir = _raw.resolve()
            lock_path = git_dir / ".pitaya_import.lock"

            # Simple cross-platform exclusive lock with diagnostics and wait-loop
            import sys

            lock_fh = open(lock_path, "a+")
            lock_fh.seek(0)
            # Best-effort metadata for diagnostics
            holder_meta = {
                "pid": os.getpid(),
                "ts": datetime.utcnow().isoformat() + "Z",
                "action": "import_branch",
                "run_id": run_id,
                "strategy_execution_id": strategy_execution_id,
            }
            # Try non-blocking first, then wait with heartbeat
            start_wait = time.monotonic()
            last_log = 0.0
            # No env-based timeout; wait indefinitely for repo lock
            max_wait = None

            if sys.platform.startswith("win"):
                import msvcrt

                # Non-blocking attempt; if fails, sleep and retry
                while True:
                    try:
                        msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except Exception:
                        now = time.monotonic()
                        if (now - last_log) >= 2.0:
                            logger.info(
                                f"waiting_for_repo_lock path={lock_path} held_by={_read_lock_holder(lock_path)} waited_s={(now-start_wait):.1f}"
                            )
                            last_log = now
                        if max_wait and (now - start_wait) > max_wait:
                            raise GitError(
                                f"Timeout waiting for repo import lock after {max_wait:.1f}s"
                            )
                        await asyncio.sleep(0.2)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        now = time.monotonic()
                        if (now - last_log) >= 2.0:
                            logger.info(
                                f"waiting_for_repo_lock path={lock_path} held_by={_read_lock_holder(lock_path)} waited_s={(now-start_wait):.1f}"
                            )
                            last_log = now
                        if max_wait and (now - start_wait) > max_wait:
                            raise GitError(
                                f"Timeout waiting for repo import lock after {max_wait:.1f}s"
                            )
                        await asyncio.sleep(0.2)

            # Write holder metadata (best-effort) once acquired
            try:
                lock_fh.seek(0)
                lock_fh.truncate(0)
                lock_fh.write(json.dumps(holder_meta))
                lock_fh.flush()
            except Exception:
                pass

            # Validate target branch name and protected refs before any ref updates
            if not _is_valid_branch_name(branch_name):
                raise GitError(f"Invalid target branch name: {branch_name}")

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

            # Disallow creating/updating protected refs unless explicitly allowed
            if _is_protected_ref(branch_name) and not allow_overwrite_protected_refs:
                raise GitError(
                    f"Refusing to update protected ref '{branch_name}'. Use --allow-overwrite-protected-refs to allow."
                )

            # Read preserved base commit
            base_commit_file = workspace_dir / ".git" / "BASE_COMMIT"
            base_commit = (
                base_commit_file.read_text().strip()
                if base_commit_file.exists()
                else ""
            )

            # Determine if workspace has new commits relative to base
            ws_head_cmd = ["git", "-C", str(workspace_dir), "rev-parse", "HEAD"]
            rc, ws_head_out = await self._run_command(ws_head_cmd)
            if rc != 0:
                raise GitError(f"Failed to get workspace HEAD: {ws_head_out}")
            workspace_head = ws_head_out.strip()

            dedupe_reason: Optional[str] = None
            duplicate_of_branch: Optional[str] = None

            # If branch exists, idempotent pre-check: identical heads → treat as success (same_task_provenance or resume-with-note)
            if branch_exists:
                br_head_cmd = [
                    "git",
                    "-C",
                    str(repo_path),
                    "rev-parse",
                    f"refs/heads/{branch_name}",
                ]
                rc, br_head_out = await self._run_command(br_head_cmd)
                if rc == 0 and br_head_out.strip() == workspace_head:
                    # Check provenance note for same task_key
                    note_rc, note_out = await self._run_command(
                        [
                            "git",
                            "-C",
                            str(repo_path),
                            "notes",
                            "--ref=pitaya",
                            "show",
                            workspace_head,
                        ]
                    )
                    if (
                        note_rc == 0
                        and task_key
                        and (f"task_key={task_key}" in note_out)
                    ):
                        dedupe_reason = "same_task_provenance"
                    else:
                        # Treat as resume of the same task: append provenance now under the lock
                        try:
                            _ts = (
                                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                                + "Z"
                            )
                            note_lines = [
                                f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={branch_name}; ts={_ts}"
                            ]
                            await self._run_command(
                                [
                                    "git",
                                    "-C",
                                    str(repo_path),
                                    "notes",
                                    "--ref=pitaya",
                                    "add",
                                    "-f",
                                    "-m",
                                    "\n".join(note_lines),
                                    workspace_head,
                                ]
                            )
                        except Exception:
                            pass
                        dedupe_reason = "by_commit"
                    logger.info(
                        "Branch already matches workspace HEAD; idempotent success"
                    )
                    return {
                        "has_changes": "true",
                        "target_branch": branch_name,
                        "commit": workspace_head,
                        "duplicate_of_branch": duplicate_of_branch,
                        "dedupe_reason": dedupe_reason,
                    }

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

                            if (
                                re.match(rf"^{branch_name}(_[0-9]+)?$", ref)
                                and obj == workspace_head
                            ):
                                # Check provenance for this task_key
                                note_rc, note_out = await self._run_command(
                                    [
                                        "git",
                                        "-C",
                                        str(repo_path),
                                        "notes",
                                        "--ref=pitaya",
                                        "show",
                                        workspace_head,
                                    ]
                                )
                                if (
                                    note_rc == 0
                                    and task_key
                                    and (f"task_key={task_key}" in note_out)
                                ):
                                    dedupe_reason = "same_task_provenance"
                                    logger.info(
                                        f"Found existing suffixed branch {ref} with matching HEAD and provenance; idempotent success"
                                    )
                                    return {
                                        "has_changes": "true",
                                        "target_branch": ref,
                                        "commit": workspace_head,
                                        "duplicate_of_branch": ref,
                                        "dedupe_reason": dedupe_reason,
                                    }
                                else:
                                    # Treat as crash_window resume: append provenance NOW and return existing
                                    try:
                                        _ts = (
                                            datetime.utcnow().strftime(
                                                "%Y-%m-%dT%H:%M:%S.%f"
                                            )[:-3]
                                            + "Z"
                                        )
                                        note_lines = [
                                            f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={ref}; ts={_ts}"
                                        ]
                                        await self._run_command(
                                            [
                                                "git",
                                                "-C",
                                                str(repo_path),
                                                "notes",
                                                "--ref=pitaya",
                                                "add",
                                                "-f",
                                                "-m",
                                                "\n".join(note_lines),
                                                workspace_head,
                                            ]
                                        )
                                    except Exception:
                                        pass
                                    duplicate_of_branch = ref
                                    dedupe_reason = "crash_window"
                                    logger.info(
                                        f"Found existing suffixed branch {ref} with matching HEAD but missing provenance; treating as resume idempotently"
                                    )
                                    return {
                                        "has_changes": "true",
                                        "target_branch": ref,
                                        "commit": workspace_head,
                                        "duplicate_of_branch": ref,
                                        "dedupe_reason": dedupe_reason,
                                    }
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
                return {
                    "has_changes": "false",
                    "target_branch": None,
                    "commit": base_commit or workspace_head,
                    "duplicate_of_branch": None,
                    "dedupe_reason": None,
                }
            if import_policy == "auto" and skip_empty_import and not has_new_commits:
                logger.info("No changes; skipping branch creation per policy")
                return {
                    "has_changes": "false",
                    "target_branch": None,
                    "commit": base_commit or workspace_head,
                    "duplicate_of_branch": None,
                    "dedupe_reason": "by_commit_no_changes",
                }

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
                    # Only allow forced updates for non-protected refs (explicit opt-in)
                    if (
                        _is_protected_ref(target_branch)
                        and not allow_overwrite_protected_refs
                    ):
                        raise GitError(
                            f"Refusing to overwrite protected ref '{target_branch}'. Use --allow-overwrite-protected-refs to allow."
                        )
                elif import_conflict_policy == "suffix":
                    # Find next available suffix
                    i = 2
                    while True:
                        check_cmd = [
                            "git",
                            "-C",
                            str(repo_path),
                            "show-ref",
                            "--verify",
                            "--quiet",
                            f"refs/heads/{branch_name}_{i}",
                        ]
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
                    (
                        f"+HEAD:{target_branch}"
                        if import_conflict_policy == "overwrite"
                        else f"HEAD:{target_branch}"
                    ),
                ]

                result = await self._run_command(fetch_cmd)
                if result[0] != 0:
                    raise GitError(f"Failed to import branch: {result[1]}")

                logger.info(f"Successfully imported branch {target_branch}")
                # Write provenance note under lock
                try:
                    # UTC ISO-8601 with milliseconds + Z for ts
                    _ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    note_lines = [
                        f"task_key={task_key}; run_id={run_id}; strategy_execution_id={strategy_execution_id}; branch={target_branch}; ts={_ts}"
                    ]
                    await self._run_command(
                        [
                            "git",
                            "-C",
                            str(repo_path),
                            "notes",
                            "--ref=pitaya",
                            "add",
                            "-f",
                            "-m",
                            "\n".join(note_lines),
                            workspace_head,
                        ]
                    )
                except Exception:
                    pass
                return {
                    "has_changes": "true",
                    "target_branch": target_branch,
                    "commit": workspace_head,
                    "duplicate_of_branch": duplicate_of_branch,
                    "dedupe_reason": dedupe_reason,
                }
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
                return {
                    "has_changes": "false",
                    "target_branch": target_branch,
                    "commit": base_sha,
                    "duplicate_of_branch": duplicate_of_branch,
                    "dedupe_reason": "by_commit_no_changes",
                }

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
                    # Clear holder metadata (best-effort) and close
                    try:
                        lock_fh.seek(0)
                        lock_fh.truncate(0)
                        lock_fh.flush()
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


# Helper to read current holder metadata of lock (best-effort, may return raw string)
def _read_lock_holder(lock_path: Path) -> str:
    try:
        with open(lock_path, "r") as fh:
            data = fh.read().strip()
            if not data:
                return "{}"
            try:
                obj = json.loads(data)
                pid = obj.get("pid")
                ts = obj.get("ts")
                act = obj.get("action")
                return f"pid={pid} ts={ts} action={act}"
            except Exception:
                return data[:120]
    except Exception:
        return "<unavailable>"
