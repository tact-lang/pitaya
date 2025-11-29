"""Import lock helpers for git operations."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

from . import GitError
from .git_utils import read_lock_holder


async def acquire_import_lock(
    repo_path: Path,
    git_ops,
    *,
    run_id: str | None,
    strategy_execution_id: str | None,
) -> Tuple[Path, any]:
    git_dir_cmd = ["git", "-C", str(repo_path), "rev-parse", "--git-dir"]
    rc, out = await git_ops._run_command(git_dir_cmd)
    if rc != 0:
        raise GitError(f"Failed to resolve git dir: {out}")
    _raw = (
        Path(out.strip())
        if Path(out.strip()).is_absolute()
        else (repo_path / out.strip())
    )
    git_dir = _raw.resolve()
    lock_path = git_dir / ".pitaya_import.lock"

    lock_fh = open(lock_path, "a+")
    lock_fh.seek(0)
    holder_meta = {
        "pid": os.getpid(),
        "ts": datetime.utcnow().isoformat() + "Z",
        "action": "import_branch",
        "run_id": run_id,
        "strategy_execution_id": strategy_execution_id,
    }

    start_wait = time.monotonic()
    last_log = 0.0
    max_wait = None

    if sys.platform.startswith("win"):
        import msvcrt

        while True:
            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except Exception:
                now = time.monotonic()
                if (now - last_log) >= 2.0:
                    print(
                        f"waiting_for_repo_lock path={lock_path} held_by={read_lock_holder(lock_path)} waited_s={(now-start_wait):.1f}"
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
                    print(
                        f"waiting_for_repo_lock path={lock_path} held_by={read_lock_holder(lock_path)} waited_s={(now-start_wait):.1f}"
                    )
                    last_log = now
                if max_wait and (now - start_wait) > max_wait:
                    raise GitError(
                        f"Timeout waiting for repo import lock after {max_wait:.1f}s"
                    )
                await asyncio.sleep(0.2)

    try:
        lock_fh.seek(0)
        lock_fh.truncate(0)
        lock_fh.write(json.dumps(holder_meta))
        lock_fh.flush()
    except Exception:
        pass

    return lock_path, lock_fh


def release_import_lock(lock_fh) -> None:
    try:
        if not lock_fh:
            return
        if sys.platform.startswith("win"):
            import msvcrt

            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            lock_fh.seek(0)
            lock_fh.truncate(0)
            lock_fh.flush()
        except Exception:
            pass
        lock_fh.close()
    except Exception:
        pass


__all__ = ["acquire_import_lock", "release_import_lock"]
