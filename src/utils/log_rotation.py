"""Log rotation and cleanup utilities."""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "cleanup_old_logs",
    "get_log_directories_by_age",
    "rotate_log_file",
    "setup_log_rotation_task",
]

_RUN_DIR_PREFIX = "run_"
_TIMESTAMP_SLICE = slice(4, 19)  # run_YYYYMMDD_HHMMSS
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_DEFAULT_RETENTION_DAYS = 7
_DEFAULT_MAX_SIZE_MB = 100
_DEFAULT_BACKUP_COUNT = 5
_EVENT_RETENTION_DAYS = 30
_EVENT_GRACE_DAYS = 7


def cleanup_old_logs(
    logs_dir: Path, retention_days: int = _DEFAULT_RETENTION_DAYS
) -> int:
    """Remove log directories older than ``retention_days``."""
    if not logs_dir.exists():
        return 0

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cleanup_count = 0

    for run_dir in _iter_run_directories(logs_dir):
        timestamp = _extract_run_timestamp(run_dir)
        if timestamp is None:
            logger.warning("Failed to parse timestamp for %s", run_dir)
            continue
        if timestamp < cutoff_date:
            try:
                logger.info("Removing old log directory: %s", run_dir)
                shutil.rmtree(run_dir)
                cleanup_count += 1
            except OSError as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to remove %s: %s", run_dir, exc)

    return cleanup_count


def rotate_log_file(
    log_file: Path,
    max_size_mb: int = _DEFAULT_MAX_SIZE_MB,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> None:
    """Rotate ``log_file`` when it exceeds ``max_size_mb``."""
    if not log_file.exists():
        return

    size_mb = log_file.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return

    logger.info("Rotating log file %s (size: %.1fMB)", log_file, size_mb)
    _rotate_backups(log_file, backup_count)
    log_file.rename(log_file.parent / f"{log_file.name}.1")
    log_file.touch()


def get_log_directories_by_age(logs_dir: Path) -> List[Tuple[Path, datetime]]:
    """Return ``(path, timestamp)`` entries for run logs sorted oldest first."""
    entries: List[Tuple[Path, datetime]] = []
    if not logs_dir.exists():
        return entries

    for run_dir in _iter_run_directories(logs_dir):
        timestamp = _extract_run_timestamp(run_dir)
        if timestamp is not None:
            entries.append((run_dir, timestamp))

    entries.sort(key=lambda item: item[1])
    return entries


async def setup_log_rotation_task(
    logs_dir: Path,
    interval_hours: int = 24,
    max_size_mb: int = _DEFAULT_MAX_SIZE_MB,
) -> None:
    """Launch a background task that periodically rotates and prunes logs."""

    async def _rotation_loop() -> None:
        sleep_seconds = interval_hours * 3600
        while True:
            try:
                _run_rotation_cycle(logs_dir, max_size_mb)
            except OSError as exc:
                logger.error("Error in log rotation task: %s", exc)
            await asyncio.sleep(sleep_seconds)

    asyncio.create_task(_rotation_loop())


def _run_rotation_cycle(logs_dir: Path, max_size_mb: int) -> None:
    removed = cleanup_old_logs(logs_dir)
    if removed:
        logger.info("Log cleanup completed: removed %s old directories", removed)

    for run_dir in _iter_run_directories(logs_dir):
        _rotate_run_logs(run_dir, max_size_mb)

    _prune_terminal_events(logs_dir)


def _iter_run_directories(logs_dir: Path) -> Iterable[Path]:
    return (path for path in logs_dir.glob(f"{_RUN_DIR_PREFIX}*") if path.is_dir())


def _extract_run_timestamp(run_dir: Path) -> Optional[datetime]:
    name = run_dir.name
    if not name.startswith(_RUN_DIR_PREFIX):
        return None
    timestamp_raw = name[_TIMESTAMP_SLICE]
    try:
        dt = datetime.strptime(timestamp_raw, _TIMESTAMP_FORMAT)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _rotate_backups(log_file: Path, backup_count: int) -> None:
    oldest_backup = log_file.parent / f"{log_file.name}.{backup_count}"
    if oldest_backup.exists():
        oldest_backup.unlink()

    for index in range(backup_count - 1, 0, -1):
        current = log_file.parent / f"{log_file.name}.{index}"
        if current.exists():
            current.rename(log_file.parent / f"{log_file.name}.{index + 1}")


def _rotate_run_logs(run_dir: Path, max_size_mb: int) -> None:
    for log_file in run_dir.glob("*.jsonl"):
        try:
            rotate_log_file(log_file, max_size_mb=max_size_mb)
        except OSError as exc:  # pragma: no cover - best effort rotation
            logger.warning("Error rotating %s: %s", log_file, exc)


def _prune_terminal_events(logs_dir: Path) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_EVENT_RETENTION_DAYS)
    grace_cutoff = datetime.now(timezone.utc) - timedelta(days=_EVENT_GRACE_DAYS)

    for run_dir in _iter_run_directories(logs_dir):
        timestamp = _extract_run_timestamp(run_dir)
        if timestamp is None or timestamp >= cutoff or timestamp >= grace_cutoff:
            continue

        for filename in ("events.jsonl", "runner.jsonl"):
            candidate = run_dir / filename
            if candidate.exists():
                try:
                    candidate.unlink()
                except OSError:  # pragma: no cover - best effort pruning
                    pass
