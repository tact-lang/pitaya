"""
Log rotation and cleanup utilities.

Implements 7-day log retention policy as per spec section 6.1.
"""

import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def cleanup_old_logs(logs_dir: Path, retention_days: int = 7) -> int:
    """
    Clean up log directories older than retention period.

    Per spec section 6.1:
    - Logs older than 7 days are automatically deleted
    - Applies to all run directories under logs_dir

    Args:
        logs_dir: Base logs directory
        retention_days: Number of days to retain logs (default: 7)

    Returns:
        Number of directories cleaned up
    """
    if not logs_dir.exists():
        return 0

    cleanup_count = 0
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

    # Find all run directories
    for run_dir in logs_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        try:
            # Extract timestamp from directory name (run_YYYYMMDD_HHMMSS or run_YYYYMMDD_HHMMSS_<short8>)
            dir_name = run_dir.name
            if dir_name.startswith("run_"):
                timestamp_full = dir_name[4:]
                # Only take the fixed-length timestamp prefix (15 chars)
                timestamp_str = timestamp_full[:15]
                # Parse timestamp
                dir_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                dir_timestamp = dir_timestamp.replace(tzinfo=timezone.utc)

                # Check if older than retention period
                if dir_timestamp < cutoff_date:
                    logger.info(f"Removing old log directory: {run_dir}")
                    shutil.rmtree(run_dir)
                    cleanup_count += 1

        except (ValueError, OSError) as e:
            logger.warning(f"Failed to process log directory {run_dir}: {e}")
            continue

    return cleanup_count


def rotate_log_file(
    log_file: Path, max_size_mb: int = 100, backup_count: int = 5
) -> None:
    """
    Rotate a log file if it exceeds the maximum size.

    Args:
        log_file: Path to the log file
        max_size_mb: Maximum size in megabytes before rotation
        backup_count: Number of backup files to keep
    """
    if not log_file.exists():
        return

    # Check file size
    size_mb = log_file.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return

    # Rotate files
    logger.info(f"Rotating log file {log_file} (size: {size_mb:.1f}MB)")

    # Remove oldest backup if at limit
    oldest_backup = log_file.parent / f"{log_file.name}.{backup_count}"
    if oldest_backup.exists():
        oldest_backup.unlink()

    # Shift existing backups
    for i in range(backup_count - 1, 0, -1):
        current = log_file.parent / f"{log_file.name}.{i}"
        next_num = log_file.parent / f"{log_file.name}.{i + 1}"
        if current.exists():
            current.rename(next_num)

    # Move current file to .1
    backup_1 = log_file.parent / f"{log_file.name}.1"
    log_file.rename(backup_1)

    # Create new empty file
    log_file.touch()


def get_log_directories_by_age(logs_dir: Path) -> List[tuple[Path, datetime]]:
    """
    Get all log directories sorted by age.

    Args:
        logs_dir: Base logs directory

    Returns:
        List of (directory_path, timestamp) tuples sorted by age (oldest first)
    """
    directories: List[Tuple[Path, datetime]] = []

    if not logs_dir.exists():
        return directories

    for run_dir in logs_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        try:
            # Extract timestamp (support optional short8 suffix)
            dir_name = run_dir.name
            if dir_name.startswith("run_"):
                timestamp_full = dir_name[4:]
                timestamp_str = timestamp_full[:15]
                dir_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                dir_timestamp = dir_timestamp.replace(tzinfo=timezone.utc)
                directories.append((run_dir, dir_timestamp))
        except ValueError:
            continue

    # Sort by timestamp (oldest first)
    directories.sort(key=lambda x: x[1])
    return directories


async def setup_log_rotation_task(
    logs_dir: Path, interval_hours: int = 24, max_size_mb: int = 100
) -> None:
    """
    Set up periodic log rotation task.

    This would typically be called from the main application startup
    to schedule regular cleanup.

    Args:
        logs_dir: Base logs directory
        interval_hours: How often to run cleanup (default: 24 hours)
    """
    import asyncio

    async def rotation_task():
        while True:
            try:
                # Run cleanup
                cleanup_count = cleanup_old_logs(logs_dir)
                if cleanup_count > 0:
                    logger.info(
                        f"Log cleanup completed: removed {cleanup_count} old directories"
                    )

                # Rotate oversized JSONL logs across run directories
                try:
                    for run_dir in logs_dir.glob("run_*"):
                        if not run_dir.is_dir():
                            continue
                        for lf in run_dir.glob(
                            "*.jsonl"
                        ):  # orchestration.jsonl, runner.jsonl, tui.jsonl, other.jsonl
                            rotate_log_file(lf, max_size_mb=max_size_mb)
                except Exception as e:
                    logger.warning(f"Error during log rotation: {e}")

                # Prune events logs for terminal runs using fixed defaults
                from datetime import timedelta

                try:
                    retention_days = 30
                    grace_days = 7
                    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
                    grace_cutoff = datetime.now(timezone.utc) - timedelta(
                        days=grace_days
                    )
                    for run_dir in logs_dir.glob("run_*"):
                        if not run_dir.is_dir():
                            continue
                        try:
                            name = run_dir.name
                            ts = name[4:19]
                            started = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(
                                tzinfo=timezone.utc
                            )
                        except Exception:
                            continue
                        if started < cutoff and started < grace_cutoff:
                            ev = run_dir / "events.jsonl"
                            rn = run_dir / "runner.jsonl"
                            try:
                                if ev.exists():
                                    ev.unlink()
                                if rn.exists():
                                    rn.unlink()
                            except Exception:
                                pass
                except Exception:
                    pass

                # Sleep until next run
                await asyncio.sleep(interval_hours * 3600)

            except (OSError, IOError) as e:
                logger.error(f"Error in log rotation task: {e}")
                # Continue running even if cleanup fails
                await asyncio.sleep(interval_hours * 3600)

    # Create background task
    asyncio.create_task(rotation_task())
