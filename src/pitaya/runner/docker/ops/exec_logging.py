"""Logging helpers for exec streams."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def open_stream_log(
    stream_log_path: Optional[str],
    container,
    command: List[str],
    plugin: Any,
    exec_instance: Dict[str, Any],
):
    if not stream_log_path:
        return None

    raw_f = None
    try:
        path = Path(stream_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_f = path.open("a", encoding="utf-8", errors="replace")
        ts = datetime.now(timezone.utc).isoformat()
        try:
            cname = getattr(container, "name", None) or "unknown"
            cid = (getattr(container, "id", None) or "unknown")[:12]
        except Exception:
            cname = "unknown"
            cid = "unknown"
        header = (
            f"=== EXEC START {ts} ===\n"
            f"container_id={cid} name={cname}\n"
            "workdir=/workspace\n"
            f"command={' '.join(command)}\n"
            f"plugin={getattr(plugin, 'name', 'tool')}\n"
        )
        raw_f.write(header)
        raw_f.write(f"exec_id={exec_instance.get('Id')}\n")
        raw_f.flush()
    except Exception:
        raw_f = None
    return raw_f


def write_exec_end(raw_f, exec_info: Dict[str, Any], *, fallback: Any = 0) -> None:
    try:
        if raw_f is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        exit_code = exec_info.get("ExitCode", fallback)
        raw_f.write(f"\n=== EXEC END {ts} exit={exit_code} ===\n")
        raw_f.flush()
    except Exception:
        pass


def close_raw_log(raw_f) -> None:
    try:
        if raw_f is not None:
            raw_f.close()
    except Exception:
        pass
