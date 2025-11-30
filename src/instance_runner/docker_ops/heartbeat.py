"""Heartbeat helpers for Docker containers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict

from docker.models.containers import Container


def _iso_millis(dt: datetime) -> str:
    s = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return s[:-3] + "Z"


async def start_heartbeat(
    tasks: Dict[str, asyncio.Task],
    container: Container,
    interval_s: float = 15.0,
) -> None:
    """Start a periodic heartbeat writer inside the container."""
    if not container:
        return
    cid = getattr(container, "id", None) or ""
    if not cid or cid in tasks:
        return

    async def _hb() -> None:
        try:
            while True:
                try:
                    exec1 = container.client.api.exec_create(
                        container.id,
                        "sh -lc 'mkdir -p /home/node/.pitaya'",
                        stdout=False,
                        stderr=False,
                    )
                    try:
                        container.client.api.exec_start(exec1["Id"], detach=True)
                    except Exception:
                        pass
                    ts = _iso_millis(datetime.now(timezone.utc))
                    exec2 = container.client.api.exec_create(
                        container.id,
                        f"sh -lc 'printf %s {ts} > /home/node/.pitaya/last_active'",
                        stdout=False,
                        stderr=False,
                    )
                    try:
                        container.client.api.exec_start(exec2["Id"], detach=True)
                    except Exception:
                        pass
                except Exception:
                    pass
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            try:
                ts = _iso_millis(datetime.now(timezone.utc))
                exec3 = container.client.api.exec_create(
                    container.id,
                    f"sh -lc 'printf %s {ts} > /home/node/.pitaya/last_active'",
                    stdout=False,
                    stderr=False,
                )
                try:
                    container.client.api.exec_start(exec3["Id"], detach=True)
                except Exception:
                    pass
            except Exception:
                pass
            return

    tasks[cid] = asyncio.create_task(_hb())


async def stop_heartbeat(tasks: Dict[str, asyncio.Task], container: Container) -> None:
    cid = getattr(container, "id", None) or ""
    task = tasks.pop(cid, None)
    if task:
        task.cancel()
