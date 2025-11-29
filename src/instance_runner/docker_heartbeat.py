"""Heartbeat helpers for DockerManager."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _iso_millis(dt: datetime) -> str:
    stamp = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return stamp[:-3] + "Z"


async def start_heartbeat(
    manager: Any, container: Any, interval_s: float = 15.0
) -> None:
    """Start periodic heartbeat writer inside the container."""
    if not container:
        return
    cid = getattr(container, "id", None) or ""
    if not cid or cid in manager._hb_tasks:
        return

    async def _hb():
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
                        logger.debug(
                            "heartbeat: exec_create mkdir id=%s",
                            exec1.get("Id", "-"),
                        )
                    except Exception:
                        pass
                    try:
                        container.client.api.exec_start(exec1["Id"], detach=True)
                        logger.debug(
                            "heartbeat: exec_start mkdir ok id=%s",
                            exec1.get("Id", "-"),
                        )
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
                        logger.debug(
                            "heartbeat: exec_create write id=%s ts=%s",
                            exec2.get("Id", "-"),
                            ts,
                        )
                    except Exception:
                        pass
                    try:
                        container.client.api.exec_start(exec2["Id"], detach=True)
                        logger.debug(
                            "heartbeat: exec_start write ok id=%s",
                            exec2.get("Id", "-"),
                        )
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
                    logger.debug(
                        "heartbeat: final exec_create id=%s ts=%s",
                        exec3.get("Id", "-"),
                        ts,
                    )
                except Exception:
                    pass
                try:
                    container.client.api.exec_start(exec3["Id"], detach=True)
                    logger.debug(
                        "heartbeat: final exec_start ok id=%s",
                        exec3.get("Id", "-"),
                    )
                except Exception:
                    pass
            except Exception:
                pass
            return

    manager._hb_tasks[cid] = asyncio.create_task(_hb())


async def stop_heartbeat(manager: Any, container: Any) -> None:
    cid = getattr(container, "id", None) or ""
    task = manager._hb_tasks.pop(cid, None)
    if task:
        task.cancel()


__all__ = ["start_heartbeat", "stop_heartbeat"]
