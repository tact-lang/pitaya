"""Small helpers for preparing exec runs."""

from __future__ import annotations

import asyncio
from typing import Any


async def ensure_codex_home(
    loop: asyncio.AbstractEventLoop, container, plugin: Any
) -> None:
    try:
        if str(getattr(plugin, "name", "")) != "codex":
            return
        mk_exec = await loop.run_in_executor(
            None,
            lambda: container.client.api.exec_create(
                container.id,
                "sh -lc 'mkdir -p /home/node/.codex'",
                stdout=False,
                stderr=False,
                tty=False,
                workdir="/workspace",
            ),
        )
        try:
            await loop.run_in_executor(
                None,
                lambda: container.client.api.exec_start(mk_exec["Id"], detach=True),
            )
        except Exception:
            pass
    except Exception:
        pass
