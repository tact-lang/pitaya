"""Container creation orchestrator (delegates to helper modules)."""

from __future__ import annotations

import logging

from .docker_config import (
    apply_auth_env,
    apply_plugin_hook,
    build_base_config,
    enforce_mount_policy,
    summarize_env,
)
from .docker_context import CreateContext
from .docker_image_reuse import ensure_image, try_reuse_container
from .docker_mounts import prepare_mounts
from .docker_start import create_and_start

logger = logging.getLogger(__name__)


async def create_container(ctx: CreateContext):
    import time

    ctx.phase_start = time.monotonic()
    logger.info(
        f"create_container entry: name={ctx.container_name}, image={ctx.image}, reuse={ctx.reuse_container}, ws={ctx.workspace_dir}"
    )
    if ctx.event_callback:
        try:
            ctx.event_callback(
                {
                    "type": "instance.container_create_entry",
                    "data": {
                        "container_name": ctx.container_name,
                        "workspace_dir": str(ctx.workspace_dir),
                        "image": ctx.image,
                        "reuse": bool(ctx.reuse_container),
                    },
                }
            )
        except Exception:
            pass

    await ensure_image(ctx)
    reused = await try_reuse_container(ctx)
    if reused:
        return reused

    prepare_mounts(ctx)
    build_base_config(ctx)
    apply_auth_env(ctx)
    await apply_plugin_hook(ctx)
    enforce_mount_policy(ctx)
    summarize_env(ctx)
    return await create_and_start(ctx)


__all__ = ["create_container"]
