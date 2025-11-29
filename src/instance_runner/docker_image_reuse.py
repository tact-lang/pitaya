"""Docker image check and reuse helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import docker
from docker.errors import ImageNotFound

from .docker_context import CreateContext
from ..exceptions import DockerError

logger = logging.getLogger(__name__)


def _emit_event(ctx: CreateContext, event_type: str, data: dict) -> None:
    if ctx.event_callback:
        try:
            ctx.event_callback({"type": event_type, "data": data})
        except Exception:
            pass


async def ensure_image(ctx: CreateContext) -> None:
    loop = ctx.loop
    try:
        img_check_start = time.monotonic()
        logger.info(f"Checking Docker image exists: {ctx.image}")
        _emit_event(ctx, "instance.container_image_check", {"image": ctx.image})
        img_future = loop.run_in_executor(
            None, lambda: ctx.manager.client.images.get(ctx.image)
        )
        await asyncio.wait_for(img_future, timeout=10)
        logger.info(
            f"Docker image found: {ctx.image} (%.2fs)",
            time.monotonic() - img_check_start,
        )
    except asyncio.TimeoutError:
        _emit_event(
            ctx,
            "instance.container_image_check_timeout",
            {"image": ctx.image, "timeout_s": 10},
        )
        raise DockerError(
            "Docker image check timed out after 10s. Docker daemon may be unresponsive; try restarting Docker Desktop."
        )
    except ImageNotFound:
        raise DockerError(
            f"Docker image '{ctx.image}' not found locally. Build it with: docker build -t {ctx.image} ."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        raise DockerError(f"Failed to inspect Docker image '{ctx.image}': {e}")


async def try_reuse_container(ctx: CreateContext) -> Optional[Any]:
    loop = ctx.loop
    if not ctx.reuse_container:
        return None
    try:
        exist_check_start = time.monotonic()
        logger.info(f"Checking for existing container: {ctx.container_name}")
        get_future = loop.run_in_executor(
            None, lambda: ctx.manager.client.containers.get(ctx.container_name)
        )
        existing = await asyncio.wait_for(get_future, timeout=10)

        await asyncio.wait_for(loop.run_in_executor(None, existing.reload), timeout=10)
        logger.info(
            f"Found existing container {ctx.container_name} with status {existing.status} (%.2fs)",
            time.monotonic() - exist_check_start,
        )

        require_ro = str(ctx.import_policy).lower() == "never"

        def _workspace_is_ro(cont) -> bool:
            try:
                mounts = cont.attrs.get("Mounts", [])
                for m in mounts:
                    if str(m.get("Destination")) == "/workspace":
                        return not bool(m.get("RW", True))
            except Exception:
                pass
            return False

        if require_ro and not _workspace_is_ro(existing):
            try:
                old_id = getattr(existing, "id", "")[:12]
                _emit_event(
                    ctx,
                    "runner.container.replaced",
                    {"old_id": old_id, "reason": "ro_required"},
                )
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.stop), timeout=15
                    )
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.remove), timeout=15
                    )
                except Exception:
                    pass
                logger.info("Replaced existing container to enforce RO /workspace")
            except Exception:
                logger.warning("Failed to replace existing container; creating new")
            return None

        if existing.status == "running":
            logger.info(f"Reusing running container {ctx.container_name}")
            return existing
        if existing.status in ["exited", "created"]:
            logger.info(f"Starting existing container {ctx.container_name}")
            await asyncio.wait_for(
                loop.run_in_executor(None, existing.start), timeout=20
            )
            for _ in range(10):
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, existing.reload)
                if existing.status == "running":
                    logger.info(f"Existing container {ctx.container_name} is running")
                    return existing
            raise DockerError(
                f"Failed to start existing container {ctx.container_name}"
            )

        logger.info(
            f"Container {ctx.container_name} in unexpected state {existing.status}, creating new"
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout while checking/starting existing container; creating new"
        )
    except docker.errors.NotFound:
        logger.debug(f"Container {ctx.container_name} not found, creating new")
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        logger.warning(f"Error checking existing container: {e}, creating new")
    return None


__all__ = ["ensure_image", "try_reuse_container"]
