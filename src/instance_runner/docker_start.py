"""Container creation and start helper."""

from __future__ import annotations

import asyncio
import logging
import time

import docker

from .docker_context import CreateContext
from ..exceptions import DockerError

logger = logging.getLogger(__name__)


def _emit_event(ctx: CreateContext, event_type: str, data: dict) -> None:
    if ctx.event_callback:
        try:
            ctx.event_callback({"type": event_type, "data": data})
        except Exception:
            pass


async def create_and_start(ctx: CreateContext) -> any:
    loop = ctx.loop
    logger.info(
        f"Creating Docker container via API: name={ctx.container_name}, image={ctx.image}"
    )
    _emit_event(
        ctx,
        "instance.container_create_attempt",
        {"container_name": ctx.container_name, "image": ctx.image},
    )
    create_start = time.monotonic()
    create_future = loop.run_in_executor(
        None, lambda: ctx.manager.client.containers.create(**ctx.config)
    )
    container = None
    try:
        container = await asyncio.wait_for(create_future, timeout=20)
        logger.info(
            "Docker create returned id=%s (%.2fs)",
            container.id[:12] if container else "unknown",
            time.monotonic() - create_start,
        )
    except asyncio.TimeoutError:
        _emit_event(
            ctx,
            "instance.container_create_timeout",
            {"container_name": ctx.container_name, "timeout_s": 20},
        )
        raise DockerError(
            f"Docker create timed out after 20s for {ctx.container_name}. Check Docker Desktop and volume sharing."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        msg = str(e)
        status_code = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        is_conflict = False
        try:
            is_conflict = (
                (status_code == 409) or ("already in use" in msg) or ("Conflict" in msg)
            )
        except Exception:
            is_conflict = False

        if is_conflict:
            _emit_event(
                ctx,
                "instance.container_create_conflict",
                {"container_name": ctx.container_name, "error": msg},
            )
            adopt_start = time.monotonic()
            for _ in range(6):
                try:
                    get_future2 = loop.run_in_executor(
                        None,
                        lambda: ctx.manager.client.containers.get(ctx.container_name),
                    )
                    existing2 = await asyncio.wait_for(get_future2, timeout=5)
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(None, existing2.reload), timeout=5
                        )
                    except Exception:
                        pass
                    if existing2.status != "running":
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(None, existing2.start), timeout=15
                            )
                        except Exception:
                            pass
                        for _ in range(10):
                            await asyncio.sleep(0.2)
                            try:
                                await loop.run_in_executor(None, existing2.reload)
                            except Exception:
                                pass
                            if existing2.status == "running":
                                break
                    logger.info(
                        "Adopted existing container %s after conflict (%.2fs)",
                        ctx.container_name,
                        time.monotonic() - adopt_start,
                    )
                    _emit_event(
                        ctx,
                        "instance.container_adopted",
                        {
                            "container_name": ctx.container_name,
                            "container_id": getattr(existing2, "id", "")[:12],
                        },
                    )
                    container = existing2
                    break
                except docker.errors.NotFound:
                    await asyncio.sleep(0.5)
                    continue
                except (docker.errors.APIError, docker.errors.DockerException):
                    await asyncio.sleep(0.5)
                    continue

            if not container:
                _emit_event(
                    ctx,
                    "instance.container_create_failed",
                    {"container_name": ctx.container_name, "error": msg},
                )
                raise DockerError(
                    f"Docker create name conflict for {ctx.container_name} but adoption failed: {e}"
                )
        else:
            _emit_event(
                ctx,
                "instance.container_create_failed",
                {"container_name": ctx.container_name, "error": str(e)},
            )
            raise DockerError(f"Docker create failed for {ctx.container_name}: {e}")

    try:
        await loop.run_in_executor(None, container.reload)
    except Exception:
        pass
    if getattr(container, "status", None) != "running":
        start_start = time.monotonic()
        start_future = loop.run_in_executor(None, container.start)
        try:
            await asyncio.wait_for(start_future, timeout=15)
            logger.info(
                "Docker start completed for %s (%.2fs)",
                container.name,
                time.monotonic() - start_start,
            )
        except asyncio.TimeoutError:
            _emit_event(
                ctx,
                "instance.container_start_timeout",
                {"container_name": ctx.container_name, "timeout_s": 15},
            )
            raise DockerError(
                f"Docker start timed out after 15s for {ctx.container_name}."
            )
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            _emit_event(
                ctx,
                "instance.container_start_failed",
                {"container_name": ctx.container_name, "error": str(e)},
            )
            raise DockerError(f"Docker start failed for {ctx.container_name}: {e}")

    for _ in range(20):
        await asyncio.sleep(0.5)
        await loop.run_in_executor(None, container.reload)
        logger.debug(f"Container status check: {container.status}")
        if container.status == "running":
            break
    else:
        raise DockerError(
            f"Container {ctx.container_name} failed to reach running state"
        )

    logger.info(
        f"Created container {ctx.container_name} (ID: {container.id[:12]}) in %.2fs",
        time.monotonic() - ctx.phase_start,
    )
    _emit_event(
        ctx,
        "instance.container_created",
        {"container_name": ctx.container_name, "container_id": container.id[:12]},
    )
    return container


__all__ = ["create_and_start"]
