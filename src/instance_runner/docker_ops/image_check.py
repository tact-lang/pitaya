"""Image validation helper."""

from __future__ import annotations

import asyncio
import time
import docker
from docker.errors import ImageNotFound

from .. import DockerError


async def ensure_image_exists(manager, loop, image: str, event_callback) -> None:
    try:
        img_check_start = time.monotonic()
        manager._emit_event(
            event_callback, "instance.container_image_check", {"image": image}
        )  # noqa: SLF001
        img_future = loop.run_in_executor(
            None, lambda: manager.client.images.get(image)
        )
        await asyncio.wait_for(img_future, timeout=10)
        manager.logger.info(  # noqa: SLF001
            f"Docker image found: {image} (%.2fs)", time.monotonic() - img_check_start
        )
    except asyncio.TimeoutError:
        manager._emit_event(  # noqa: SLF001
            event_callback,
            "instance.container_image_check_timeout",
            {"image": image, "timeout_s": 10},
        )
        raise DockerError(
            "Docker image check timed out after 10s. Docker daemon may be unresponsive; try restarting Docker Desktop."
        )
    except ImageNotFound:
        raise DockerError(
            f"Docker image '{image}' not found locally. Build it with: docker build -t {image} ."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
        raise DockerError(f"Failed to inspect Docker image '{image}': {exc}")
