"""Container reuse and lifecycle helpers."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import docker

from .. import DockerError


async def reuse_existing_container(
    manager,
    *,
    loop,
    container_name: str,
    reuse_container: bool,
    import_policy: str,
    event_callback: Optional[Callable],
    workspace_ro_checker,
):
    if not reuse_container:
        return None

    try:
        exist_check_start = time.monotonic()
        get_future = loop.run_in_executor(
            None, lambda: manager.client.containers.get(container_name)
        )
        existing = await asyncio.wait_for(get_future, timeout=10)
        await asyncio.wait_for(loop.run_in_executor(None, existing.reload), timeout=10)
        manager.logger.info(  # noqa: SLF001
            f"Found existing container {container_name} with status {existing.status} (%.2fs)",
            time.monotonic() - exist_check_start,
        )

        require_ro = str(import_policy).lower() == "never"
        if require_ro and not workspace_ro_checker(existing):
            try:
                old_id = getattr(existing, "id", "")[:12]
                manager._emit_event(  # noqa: SLF001
                    event_callback,
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
            except Exception:
                manager.logger.warning(
                    "Failed to replace existing container; creating new"
                )  # noqa: SLF001
            return None

        if existing.status == "running":
            manager.logger.info(
                f"Reusing running container {container_name}"
            )  # noqa: SLF001
            return existing

        if existing.status in ["exited", "created"]:
            manager.logger.info(
                f"Starting existing container {container_name}"
            )  # noqa: SLF001
            await asyncio.wait_for(
                loop.run_in_executor(None, existing.start), timeout=20
            )
            for _ in range(10):
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, existing.reload)
                if existing.status == "running":
                    manager.logger.info(  # noqa: SLF001
                        f"Existing container {container_name} is running"
                    )
                    return existing
            raise DockerError(f"Failed to start existing container {container_name}")

        manager.logger.info(  # noqa: SLF001
            f"Container {container_name} in unexpected state {existing.status}, creating new"
        )
    except asyncio.TimeoutError:
        manager.logger.warning(  # noqa: SLF001
            "Timeout while checking/starting existing container; proceeding to create new"
        )
    except manager.NotFound:  # type: ignore[attr-defined]
        manager.logger.debug(
            f"Container {container_name} not found, creating new"
        )  # noqa: SLF001
    except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
        manager.logger.warning(
            f"Error checking existing container: {exc}, creating new"
        )  # noqa: SLF001

    return None


async def create_container_instance(
    manager,
    *,
    loop,
    config,
    container_name: str,
    image: str,
    event_callback: Optional[Callable],
):
    manager._emit_event(  # noqa: SLF001
        event_callback,
        "instance.container_create_attempt",
        {"container_name": container_name, "image": image},
    )
    create_start = time.monotonic()
    create_future = loop.run_in_executor(
        None, lambda: manager.client.containers.create(**config)
    )
    try:
        container = await asyncio.wait_for(create_future, timeout=20)
        manager.logger.info(  # noqa: SLF001
            "Docker create returned id=%s (%.2fs)",
            container.id[:12] if container else "unknown",
            time.monotonic() - create_start,
        )
        return container
    except asyncio.TimeoutError:
        manager._emit_event(  # noqa: SLF001
            event_callback,
            "instance.container_create_timeout",
            {"container_name": container_name, "timeout_s": 20},
        )
        raise DockerError(
            f"Docker create timed out after 20s for {container_name}. Check Docker Desktop and volume sharing."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
        msg = str(exc)
        status_code = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        try:
            is_conflict = (
                status_code == 409 or ("already in use" in msg) or ("Conflict" in msg)
            )
        except Exception:
            is_conflict = False

        if is_conflict:
            manager._emit_event(  # noqa: SLF001
                event_callback,
                "instance.container_create_conflict",
                {"container_name": container_name, "error": msg},
            )
            adopted = await adopt_existing_container(
                manager,
                loop=loop,
                container_name=container_name,
                event_callback=event_callback,
            )
            if adopted:
                return adopted
            manager._emit_event(  # noqa: SLF001
                event_callback,
                "instance.container_create_failed",
                {"container_name": container_name, "error": msg},
            )
            raise DockerError(
                f"Docker create name conflict for {container_name} but adoption failed: {exc}"
            )

        manager._emit_event(  # noqa: SLF001
            event_callback,
            "instance.container_create_failed",
            {"container_name": container_name, "error": str(exc)},
        )
        raise DockerError(f"Docker create failed for {container_name}: {exc}")


async def adopt_existing_container(
    manager,
    *,
    loop,
    container_name: str,
    event_callback: Optional[Callable],
):
    adopt_start = time.monotonic()
    container = None
    for _ in range(6):
        try:
            get_future = loop.run_in_executor(
                None, lambda: manager.client.containers.get(container_name)
            )
            existing = await asyncio.wait_for(get_future, timeout=5)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, existing.reload), timeout=5
                )
            except Exception:
                pass

            if existing.status != "running":
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.start), timeout=15
                    )
                except Exception:
                    pass
                for _ in range(10):
                    await asyncio.sleep(0.2)
                    try:
                        await loop.run_in_executor(None, existing.reload)
                    except Exception:
                        pass
                    if existing.status == "running":
                        break

            manager.logger.info(  # noqa: SLF001
                "Adopted existing container %s after conflict (%.2fs)",
                container_name,
                time.monotonic() - adopt_start,
            )
            manager._emit_event(  # noqa: SLF001
                event_callback,
                "instance.container_adopted",
                {
                    "container_name": container_name,
                    "container_id": getattr(existing, "id", "")[:12],
                },
            )
            container = existing
            break
        except manager.NotFound:  # type: ignore[attr-defined]
            await asyncio.sleep(0.5)
            continue
        except (docker.errors.APIError, docker.errors.DockerException):  # type: ignore[attr-defined]
            await asyncio.sleep(0.5)
            continue
    return container


async def start_container_and_wait(
    manager,
    *,
    loop,
    container,
    container_name: str,
    event_callback: Optional[Callable],
) -> None:
    try:
        await loop.run_in_executor(None, container.reload)
    except Exception:
        pass

    if getattr(container, "status", None) != "running":
        start_start = time.monotonic()
        start_future = loop.run_in_executor(None, container.start)
        try:
            await asyncio.wait_for(start_future, timeout=15)
            manager.logger.info(  # noqa: SLF001
                "Docker start completed for %s (%.2fs)",
                container.name,
                time.monotonic() - start_start,
            )
        except asyncio.TimeoutError:
            manager._emit_event(  # noqa: SLF001
                event_callback,
                "instance.container_start_timeout",
                {"container_name": container_name, "timeout_s": 15},
            )
            raise DockerError(f"Docker start timed out after 15s for {container_name}.")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
            manager._emit_event(  # noqa: SLF001
                event_callback,
                "instance.container_start_failed",
                {"container_name": container_name, "error": str(exc)},
            )
            raise DockerError(f"Docker start failed for {container_name}: {exc}")

    for i in range(20):
        await asyncio.sleep(0.5)
        await loop.run_in_executor(None, container.reload)
        manager.logger.debug(
            f"Container status check {i}: {container.status}"
        )  # noqa: SLF001
        if container.status == "running":
            break
    else:
        raise DockerError(f"Container {container_name} failed to reach running state")
