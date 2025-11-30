"""Cleanup and verification helpers for DockerManager."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import docker

from .. import DockerError


async def stop_container(manager, container, timeout: int | None = None) -> None:
    loop = asyncio.get_event_loop()
    eff_timeout = int(timeout) if timeout is not None else 1
    try:
        await loop.run_in_executor(
            None, lambda: container.stop(timeout=max(0, eff_timeout))
        )
        manager.logger.info(
            f"Stopped container {container.name} gracefully"
        )  # noqa: SLF001
    except manager.NotFound:  # type: ignore[attr-defined]
        manager.logger.debug(f"Container {container.name} not found")  # noqa: SLF001
    except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
        manager.logger.error(
            f"Failed to stop container {container.name}: {exc}"
        )  # noqa: SLF001


async def verify_container_tools(manager, container, tools: list[str]) -> None:
    loop = asyncio.get_event_loop()
    missing = []
    for tool in tools:
        found = False
        for candidate in (tool, f"/usr/local/bin/{tool}"):
            try:
                exec_id = await loop.run_in_executor(
                    None,
                    lambda: container.client.api.exec_create(
                        container.id,
                        f"{candidate} --version",
                        stdout=True,
                        stderr=True,
                    ),
                )
                await loop.run_in_executor(
                    None,
                    lambda: container.client.api.exec_start(
                        exec_id["Id"], stream=False
                    ),
                )
                inspect = await loop.run_in_executor(
                    None,
                    lambda: container.client.api.exec_inspect(exec_id["Id"]),
                )
                if inspect.get("ExitCode", 1) == 0:
                    found = True
                    break
            except Exception:
                continue
        if not found:
            missing.append(tool)
    if missing:
        raise DockerError(f"Required tools missing in container: {', '.join(missing)}")


async def cleanup_container(
    manager,
    container,
    force: bool = True,
    *,
    remove_home_volume: bool = False,
) -> None:
    loop = asyncio.get_event_loop()

    home_volume_name: Optional[str] = None
    try:
        await loop.run_in_executor(None, container.reload)
        mounts = (container.attrs or {}).get("Mounts", []) or []
        for mount in mounts:
            try:
                mtype = str(mount.get("Type") or mount.get("type") or "").lower()
                if mtype != "volume":
                    continue
                dest = (
                    mount.get("Destination")
                    or mount.get("Target")
                    or mount.get("Mountpoint")
                    or ""
                )
                if str(dest) == "/home/node":
                    name = mount.get("Name") or mount.get("name") or mount.get("Source")
                    if name:
                        home_volume_name = str(name)
                        break
            except Exception:
                continue
    except Exception:
        pass

    await loop.run_in_executor(None, container.reload)
    if container.status == "running":
        await stop_container(manager, container)

    await loop.run_in_executor(None, lambda: container.remove(force=force))
    manager.logger.info(f"Removed container {container.name}")  # noqa: SLF001

    if remove_home_volume and home_volume_name:
        try:
            name = home_volume_name
            if name.startswith("pitaya_home_") and not re.fullmatch(
                r"pitaya_home_g[0-9a-f]{8}", name
            ):
                volume = await loop.run_in_executor(
                    None, lambda: manager.client.volumes.get(name)
                )
                await loop.run_in_executor(None, lambda: volume.remove(force=True))
                manager.logger.info(f"Removed session volume {name}")  # noqa: SLF001
            else:
                manager.logger.debug(
                    f"Skipping volume removal for {name} (not run-scoped or not Pitaya)"
                )
        except Exception as exc:
            manager.logger.warning(
                f"Failed to remove session volume {home_volume_name}: {exc}"
            )  # noqa: SLF001
