"""Container cleanup and verification helpers."""

import asyncio
import logging
import re
from typing import Any, Optional

import docker

from ..exceptions import DockerError

logger = logging.getLogger(__name__)


async def stop_container(container: Any, timeout: int | None = None) -> None:
    """Stop a container gracefully with SIGTERM, then SIGKILL if needed."""
    try:
        loop = asyncio.get_event_loop()
        eff_timeout = int(timeout) if timeout is not None else 1
        await loop.run_in_executor(
            None, lambda: container.stop(timeout=max(0, eff_timeout))
        )
        logger.info(f"Stopped container {container.name} gracefully")
    except docker.errors.NotFound:
        logger.debug(f"Container {container.name} not found")
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        logger.error(f"Failed to stop container {container.name}: {e}")


async def verify_container_tools(container: Any, tools: list[str]) -> None:
    """Verify required tools are available in the container."""
    loop = asyncio.get_event_loop()
    missing: list[str] = []
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
    manager: Any,
    container: Any,
    force: bool = True,
    *,
    remove_home_volume: bool = False,
) -> None:
    """Remove a container after stopping it."""

    try:
        loop = asyncio.get_event_loop()
        home_volume_name: Optional[str] = None
        try:
            await loop.run_in_executor(None, container.reload)
            mounts = (container.attrs or {}).get("Mounts", []) or []
            for m in mounts:
                try:
                    mtype = str(m.get("Type") or m.get("type") or "").lower()
                    if mtype != "volume":
                        continue
                    dest = (
                        m.get("Destination") or m.get("Target") or m.get("Mountpoint")
                    )
                    if str(dest) == "/home/node":
                        name = m.get("Name") or m.get("name") or m.get("Source")
                        if name:
                            home_volume_name = str(name)
                            break
                except Exception:
                    continue
        except Exception:
            pass

        await loop.run_in_executor(None, container.reload)
        if container.status == "running":
            await stop_container(container)

        await loop.run_in_executor(None, lambda: container.remove(force=force))
        logger.info(f"Removed container {container.name}")

        if remove_home_volume and home_volume_name:
            try:
                name = home_volume_name
                if name.startswith("pitaya_home_") and not re.fullmatch(
                    r"pitaya_home_g[0-9a-f]{8}", name
                ):
                    vol = await loop.run_in_executor(
                        None, lambda: manager.client.volumes.get(name)
                    )
                    await loop.run_in_executor(None, lambda: vol.remove(force=True))
                    logger.info(f"Removed session volume {name}")
                else:
                    logger.debug(
                        f"Skipping volume removal for {name} (not run-scoped or not Pitaya)"
                    )
            except Exception as ve:
                logger.warning(
                    f"Failed to remove session volume {home_volume_name}: {ve}"
                )
    except docker.errors.NotFound:
        logger.debug(f"Container {container.name} already removed")
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        logger.error(f"Failed to remove container {container.name}: {e}")


__all__ = ["cleanup_container", "verify_container_tools", "stop_container"]
