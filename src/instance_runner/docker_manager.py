"""
Docker container management for instance execution.

Provides lifecycle, resource limits, volume mounts, and cleanup through a small
facade that delegates to helper modules for clarity and compliance with size
limits.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from docker.models.containers import Container
else:  # pragma: no cover - runtime fallback
    try:
        from docker.models.containers import Container
    except ImportError:  # pragma: no cover - very unlikely
        Container = Any

from . import DockerError, TimeoutError
from .types import AuthConfig
from .docker_ops import cleanup, container_create, exec_runner, heartbeat

# Suppress the urllib3 exception on close that happens with docker-py
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Workaround: suppress benign "I/O operation on closed file" from urllib3 during close
try:  # pragma: no cover - environment dependent
    import urllib3.response as _u3r  # type: ignore

    _original_close = _u3r.HTTPResponse.close  # type: ignore[attr-defined]

    def _patched_close(self):  # type: ignore[no-redef]
        try:
            _original_close(self)
        except (ValueError, OSError):
            pass

    _u3r.HTTPResponse.close = _patched_close  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best effort
    pass

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker containers for instance execution."""

    _hb_tasks: Dict[str, asyncio.Task] = {}

    def __init__(self, api_timeout: int = 20):
        """Initialize Docker client.

        Args:
            api_timeout: Per-request timeout (seconds) for Docker SDK calls.
        """

        try:
            self.client = docker.from_env(timeout=int(api_timeout))
            # Expose commonly used error types for helper modules
            self.NotFound = NotFound
            self.ImageNotFound = ImageNotFound
            self.DockerException = DockerException
            # Provide instance logger for helpers
            self.logger = logger
        except DockerException as exc:  # type: ignore[misc]
            raise DockerError(f"Failed to connect to Docker daemon: {exc}")

    def close(self) -> None:
        """Close the Docker client connection."""
        try:
            self.client.close()
        except Exception:
            pass

    async def start_heartbeat(
        self, container: Container, interval_s: float = 15.0
    ) -> None:
        """Start a periodic heartbeat writer inside the container."""

        await heartbeat.start_heartbeat(self._hb_tasks, container, interval_s)

    async def stop_heartbeat(self, container: Container) -> None:
        await heartbeat.stop_heartbeat(self._hb_tasks, container)

    async def _async_iter(self, blocking_iter):
        """Convert a blocking iterator to an async iterator."""

        loop = asyncio.get_event_loop()
        sentinel = object()
        while True:
            item = await loop.run_in_executor(
                None, lambda: next(blocking_iter, sentinel)
            )
            if item is sentinel:
                break
            yield item

    async def initialize(self) -> None:
        """Initialize Docker manager with startup tasks (currently no-op)."""

    async def validate_environment(self) -> bool:
        """Validate Docker daemon is accessible and working."""

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.ping)
            return True
        except (docker.errors.APIError, docker.errors.DockerException) as exc:  # type: ignore[attr-defined]
            logger.error(f"Docker validation failed: {exc}")
            return False

    async def create_container(
        self,
        container_name: str,
        workspace_dir: Path,
        cpu_count: int = 2,
        memory_gb: int = 4,
        memory_swap_gb: int = 4,
        run_id: Optional[str] = None,
        strategy_execution_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        image: str = "pitaya-agents:latest",
        session_id: Optional[str] = None,
        auth_config: Optional[AuthConfig] = None,
        reuse_container: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
        plugin: Optional[Any] = None,
        session_group_key: Optional[str] = None,
        import_policy: str = "auto",
        network_egress: str = "online",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        task_key: Optional[str] = None,
        *,
        plugin_name: Optional[str] = None,
        resolved_model_id: Optional[str] = None,
        allow_global_session_volume: bool = False,
    ) -> Container:
        """Create or reuse a Docker container for instance execution."""

        try:
            return await container_create.create_container(
                self,
                container_name=container_name,
                workspace_dir=workspace_dir,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                memory_swap_gb=memory_swap_gb,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                instance_id=instance_id,
                image=image,
                session_id=session_id,
                auth_config=auth_config,
                reuse_container=reuse_container,
                extra_env=extra_env,
                plugin=plugin,
                session_group_key=session_group_key,
                import_policy=import_policy,
                network_egress=network_egress,
                event_callback=event_callback,
                task_key=task_key,
                plugin_name=plugin_name,
                resolved_model_id=resolved_model_id,
                allow_global_session_volume=allow_global_session_volume,
            )
        except DockerException as exc:  # type: ignore[misc]
            raise DockerError(f"Failed to create container: {exc}")

    async def execute_command(
        self,
        container: Container,
        command: List[str],
        plugin: Any,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_seconds: int = 3600,
        max_turns: Optional[int] = None,
        stream_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute AI tool command in container and parse output."""

        try:
            return await exec_runner.execute_command(
                self,
                container=container,
                command=command,
                plugin=plugin,
                event_callback=event_callback,
                timeout_seconds=timeout_seconds,
                max_turns=max_turns,
                stream_log_path=stream_log_path,
            )
        except TimeoutError:
            raise
        except (docker.errors.APIError, docker.errors.DockerException, OSError) as exc:  # type: ignore[attr-defined]
            raise DockerError(f"Failed to execute agent tool: {exc}")

    async def stop_container(
        self, container: Container, timeout: int | None = None
    ) -> None:
        """Stop a container gracefully with SIGTERM, then SIGKILL if needed."""
        await cleanup.stop_container(self, container, timeout)

    async def verify_container_tools(
        self, container: Container, tools: list[str]
    ) -> None:
        """Verify required tools are available in the container."""
        await cleanup.verify_container_tools(self, container, tools)

    async def cleanup_container(
        self,
        container: Container,
        force: bool = True,
        *,
        remove_home_volume: bool = False,
    ) -> None:
        """Remove a container after stopping it."""
        await cleanup.cleanup_container(
            self,
            container,
            force=force,
            remove_home_volume=remove_home_volume,
        )

    def get_container(self, name: str) -> Optional[Container]:
        """Return container by name or None if missing."""

        try:
            return self.client.containers.get(name)
        except Exception:
            return None

    # Internal helpers -------------------------------------------------

    @staticmethod
    def _emit_event(
        callback: Optional[Callable[[Dict[str, Any]], None]],
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        if callback:
            callback({"type": event_type, "data": data})

    @staticmethod
    def is_docker_error(exc: Exception) -> bool:
        return isinstance(exc, (docker.errors.APIError, docker.errors.DockerException))  # type: ignore[attr-defined]


__all__ = ["DockerManager"]
