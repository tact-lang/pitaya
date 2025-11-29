"""Docker container management for instance execution."""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import docker

from .docker_cleanup import (
    cleanup_container,
    stop_container as stop_container_helper,
    verify_container_tools,
)
from .docker_context import CreateContext
from .docker_create import create_container
from .docker_execute import execute_command
from .docker_heartbeat import (
    start_heartbeat as start_hb_helper,
    stop_heartbeat as stop_hb_helper,
)
from ..exceptions import DockerError
from ..shared import AuthConfig

if TYPE_CHECKING:
    from docker.errors import DockerException
    from docker.models.containers import Container
else:
    try:
        from docker.errors import DockerException
        from docker.models.containers import Container
    except ImportError:  # pragma: no cover - defensive fallback
        DockerException = Exception
        Container = Any

# Suppress the urllib3 exception on close that happens with docker-py
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker containers for instance execution."""

    _hb_tasks: Dict[str, asyncio.Task] = {}

    def __init__(self, api_timeout: int = 20):
        try:
            self.client = docker.from_env(timeout=int(api_timeout))
        except DockerException as e:
            raise DockerError(f"Failed to connect to Docker daemon: {e}")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    async def start_heartbeat(
        self, container: Container, interval_s: float = 15.0
    ) -> None:
        await start_hb_helper(self, container, interval_s)

    async def stop_heartbeat(self, container: Container) -> None:
        await stop_hb_helper(self, container)

    async def initialize(self) -> None:
        """Initialize Docker manager with startup tasks (no-op)."""

    async def validate_environment(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.ping)
            return True
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Docker validation failed: {e}")
            return False

    async def create_container(
        self,
        container_name: str,
        workspace_dir,
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
        ctx = CreateContext(
            manager=self,
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
        return await create_container(ctx)

    async def execute_command(
        self,
        container: Container,
        command: list[str],
        plugin: Any,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_seconds: int = 3600,
        max_turns: Optional[int] = None,
        stream_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await execute_command(
            container,
            command,
            plugin,
            event_callback=event_callback,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
            stream_log_path=stream_log_path,
        )

    async def stop_container(
        self, container: Container, timeout: int | None = None
    ) -> None:
        await stop_container_helper(container, timeout)

    async def verify_container_tools(
        self, container: Container, tools: list[str]
    ) -> None:
        await verify_container_tools(container, tools)

    async def cleanup_container(
        self,
        container: Container,
        force: bool = True,
        *,
        remove_home_volume: bool = False,
    ) -> None:
        await cleanup_container(
            self, container, force=force, remove_home_volume=remove_home_volume
        )

    def get_container(self, name: str) -> Optional[Container]:
        try:
            return self.client.containers.get(name)
        except Exception:
            return None


__all__ = ["DockerManager"]
