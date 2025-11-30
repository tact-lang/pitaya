"""Container lifecycle management for instance attempts."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

from pitaya.exceptions import DockerError
from pitaya.runner.docker.manager import DockerManager
from pitaya.shared.plugin import RunnerPlugin
from pitaya.runner.runner_params import RunnerParams

logger = logging.getLogger(__name__)


class ContainerManager:
    """Create, verify, and clean up containers."""

    def __init__(
        self,
        *,
        docker_manager: DockerManager,
        plugin: RunnerPlugin,
        params: RunnerParams,
        emit_event: Callable[[str, Dict[str, Any]], None],
        session_id: Optional[str],
        reuse_container: bool,
    ) -> None:
        self.docker_manager = docker_manager
        self.plugin = plugin
        self.params = params
        self.emit_event = emit_event
        self.session_id = session_id
        self.reuse_container = reuse_container
        self.container = None

    async def prepare(self, workspace_dir) -> None:
        self.env_vars = await self._prepare_env_vars()
        self.emit_event(
            "instance.container_creating",
            {
                "container_name": self.params.container_name,
                "workspace_dir": str(workspace_dir),
                "image": self.params.docker_image,
            },
        )
        cc_start = time.time()
        try:
            self.container = await asyncio.wait_for(
                self.docker_manager.create_container(
                    container_name=self.params.container_name,
                    workspace_dir=workspace_dir,
                    cpu_count=self.params.container_limits.cpu_count,
                    memory_gb=self.params.container_limits.memory_gb,
                    memory_swap_gb=self.params.container_limits.memory_swap_gb,
                    run_id=self.params.run_id,
                    strategy_execution_id=self.params.strategy_execution_id,
                    instance_id=self.params.instance_id,
                    session_id=self.session_id,
                    session_group_key=self.params.session_group_key,
                    import_policy=self.params.import_policy,
                    image=self.params.docker_image,
                    auth_config=self.params.auth_config,
                    reuse_container=self.reuse_container,
                    extra_env=self.env_vars,
                    plugin=self.plugin,
                    network_egress=(self.params.network_egress or "online"),
                    event_callback=self.params.event_callback,
                    task_key=self.params.task_key,
                    plugin_name=getattr(self.plugin, "name", "claude-code"),
                    resolved_model_id=self.params.model,
                    allow_global_session_volume=self.params.allow_global_session_volume,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError as exc:
            self.emit_event(
                "instance.container_create_timeout",
                {"container_name": self.params.container_name, "timeout_s": 60},
            )
            raise DockerError(
                "Container creation phase exceeded 60s. Docker may be unresponsive; try restarting Docker Desktop."
            ) from exc
        finally:
            logger.debug(
                "create_container finished in %.2fs (including error path)",
                time.time() - cc_start,
            )

        self.emit_event("instance.phase_completed", {"phase": "container_creation"})
        await self._verify_tools()
        await self._start_heartbeat()

    async def _prepare_env_vars(self) -> Dict[str, str]:
        self.emit_event(
            "instance.container_env_preparing",
            {"container_name": self.params.container_name},
        )
        try:
            auth_dict = (
                asdict(self.params.auth_config) if self.params.auth_config else None
            )
            env_vars = await self.plugin.prepare_environment(None, auth_dict)
            self.emit_event(
                "instance.container_env_prepared",
                {
                    "env_vars_count": len(env_vars or {}),
                    "container_name": self.params.container_name,
                },
            )
            return env_vars or {}
        except Exception:
            self.emit_event(
                "instance.container_env_prepared",
                {"env_vars_count": 0, "container_name": self.params.container_name},
            )
            return {}

    async def _verify_tools(self) -> None:
        await self.docker_manager.verify_container_tools(self.container, ["git"])

    async def _start_heartbeat(self) -> None:
        try:
            await self.docker_manager.start_heartbeat(self.container)
        except Exception:
            pass

    async def stop_and_cleanup(self, *, remove_home_volume: bool) -> None:
        if not self.container:
            return
        try:
            try:
                await self.docker_manager.stop_heartbeat(self.container)
            except Exception:
                pass
            await self.docker_manager.cleanup_container(
                self.container, remove_home_volume=remove_home_volume
            )
        except Exception:
            pass
