"""Container creation phase for instance attempts."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from .docker_manager import DockerManager
from .plugin_interface import RunnerPlugin

logger = logging.getLogger(__name__)


async def prepare_container_phase(
    docker_manager: DockerManager,
    plugin: RunnerPlugin,
    container_name: str,
    workspace_dir,
    container_limits,
    run_ids: Dict[str, Optional[str]],
    instance_id: str,
    session_id: Optional[str],
    auth_config,
    reuse_container: bool,
    session_group_key: Optional[str],
    import_policy: str,
    network_egress: Optional[str],
    event_callback,
    task_key: Optional[str],
    plugin_name: Optional[str],
    resolved_model_id: Optional[str],
    allow_global_session_volume: bool,
) -> Any:
    if event_callback:
        event_callback(
            "instance.container_env_preparing", {"container_name": container_name}
        )

    env_vars = await plugin.prepare_environment(
        None, asdict(auth_config) if auth_config else None
    )
    if event_callback:
        try:
            event_callback(
                "instance.container_env_prepared",
                {"container_name": container_name, "env_keys": len(env_vars or {})},
            )
        except Exception:
            pass

    container = await docker_manager.create_container(
        container_name=container_name,
        workspace_dir=workspace_dir,
        cpu_count=container_limits.cpu_count,
        memory_gb=container_limits.memory_gb,
        memory_swap_gb=container_limits.memory_swap_gb,
        run_id=run_ids.get("run_id"),
        strategy_execution_id=run_ids.get("strategy_execution_id"),
        instance_id=instance_id,
        image=run_ids.get("docker_image"),
        session_id=session_id,
        auth_config=auth_config,
        reuse_container=reuse_container,
        extra_env=env_vars,
        plugin=plugin,
        session_group_key=session_group_key,
        import_policy=import_policy,
        network_egress=network_egress or "online",
        event_callback=event_callback,
        task_key=task_key,
        plugin_name=plugin_name,
        resolved_model_id=resolved_model_id,
        allow_global_session_volume=allow_global_session_volume,
    )

    if event_callback:
        event_callback(
            "instance.container_created",
            {
                "container_name": getattr(container, "name", container_name),
                "container_id": getattr(container, "id", "")[:12],
            },
        )
    return container


__all__ = ["prepare_container_phase"]
