"""Orchestrates container creation by composing helper modules."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..types import AuthConfig
from .config_builder import build_container_config
from .image_check import ensure_image_exists
from .mount_utils import (
    build_volume_metadata,
    extract_strategy_index,
    log_mounts_prepared,
    prepare_mounts,
    workspace_is_ro,
)
from .reuse import (
    create_container_instance,
    reuse_existing_container,
    start_container_and_wait,
)


async def create_container(
    manager,
    *,
    container_name: str,
    workspace_dir: Path,
    cpu_count: int,
    memory_gb: int,
    memory_swap_gb: int,
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
    instance_id: Optional[str],
    image: str,
    session_id: Optional[str],
    auth_config: Optional[AuthConfig],
    reuse_container: bool,
    extra_env: Optional[Dict[str, str]],
    plugin: Optional[Any],
    session_group_key: Optional[str],
    import_policy: str,
    network_egress: str,
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
    task_key: Optional[str],
    plugin_name: Optional[str],
    resolved_model_id: Optional[str],
    allow_global_session_volume: bool,
) -> Any:
    phase_start = time.monotonic()
    manager._emit_event(  # noqa: SLF001
        event_callback,
        "instance.container_create_entry",
        {
            "container_name": container_name,
            "workspace_dir": str(workspace_dir),
            "image": image,
            "reuse": bool(reuse_container),
        },
    )

    loop = asyncio.get_event_loop()
    await ensure_image_exists(manager, loop, image, event_callback)

    existing = await reuse_existing_container(
        manager,
        loop=loop,
        container_name=container_name,
        reuse_container=reuse_container,
        import_policy=import_policy,
        event_callback=event_callback,
        workspace_ro_checker=workspace_is_ro,
    )
    if existing:
        return existing

    workspace_dir = workspace_dir.resolve()
    strategy_index = extract_strategy_index(container_name)
    volume_name, ghash, khash = build_volume_metadata(
        container_name=container_name,
        run_id=run_id,
        session_group_key=session_group_key,
        task_key=task_key,
        instance_id=instance_id,
        plugin_name=plugin_name,
        resolved_model_id=resolved_model_id,
        allow_global_session_volume=allow_global_session_volume,
        strategy_index=strategy_index,
    )

    mounts, volumes = prepare_mounts(
        workspace_dir=workspace_dir,
        import_policy=import_policy,
        volume_name=volume_name,
    )
    log_mounts_prepared(mounts, volume_name, event_callback)

    config = await build_container_config(
        container_name=container_name,
        image=image,
        mounts=mounts,
        volumes=volumes,
        run_id=run_id,
        strategy_execution_id=strategy_execution_id,
        strategy_index=strategy_index,
        task_key=task_key,
        session_group_key=session_group_key,
        instance_id=instance_id,
        khash=khash,
        ghash=ghash,
        memory_gb=memory_gb,
        memory_swap_gb=memory_swap_gb,
        cpu_count=cpu_count,
        network_egress=network_egress,
        extra_env=extra_env,
        auth_config=auth_config,
        session_id=session_id,
        plugin=plugin,
        event_callback=event_callback,
    )

    container = await create_container_instance(
        manager,
        loop=loop,
        config=config,
        container_name=container_name,
        image=image,
        event_callback=event_callback,
    )

    await start_container_and_wait(
        manager,
        loop=loop,
        container=container,
        container_name=container_name,
        event_callback=event_callback,
    )

    manager._emit_event(  # noqa: SLF001
        event_callback,
        "instance.container_created",
        {
            "container_name": container_name,
            "container_id": container.id[:12],
        },
    )
    manager.logger.info(  # noqa: SLF001
        f"Created container {container_name} (ID: {container.id[:12]}) in %.2fs",
        time.monotonic() - phase_start,
    )
    return container
