"""Shared container creation context for docker manager helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from docker.types import Mount

from ..shared import AuthConfig


EventCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class CreateContext:
    """Mutable state passed across container creation steps."""

    manager: Any
    container_name: str
    workspace_dir: Path
    cpu_count: int
    memory_gb: int
    memory_swap_gb: int
    run_id: Optional[str]
    strategy_execution_id: Optional[str]
    instance_id: Optional[str]
    image: str
    session_id: Optional[str]
    auth_config: Optional[AuthConfig]
    reuse_container: bool
    extra_env: Optional[Dict[str, str]]
    plugin: Optional[Any]
    session_group_key: Optional[str]
    import_policy: str
    network_egress: str
    event_callback: EventCallback
    task_key: Optional[str]
    plugin_name: Optional[str]
    resolved_model_id: Optional[str]
    allow_global_session_volume: bool
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_event_loop)

    phase_start: float = 0.0

    container: Any = None
    mounts: list[Mount] = field(default_factory=list)
    volumes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    ws_source: str = ""
    ws_ro: bool = False
    sidx: str = "0"
    volume_name: str = ""
    khash: str = ""
    ghash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


__all__ = ["CreateContext", "EventCallback"]
