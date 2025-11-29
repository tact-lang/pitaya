"""Workspace and volume mount helpers."""

from __future__ import annotations

import json
import logging
import os
import platform

from docker.types import Mount

from .docker_context import CreateContext
from ..utils.platform_utils import normalize_path_for_docker

logger = logging.getLogger(__name__)


def prepare_mounts(ctx: CreateContext) -> None:
    ctx.workspace_dir = ctx.workspace_dir.resolve()
    name_parts = ctx.container_name.split("_")
    ctx.sidx = name_parts[-2].lstrip("s") if len(name_parts) >= 4 else "0"

    import hashlib

    eff_sgk = ctx.session_group_key or (
        ctx.task_key or ctx.instance_id or ctx.container_name or ""
    )
    if ctx.allow_global_session_volume:
        payload = {
            "session_group_key": eff_sgk,
            "plugin": (ctx.plugin_name or ""),
            "model": (ctx.resolved_model_id or ""),
        }
        enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ctx.ghash = hashlib.sha256(enc.encode("utf-8", errors="ignore")).hexdigest()[:8]
        ctx.volume_name = f"pitaya_home_g{ctx.ghash}"
    else:
        payload = {"session_group_key": eff_sgk}
        enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ctx.ghash = hashlib.sha256(enc.encode("utf-8", errors="ignore")).hexdigest()[:8]
        ctx.volume_name = (
            f"pitaya_home_{(ctx.run_id or 'norun')}_s{ctx.sidx}_g{ctx.ghash}"
        )

    ctx.khash = (
        hashlib.sha256(
            (ctx.task_key or "").encode("utf-8", errors="ignore")
        ).hexdigest()[:8]
        if ctx.task_key
        else ""
    )

    selinux_mode = ""
    if platform.system() == "Linux" and os.path.exists("/sys/fs/selinux"):
        selinux_mode = "z"

    review_mode = "ro"
    ctx.mounts = []
    ctx.volumes = {}
    ctx.ws_source = normalize_path_for_docker(ctx.workspace_dir)
    ctx.ws_ro = ctx.import_policy == "never" and review_mode == "ro"
    if selinux_mode:
        mode = "z,ro" if ctx.ws_ro else "z"
        ctx.volumes[ctx.ws_source] = {"bind": "/workspace", "mode": mode}
    else:
        ctx.mounts.append(
            Mount(
                target="/workspace",
                source=ctx.ws_source,
                type="bind",
                read_only=ctx.ws_ro,
            )
        )
    ctx.mounts.append(Mount(target="/home/node", source=ctx.volume_name, type="volume"))
    if ctx.event_callback:
        try:
            ctx.event_callback(
                {
                    "type": "instance.container_mounts_prepared",
                    "data": {
                        "workspace_source": ctx.ws_source,
                        "workspace_read_only": bool(ctx.ws_ro),
                        "home_volume": ctx.volume_name,
                    },
                }
            )
        except Exception:
            pass


__all__ = ["prepare_mounts"]
