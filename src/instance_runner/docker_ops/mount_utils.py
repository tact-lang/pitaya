"""Utilities for mount preparation and naming."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docker.types import Mount

from ...utils.platform_utils import normalize_path_for_docker


def extract_strategy_index(container_name: str) -> str:
    parts = container_name.split("_")
    if len(parts) >= 4:
        return parts[-2].lstrip("s")
    return "0"


def build_volume_metadata(
    *,
    container_name: str,
    run_id: Optional[str],
    session_group_key: Optional[str],
    task_key: Optional[str],
    instance_id: Optional[str],
    plugin_name: Optional[str],
    resolved_model_id: Optional[str],
    allow_global_session_volume: bool,
    strategy_index: str,
) -> Tuple[str, str, str]:
    eff_sgk = session_group_key or (task_key or instance_id or container_name or "")
    if bool(allow_global_session_volume):
        payload = {
            "session_group_key": eff_sgk,
            "plugin": plugin_name or "",
            "model": resolved_model_id or "",
        }
        enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ghash = hashlib.sha256(enc.encode("utf-8", errors="ignore")).hexdigest()[:8]
        volume_name = f"pitaya_home_g{ghash}"
    else:
        payload = {"session_group_key": eff_sgk}
        enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ghash = hashlib.sha256(enc.encode("utf-8", errors="ignore")).hexdigest()[:8]
        volume_name = f"pitaya_home_{(run_id or 'norun')}_s{strategy_index}_g{ghash}"

    khash = (
        hashlib.sha256((task_key or "").encode("utf-8", errors="ignore")).hexdigest()[
            :8
        ]
        if task_key
        else ""
    )
    return volume_name, ghash, khash


def prepare_mounts(
    *,
    workspace_dir: Path,
    import_policy: str,
    volume_name: str,
) -> Tuple[List[Mount], Dict[str, Dict[str, str]]]:
    selinux_mode = ""
    if platform.system() == "Linux" and os.path.exists("/sys/fs/selinux"):
        selinux_mode = "z"

    review_mode = "ro"
    mounts: List[Mount] = []
    volumes: Dict[str, Dict[str, str]] = {}
    ws_source = normalize_path_for_docker(workspace_dir)
    ws_ro = import_policy == "never" and review_mode == "ro"
    if selinux_mode:
        mode = "z,ro" if ws_ro else "z"
        volumes[ws_source] = {"bind": "/workspace", "mode": mode}
    else:
        mounts.append(
            Mount(target="/workspace", source=ws_source, type="bind", read_only=ws_ro)
        )
    mounts.append(Mount(target="/home/node", source=volume_name, type="volume"))
    return mounts, volumes


def log_mounts_prepared(
    mounts: List[Mount],
    volume_name: str,
    event_callback,
) -> None:
    try:
        event_callback and event_callback(  # type: ignore[func-returns-value]
            {
                "type": "instance.container_mounts_prepared",
                "data": {
                    "workspace_source": mounts[0].source,
                    "workspace_read_only": bool(mounts[0].read_only),
                    "home_volume": volume_name,
                },
            }
        )
    except Exception:
        pass


def workspace_is_ro(container) -> bool:
    try:
        mounts = container.attrs.get("Mounts", [])
        for mount in mounts:
            if str(mount.get("Destination")) == "/workspace":
                return not bool(mount.get("RW", True))
    except Exception:
        pass
    return False
