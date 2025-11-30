"""Build hardened container configuration."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from docker.types import Ulimit

from ..types import AuthConfig


async def build_container_config(
    *,
    container_name: str,
    image: str,
    mounts,
    volumes,
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
    strategy_index: str,
    task_key: Optional[str],
    session_group_key: Optional[str],
    instance_id: Optional[str],
    khash: str,
    ghash: str,
    memory_gb: int,
    memory_swap_gb: int,
    cpu_count: int,
    network_egress: str,
    extra_env: Optional[Dict[str, str]],
    auth_config: Optional[AuthConfig],
    session_id: Optional[str],
    plugin,
    event_callback,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "image": image,
        "name": container_name,
        "command": "sleep infinity",
        "detach": True,
        "labels": {
            "pitaya": "true",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id or "",
            "strategy_execution_id": strategy_execution_id or "",
            "strategy_index": strategy_index,
            "task_key": task_key or "",
            "session_group_key": session_group_key or "",
            "pitaya.last_active_ts": datetime.now(timezone.utc).isoformat(),
            "instance_id": instance_id or "",
        },
        "mounts": mounts,
        "tmpfs": {"/tmp": "rw,size=512m"},
        "working_dir": "/workspace",
        "read_only": True,
        "user": "node",
        "environment": {
            "PYTHONUNBUFFERED": "1",
            "GIT_AUTHOR_NAME": "AI Agent",
            "GIT_AUTHOR_EMAIL": "agent@pitaya.local",
            "GIT_COMMITTER_NAME": "AI Agent",
            "GIT_COMMITTER_EMAIL": "agent@pitaya.local",
            "TASK_KEY": task_key or "",
            "SESSION_GROUP_KEY": session_group_key or "",
            "KHASH": khash,
            "GHASH": ghash,
        },
        "mem_limit": f"{memory_gb}g",
        "memswap_limit": f"{memory_swap_gb}g",
        "auto_remove": False,
    }
    if volumes:
        config["volumes"] = volumes

    _apply_security_defaults(config)
    _apply_network_egress(config, network_egress)
    _merge_extra_environment(config, extra_env)
    _add_normative_environment(
        config=config,
        session_group_key=session_group_key,
        task_key=task_key,
        instance_id=instance_id,
        container_name=container_name,
    )
    _apply_proxy_environment(config, network_egress)
    _apply_cpu_limit(config, cpu_count)
    _apply_auth_config(config, auth_config)
    _maybe_set_session_id(config, session_id)
    config = await _run_plugin_prepare(config, plugin, session_id)
    _enforce_mount_policy(config)
    _summarize_environment(config, event_callback)
    return config


def _apply_security_defaults(config: Dict[str, Any]) -> None:
    try:
        config["cap_drop"] = ["ALL"]
    except Exception:
        pass
    try:
        secopts = set(config.get("security_opt") or [])
        secopts.add("no-new-privileges:true")
        config["security_opt"] = list(secopts)
    except Exception:
        pass
    try:
        config["pids_limit"] = 512
    except Exception:
        config["pids_limit"] = 512
    try:
        config["ulimits"] = (config.get("ulimits") or []) + [
            Ulimit(name="nofile", soft=4096, hard=4096)
        ]
    except Exception:
        pass


def _apply_network_egress(config: Dict[str, Any], network_egress: str) -> None:
    try:
        if str(network_egress).lower() == "offline":
            if "network_disabled" in config:
                try:
                    del config["network_disabled"]
                except Exception:
                    pass
            config["network_mode"] = "none"
    except Exception:
        pass


def _merge_extra_environment(
    config: Dict[str, Any], extra_env: Optional[Dict[str, str]]
) -> None:
    if extra_env:
        try:
            config.setdefault("environment", {}).update(extra_env)
        except Exception:
            pass


def _add_normative_environment(
    *,
    config: Dict[str, Any],
    session_group_key: Optional[str],
    task_key: Optional[str],
    instance_id: Optional[str],
    container_name: str,
) -> None:
    try:
        eff_sgk = session_group_key or (task_key or instance_id or container_name)
        durable = task_key or (instance_id or "")
        khash_val = hashlib.sha256(
            str(durable).encode("utf-8", errors="ignore")
        ).hexdigest()[:8]
        payload = json.dumps(
            {"session_group_key": eff_sgk}, separators=(",", ":"), sort_keys=True
        )
        ghash_val = hashlib.sha256(
            payload.encode("utf-8", errors="ignore")
        ).hexdigest()[:8]
        config.setdefault("environment", {}).update(
            {
                "TASK_KEY": str(durable),
                "SESSION_GROUP_KEY": str(eff_sgk),
                "KHASH": khash_val,
                "GHASH": ghash_val,
            }
        )
    except Exception:
        pass


def _apply_proxy_environment(config: Dict[str, Any], network_egress: str) -> None:
    try:
        if str(network_egress).lower() == "proxy":
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "no_proxy",
            ):
                if key in os.environ:
                    config.setdefault("environment", {})[key] = os.environ[key]
    except Exception:
        pass


def _apply_cpu_limit(config: Dict[str, Any], cpu_count: int) -> None:
    try:
        config["nano_cpus"] = int(max(1, int(cpu_count)) * 1_000_000_000)
    except Exception:
        pass


def _apply_auth_config(
    config: Dict[str, Any], auth_config: Optional[AuthConfig]
) -> None:
    env = config.setdefault("environment", {})
    if auth_config:
        if auth_config.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = auth_config.oauth_token
        elif auth_config.api_key:
            env["ANTHROPIC_API_KEY"] = auth_config.api_key
            if auth_config.base_url:
                env["ANTHROPIC_BASE_URL"] = auth_config.base_url
    else:
        if oauth_token := os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        elif api_key := os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = api_key
            if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
                env["ANTHROPIC_BASE_URL"] = base_url


def _maybe_set_session_id(config: Dict[str, Any], session_id: Optional[str]) -> None:
    if session_id:
        config.setdefault("environment", {})["CLAUDE_CODE_SESSION_ID"] = session_id


async def _run_plugin_prepare(
    config: Dict[str, Any], plugin, session_id: Optional[str]
) -> Dict[str, Any]:
    try:
        if plugin is not None and hasattr(plugin, "prepare_container"):
            return await plugin.prepare_container(config, session_id=session_id)
    except Exception:
        pass
    return config


def _enforce_mount_policy(config: Dict[str, Any]) -> None:
    try:
        if isinstance(config.get("tmpfs"), dict):
            tmap = config.get("tmpfs") or {}
            if set(tmap.keys()) != {"/tmp"}:
                tmp_only: Dict[str, str] = {}
                if "/tmp" in tmap:
                    tmp_only["/tmp"] = tmap["/tmp"]
                config["tmpfs"] = tmp_only
        for key in ("devices", "device_requests"):
            if key in config:
                try:
                    del config[key]
                except Exception:
                    pass
        if bool(config.get("privileged")):
            config["privileged"] = False
        if config.get("cap_add"):
            config["cap_add"] = []
        try:
            secopts = set(config.get("security_opt") or [])
            secopts.add("no-new-privileges:true")
            config["security_opt"] = list(secopts)
        except Exception:
            pass
    except Exception:
        pass


def _summarize_environment(config: Dict[str, Any], event_callback) -> None:
    try:
        env = config.get("environment", {}) or {}
        event_callback and event_callback(  # type: ignore[func-returns-value]
            {
                "type": "instance.container_config_ready",
                "data": {
                    "env_keys": len(env),
                    "has_oauth": bool(env.get("CLAUDE_CODE_OAUTH_TOKEN")),
                    "has_api_key": bool(env.get("ANTHROPIC_API_KEY")),
                    "base_url_set": bool(env.get("ANTHROPIC_BASE_URL")),
                },
            }
        )
    except Exception:
        pass
