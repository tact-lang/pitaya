"""Container config construction helpers."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from docker.types import Ulimit

from .docker_context import CreateContext
from ..shared import AuthConfig

logger = logging.getLogger(__name__)


def build_base_config(ctx: CreateContext) -> None:
    ctx.config = {
        "image": ctx.image,
        "name": ctx.container_name,
        "command": "sleep infinity",
        "detach": True,
        "labels": {
            "pitaya": "true",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": ctx.run_id or "",
            "strategy_execution_id": ctx.strategy_execution_id or "",
            "strategy_index": ctx.sidx,
            "task_key": ctx.task_key or "",
            "session_group_key": ctx.session_group_key or "",
            "pitaya.last_active_ts": datetime.now(timezone.utc).isoformat(),
            "instance_id": ctx.instance_id or "",
        },
        "mounts": ctx.mounts,
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
            "TASK_KEY": ctx.task_key or "",
            "SESSION_GROUP_KEY": ctx.session_group_key or "",
            "KHASH": ctx.khash,
            "GHASH": ctx.ghash,
        },
        "mem_limit": f"{ctx.memory_gb}g",
        "memswap_limit": f"{ctx.memory_swap_gb}g",
        "auto_remove": False,
    }
    if ctx.volumes:
        ctx.config["volumes"] = ctx.volumes

    try:
        ctx.config["cap_drop"] = ["ALL"]
    except Exception:
        pass
    try:
        secopts = set(ctx.config.get("security_opt") or [])
        secopts.add("no-new-privileges:true")
        ctx.config["security_opt"] = list(secopts)
    except Exception:
        pass
    try:
        ctx.config["pids_limit"] = 512
    except Exception:
        ctx.config["pids_limit"] = 512
    try:
        ctx.config["ulimits"] = (ctx.config.get("ulimits") or []) + [
            Ulimit(name="nofile", soft=4096, hard=4096)
        ]
    except Exception:
        pass

    try:
        if str(ctx.network_egress).lower() == "offline":
            if "network_disabled" in ctx.config:
                del ctx.config["network_disabled"]
            ctx.config["network_mode"] = "none"
    except Exception:
        pass

    if ctx.extra_env:
        try:
            ctx.config["environment"].update(ctx.extra_env)
        except Exception:
            pass

    try:
        import hashlib

        eff_sgk = ctx.session_group_key or (
            ctx.task_key or ctx.instance_id or ctx.container_name
        )
        durable = ctx.task_key or (ctx.instance_id or "")
        khash = hashlib.sha256(
            str(durable).encode("utf-8", errors="ignore")
        ).hexdigest()[:8]
        payload = json.dumps(
            {"session_group_key": eff_sgk}, separators=(",", ":"), sort_keys=True
        )
        ghash = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:8]
        ctx.config["environment"].update(
            {
                "TASK_KEY": str(durable),
                "SESSION_GROUP_KEY": str(eff_sgk),
                "KHASH": khash,
                "GHASH": ghash,
            }
        )
    except Exception:
        pass

    try:
        if str(ctx.network_egress).lower() == "proxy":
            for k in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "no_proxy",
            ):
                if k in os.environ:
                    ctx.config["environment"][k] = os.environ[k]
    except Exception:
        pass

    try:
        ctx.config["nano_cpus"] = int(max(1, int(ctx.cpu_count)) * 1_000_000_000)
    except Exception:
        pass


def apply_auth_env(ctx: CreateContext) -> None:
    auth_config: AuthConfig | None = ctx.auth_config
    if auth_config:
        if auth_config.oauth_token:
            ctx.config["environment"][
                "CLAUDE_CODE_OAUTH_TOKEN"
            ] = auth_config.oauth_token
        elif auth_config.api_key:
            ctx.config["environment"]["ANTHROPIC_API_KEY"] = auth_config.api_key
            if auth_config.base_url:
                ctx.config["environment"]["ANTHROPIC_BASE_URL"] = auth_config.base_url
    else:
        if oauth_token := os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            ctx.config["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        elif api_key := os.environ.get("ANTHROPIC_API_KEY"):
            ctx.config["environment"]["ANTHROPIC_API_KEY"] = api_key
            if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
                ctx.config["environment"]["ANTHROPIC_BASE_URL"] = base_url

    if ctx.session_id:
        ctx.config["environment"]["CLAUDE_CODE_SESSION_ID"] = ctx.session_id


async def apply_plugin_hook(ctx: CreateContext) -> None:
    try:
        if ctx.plugin is not None and hasattr(ctx.plugin, "prepare_container"):
            ctx.config = await ctx.plugin.prepare_container(
                ctx.config, session_id=ctx.session_id
            )
    except Exception:
        pass


def enforce_mount_policy(ctx: CreateContext) -> None:
    try:
        if isinstance(ctx.config.get("tmpfs"), dict):
            tmap = ctx.config.get("tmpfs") or {}
            if set(tmap.keys()) != {"/tmp"}:
                tmp_only = {}
                if "/tmp" in tmap:
                    tmp_only["/tmp"] = tmap["/tmp"]
                ctx.config["tmpfs"] = tmp_only
        for k in ("devices", "device_requests"):
            if k in ctx.config:
                del ctx.config[k]
        if bool(ctx.config.get("privileged")):
            ctx.config["privileged"] = False
        if ctx.config.get("cap_add"):
            ctx.config["cap_add"] = []
        secopts = set(ctx.config.get("security_opt") or [])
        secopts.add("no-new-privileges:true")
        ctx.config["security_opt"] = list(secopts)
    except Exception:
        pass


def summarize_env(ctx: CreateContext) -> None:
    try:
        env = ctx.config.get("environment", {}) or {}
        logger.debug(
            "Container config ready: env_keys=%d, cpu=%.2f, mem=%s, swap=%s",
            len(env),
            (ctx.config.get("nano_cpus", 0) or 0) / 1_000_000_000,
            ctx.config.get("mem_limit"),
            ctx.config.get("memswap_limit"),
        )
        if ctx.event_callback:
            ctx.event_callback(
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


__all__ = [
    "build_base_config",
    "apply_auth_env",
    "apply_plugin_hook",
    "enforce_mount_policy",
    "summarize_env",
]
