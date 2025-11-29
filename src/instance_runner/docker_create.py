"""Container creation helpers for DockerManager."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import docker
from docker.errors import ImageNotFound
from docker.types import Mount, Ulimit

from .docker_context import CreateContext
from ..exceptions import DockerError
from ..shared import AuthConfig
from ..utils.platform_utils import normalize_path_for_docker

logger = logging.getLogger(__name__)


def _emit_event(ctx: CreateContext, event_type: str, data: Dict[str, Any]) -> None:
    if ctx.event_callback:
        try:
            ctx.event_callback({"type": event_type, "data": data})
        except Exception:
            pass


async def ensure_image(ctx: CreateContext) -> None:
    loop = ctx.loop
    try:
        img_check_start = time.monotonic()
        logger.info(f"Checking Docker image exists: {ctx.image}")
        _emit_event(
            ctx,
            "instance.container_image_check",
            {"image": ctx.image},
        )
        img_future = loop.run_in_executor(
            None, lambda: ctx.manager.client.images.get(ctx.image)
        )
        await asyncio.wait_for(img_future, timeout=10)
        logger.info(
            f"Docker image found: {ctx.image} (%.2fs)",
            time.monotonic() - img_check_start,
        )
    except asyncio.TimeoutError:
        _emit_event(
            ctx,
            "instance.container_image_check_timeout",
            {"image": ctx.image, "timeout_s": 10},
        )
        raise DockerError(
            "Docker image check timed out after 10s. Docker daemon may be unresponsive; try restarting Docker Desktop."
        )
    except ImageNotFound:
        raise DockerError(
            f"Docker image '{ctx.image}' not found locally. Build it with: docker build -t {ctx.image} ."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        raise DockerError(f"Failed to inspect Docker image '{ctx.image}': {e}")


async def try_reuse_container(ctx: CreateContext) -> Optional[Any]:
    loop = ctx.loop
    if not ctx.reuse_container:
        return None
    try:
        exist_check_start = time.monotonic()
        logger.info(f"Checking for existing container: {ctx.container_name}")
        get_future = loop.run_in_executor(
            None, lambda: ctx.manager.client.containers.get(ctx.container_name)
        )
        existing = await asyncio.wait_for(get_future, timeout=10)

        await asyncio.wait_for(loop.run_in_executor(None, existing.reload), timeout=10)
        logger.info(
            f"Found existing container {ctx.container_name} with status {existing.status} (%.2fs)",
            time.monotonic() - exist_check_start,
        )

        require_ro = str(ctx.import_policy).lower() == "never"

        def _workspace_is_ro(cont) -> bool:
            try:
                mounts = cont.attrs.get("Mounts", [])
                for m in mounts:
                    if str(m.get("Destination")) == "/workspace":
                        return not bool(m.get("RW", True))
            except Exception:
                pass
            return False

        if require_ro and not _workspace_is_ro(existing):
            try:
                old_id = getattr(existing, "id", "")[:12]
                _emit_event(
                    ctx,
                    "runner.container.replaced",
                    {"old_id": old_id, "reason": "ro_required"},
                )
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.stop), timeout=15
                    )
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.remove), timeout=15
                    )
                except Exception:
                    pass
                logger.info("Replaced existing container to enforce RO /workspace")
            except Exception:
                logger.warning("Failed to replace existing container; creating new")
            return None

        if existing.status == "running":
            logger.info(f"Reusing running container {ctx.container_name}")
            return existing
        if existing.status in ["exited", "created"]:
            logger.info(f"Starting existing container {ctx.container_name}")
            await asyncio.wait_for(
                loop.run_in_executor(None, existing.start), timeout=20
            )
            for _ in range(10):
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, existing.reload)
                if existing.status == "running":
                    logger.info(f"Existing container {ctx.container_name} is running")
                    return existing
            raise DockerError(
                f"Failed to start existing container {ctx.container_name}"
            )

        logger.info(
            f"Container {ctx.container_name} in unexpected state {existing.status}, creating new"
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout while checking/starting existing container; proceeding to create new"
        )
    except docker.errors.NotFound:
        logger.debug(f"Container {ctx.container_name} not found, creating new")
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        logger.warning(f"Error checking existing container: {e}, creating new")
    return None


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
    _emit_event(
        ctx,
        "instance.container_mounts_prepared",
        {
            "workspace_source": ctx.ws_source,
            "workspace_read_only": bool(ctx.ws_ro),
            "home_volume": ctx.volume_name,
        },
    )


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
    auth_config: Optional[AuthConfig] = ctx.auth_config
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
        _emit_event(
            ctx,
            "instance.container_config_ready",
            {
                "env_keys": len(env),
                "has_oauth": bool(env.get("CLAUDE_CODE_OAUTH_TOKEN")),
                "has_api_key": bool(env.get("ANTHROPIC_API_KEY")),
                "base_url_set": bool(env.get("ANTHROPIC_BASE_URL")),
            },
        )
    except Exception:
        pass


async def create_and_start(ctx: CreateContext) -> Any:
    loop = ctx.loop
    logger.info(
        f"Creating Docker container via API: name={ctx.container_name}, image={ctx.image}"
    )
    _emit_event(
        ctx,
        "instance.container_create_attempt",
        {"container_name": ctx.container_name, "image": ctx.image},
    )
    create_start = time.monotonic()
    create_future = loop.run_in_executor(
        None, lambda: ctx.manager.client.containers.create(**ctx.config)
    )
    container = None
    try:
        container = await asyncio.wait_for(create_future, timeout=20)
        logger.info(
            "Docker create returned id=%s (%.2fs)",
            container.id[:12] if container else "unknown",
            time.monotonic() - create_start,
        )
    except asyncio.TimeoutError:
        _emit_event(
            ctx,
            "instance.container_create_timeout",
            {"container_name": ctx.container_name, "timeout_s": 20},
        )
        raise DockerError(
            f"Docker create timed out after 20s for {ctx.container_name}. Check Docker Desktop and volume sharing."
        )
    except (docker.errors.APIError, docker.errors.DockerException) as e:
        msg = str(e)
        status_code = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        is_conflict = False
        try:
            is_conflict = (
                (status_code == 409) or ("already in use" in msg) or ("Conflict" in msg)
            )
        except Exception:
            is_conflict = False

        if is_conflict:
            _emit_event(
                ctx,
                "instance.container_create_conflict",
                {"container_name": ctx.container_name, "error": msg},
            )
            adopt_start = time.monotonic()
            for _ in range(6):
                try:
                    get_future2 = loop.run_in_executor(
                        None,
                        lambda: ctx.manager.client.containers.get(ctx.container_name),
                    )
                    existing2 = await asyncio.wait_for(get_future2, timeout=5)
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(None, existing2.reload), timeout=5
                        )
                    except Exception:
                        pass
                    if existing2.status != "running":
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(None, existing2.start), timeout=15
                            )
                        except Exception:
                            pass
                        for _ in range(10):
                            await asyncio.sleep(0.2)
                            try:
                                await loop.run_in_executor(None, existing2.reload)
                            except Exception:
                                pass
                            if existing2.status == "running":
                                break
                    logger.info(
                        "Adopted existing container %s after conflict (%.2fs)",
                        ctx.container_name,
                        time.monotonic() - adopt_start,
                    )
                    _emit_event(
                        ctx,
                        "instance.container_adopted",
                        {
                            "container_name": ctx.container_name,
                            "container_id": getattr(existing2, "id", "")[:12],
                        },
                    )
                    container = existing2
                    break
                except docker.errors.NotFound:
                    await asyncio.sleep(0.5)
                    continue
                except (docker.errors.APIError, docker.errors.DockerException):
                    await asyncio.sleep(0.5)
                    continue

            if not container:
                _emit_event(
                    ctx,
                    "instance.container_create_failed",
                    {"container_name": ctx.container_name, "error": msg},
                )
                raise DockerError(
                    f"Docker create name conflict for {ctx.container_name} but adoption failed: {e}"
                )
        else:
            _emit_event(
                ctx,
                "instance.container_create_failed",
                {"container_name": ctx.container_name, "error": str(e)},
            )
            raise DockerError(f"Docker create failed for {ctx.container_name}: {e}")

    try:
        await loop.run_in_executor(None, container.reload)
    except Exception:
        pass
    if getattr(container, "status", None) != "running":
        start_start = time.monotonic()
        start_future = loop.run_in_executor(None, container.start)
        try:
            await asyncio.wait_for(start_future, timeout=15)
            logger.info(
                "Docker start completed for %s (%.2fs)",
                container.name,
                time.monotonic() - start_start,
            )
        except asyncio.TimeoutError:
            _emit_event(
                ctx,
                "instance.container_start_timeout",
                {"container_name": ctx.container_name, "timeout_s": 15},
            )
            raise DockerError(
                f"Docker start timed out after 15s for {ctx.container_name}."
            )
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            _emit_event(
                ctx,
                "instance.container_start_failed",
                {"container_name": ctx.container_name, "error": str(e)},
            )
            raise DockerError(f"Docker start failed for {ctx.container_name}: {e}")

    for _ in range(20):
        await asyncio.sleep(0.5)
        await loop.run_in_executor(None, container.reload)
        logger.debug(f"Container status check: {container.status}")
        if container.status == "running":
            break
    else:
        raise DockerError(
            f"Container {ctx.container_name} failed to reach running state"
        )

    logger.info(
        f"Created container {ctx.container_name} (ID: {container.id[:12]}) in %.2fs",
        time.monotonic() - ctx.phase_start,
    )
    _emit_event(
        ctx,
        "instance.container_created",
        {"container_name": ctx.container_name, "container_id": container.id[:12]},
    )
    return container


async def create_container(ctx: CreateContext) -> Any:
    ctx.phase_start = time.monotonic()
    logger.info(
        f"create_container entry: name={ctx.container_name}, image={ctx.image}, reuse={ctx.reuse_container}, ws={ctx.workspace_dir}"
    )
    _emit_event(
        ctx,
        "instance.container_create_entry",
        {
            "container_name": ctx.container_name,
            "workspace_dir": str(ctx.workspace_dir),
            "image": ctx.image,
            "reuse": bool(ctx.reuse_container),
        },
    )

    await ensure_image(ctx)
    reused = await try_reuse_container(ctx)
    if reused:
        return reused

    prepare_mounts(ctx)
    build_base_config(ctx)
    apply_auth_env(ctx)
    await apply_plugin_hook(ctx)
    enforce_mount_policy(ctx)
    summarize_env(ctx)
    return await create_and_start(ctx)


__all__ = ["CreateContext", "create_container"]
