"""
Docker container management for instance execution.

Handles container lifecycle, resource limits, volume mounts, and cleanup.
"""

import asyncio
import time
import json
import logging
import warnings
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import docker
from docker.errors import ImageNotFound
from docker.types import Mount, Ulimit

if TYPE_CHECKING:
    from docker.errors import DockerException, NotFound
    from docker.models.containers import Container
else:
    try:
        from docker.errors import DockerException, NotFound
        from docker.models.containers import Container
    except ImportError:
        DockerException = Exception
        NotFound = Exception
        Container = Any

from . import DockerError, TimeoutError
from .types import AuthConfig
from ..utils.platform_utils import normalize_path_for_docker

# Suppress the urllib3 exception on close that happens with docker-py
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Workaround: suppress benign "I/O operation on closed file" from urllib3 during close
try:  # pragma: no cover - environment dependent
    import urllib3.response as _u3r  # type: ignore

    _original_close = _u3r.HTTPResponse.close  # type: ignore[attr-defined]

    def _patched_close(self):  # type: ignore[no-redef]
        try:
            _original_close(self)
        except (ValueError, OSError):
            # Ignore noisy close/flush errors from underlying fp
            pass

    _u3r.HTTPResponse.close = _patched_close  # type: ignore[attr-defined]
except Exception:
    # If patching fails, just continue; this is a best-effort mitigation
    pass

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker containers for instance execution."""

    def __init__(self, api_timeout: int = 20):
        """Initialize Docker client.

        Args:
            api_timeout: Per-request timeout (seconds) for Docker SDK calls.
        """
        try:
            # Use a bounded API timeout so SDK calls can't hang indefinitely
            self.client = docker.from_env(timeout=int(api_timeout))
        except DockerException as e:
            raise DockerError(f"Failed to connect to Docker daemon: {e}")

    def close(self) -> None:
        """Close the Docker client connection."""
        try:
            self.client.close()
        except Exception:
            # Ignore errors during close
            pass

    # Heartbeat task registry
    _hb_tasks: Dict[str, asyncio.Task] = {}

    async def start_heartbeat(
        self, container: Container, interval_s: float = 15.0
    ) -> None:
        """Start a periodic heartbeat writer inside the container at /home/node/.pitaya/last_active.

        Emits DEBUG logs indicating exec_create/exec_start success for easier diagnostics.
        """
        if not container:
            return
        cid = getattr(container, "id", None) or ""
        if not cid or cid in self._hb_tasks:
            return

        def _iso_millis(dt: datetime) -> str:
            s = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
            return s[:-3] + "Z"

        async def _hb():
            try:
                while True:
                    try:
                        # Ensure directory exists and write timestamp
                        exec1 = container.client.api.exec_create(
                            container.id,
                            "sh -lc 'mkdir -p /home/node/.pitaya'",
                            stdout=False,
                            stderr=False,
                        )
                        try:
                            logger.debug(
                                "heartbeat: exec_create mkdir id=%s",
                                exec1.get("Id", "-"),
                            )
                        except Exception:
                            pass
                        try:
                            container.client.api.exec_start(exec1["Id"], detach=True)
                            logger.debug(
                                "heartbeat: exec_start mkdir ok id=%s",
                                exec1.get("Id", "-"),
                            )
                        except Exception:
                            pass
                        ts = _iso_millis(datetime.now(timezone.utc))
                        exec2 = container.client.api.exec_create(
                            container.id,
                            f"sh -lc 'printf %s {ts} > /home/node/.pitaya/last_active'",
                            stdout=False,
                            stderr=False,
                        )
                        try:
                            logger.debug(
                                "heartbeat: exec_create write id=%s ts=%s",
                                exec2.get("Id", "-"),
                                ts,
                            )
                        except Exception:
                            pass
                        try:
                            container.client.api.exec_start(exec2["Id"], detach=True)
                            logger.debug(
                                "heartbeat: exec_start write ok id=%s",
                                exec2.get("Id", "-"),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                    await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                # One final heartbeat on cancel
                try:
                    ts = _iso_millis(datetime.now(timezone.utc))
                    exec3 = container.client.api.exec_create(
                        container.id,
                        f"sh -lc 'printf %s {ts} > /home/node/.pitaya/last_active'",
                        stdout=False,
                        stderr=False,
                    )
                    try:
                        logger.debug(
                            "heartbeat: final exec_create id=%s ts=%s",
                            exec3.get("Id", "-"),
                            ts,
                        )
                    except Exception:
                        pass
                    try:
                        container.client.api.exec_start(exec3["Id"], detach=True)
                        logger.debug(
                            "heartbeat: final exec_start ok id=%s", exec3.get("Id", "-")
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
                return

        self._hb_tasks[cid] = asyncio.create_task(_hb())

    async def stop_heartbeat(self, container: Container) -> None:
        cid = getattr(container, "id", None) or ""
        task = self._hb_tasks.pop(cid, None)
        if task:
            task.cancel()

    async def _async_iter(self, blocking_iter):
        """Convert a blocking iterator to an async iterator."""
        loop = asyncio.get_event_loop()

        # Create a sentinel to detect end of iteration
        sentinel = object()

        while True:
            # Get next item from iterator in thread pool
            item = await loop.run_in_executor(
                None, lambda: next(blocking_iter, sentinel)
            )

            if item is sentinel:
                break

            yield item

    async def initialize(self) -> None:
        """
        Initialize Docker manager with startup tasks.

        Per spec, no global orphan cleanup is performed.
        """

    async def validate_environment(self) -> bool:
        """Validate Docker daemon is accessible and working."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.ping)
            return True
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Docker validation failed: {e}")
            return False

    async def create_container(
        self,
        container_name: str,
        workspace_dir: Path,
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
        """
        Create a Docker container for instance execution.

        Args:
            container_name: Full container name (provided by orchestration)
            workspace_dir: Path to prepared git workspace
            cpu_count: Number of CPUs to allocate
            memory_gb: Memory limit in GB
            memory_swap_gb: Total memory + swap limit in GB
            run_id: Run identifier
            strategy_execution_id: Strategy execution identifier
            instance_id: Instance identifier
            image: Docker image to use
            session_id: Agent session to resume
            auth_config: Authentication configuration
            reuse_container: Whether to reuse existing container

        Returns:
            Created or reused container instance
        """
        try:
            phase_start = time.monotonic()
            logger.info(
                f"create_container entry: name={container_name}, image={image}, reuse={reuse_container}, ws={workspace_dir}"
            )
            if event_callback:
                event_callback(
                    {
                        "type": "instance.container_create_entry",
                        "data": {
                            "container_name": container_name,
                            "workspace_dir": str(workspace_dir),
                            "image": image,
                            "reuse": bool(reuse_container),
                        },
                    }
                )

            # Ensure required image exists locally; fail fast with clear message
            loop = asyncio.get_event_loop()
            try:
                img_check_start = time.monotonic()
                logger.info(f"Checking Docker image exists: {image}")
                if event_callback:
                    event_callback(
                        {
                            "type": "instance.container_image_check",
                            "data": {"image": image},
                        }
                    )
                # Bound the image-inspect call with a timeout to avoid hangs
                img_future = loop.run_in_executor(
                    None, lambda: self.client.images.get(image)
                )
                await asyncio.wait_for(img_future, timeout=10)
                logger.info(
                    f"Docker image found: {image} (%.2fs)",
                    time.monotonic() - img_check_start,
                )
            except asyncio.TimeoutError:
                if event_callback:
                    event_callback(
                        {
                            "type": "instance.container_image_check_timeout",
                            "data": {"image": image, "timeout_s": 10},
                        }
                    )
                raise DockerError(
                    "Docker image check timed out after 10s. Docker daemon may be unresponsive; try restarting Docker Desktop."
                )
            except ImageNotFound:
                raise DockerError(
                    f"Docker image '{image}' not found locally. Build it with: docker build -t {image} ."
                )
            except (docker.errors.APIError, docker.errors.DockerException) as e:
                raise DockerError(f"Failed to inspect Docker image '{image}': {e}")

            # Check for existing container if reuse_container=True
            if reuse_container:
                try:
                    exist_check_start = time.monotonic()
                    logger.info(f"Checking for existing container: {container_name}")
                    get_future = loop.run_in_executor(
                        None, lambda: self.client.containers.get(container_name)
                    )
                    existing = await asyncio.wait_for(get_future, timeout=10)

                    # Reload to get current status and mounts
                    await asyncio.wait_for(
                        loop.run_in_executor(None, existing.reload), timeout=10
                    )
                    logger.info(
                        f"Found existing container {container_name} with status {existing.status} (%.2fs)",
                        time.monotonic() - exist_check_start,
                    )

                    # Determine if we must enforce RO /workspace (spec: review tasks)
                    require_ro = str(import_policy).lower() == "never"

                    def _workspace_is_ro(cont) -> bool:
                        try:
                            mounts = cont.attrs.get("Mounts", [])
                            for m in mounts:
                                # Docker reports RW boolean (True if writable)
                                if str(m.get("Destination")) == "/workspace":
                                    return not bool(m.get("RW", True))
                        except Exception:
                            pass
                        # Unknown -> assume not RO
                        return False

                    if require_ro and not _workspace_is_ro(existing):
                        # Replace container to honor RO mount requirement
                        try:
                            old_id = getattr(existing, "id", "")[:12]
                            if event_callback:
                                event_callback(
                                    {
                                        "type": "runner.container.replaced",
                                        "data": {
                                            "old_id": old_id,
                                            "reason": "ro_required",
                                        },
                                    }
                                )
                            try:
                                await asyncio.wait_for(
                                    loop.run_in_executor(None, existing.stop),
                                    timeout=15,
                                )
                            except Exception:
                                pass
                            try:
                                await asyncio.wait_for(
                                    loop.run_in_executor(None, existing.remove),
                                    timeout=15,
                                )
                            except Exception:
                                pass
                            logger.info(
                                "Replaced existing container to enforce RO /workspace"
                            )
                        except Exception:
                            logger.warning(
                                "Failed to replace existing container; creating new"
                            )
                        # Fall through to creation of a new one
                    else:
                        # Reuse the existing container as-is (start if needed)
                        if existing.status == "running":
                            logger.info(f"Reusing running container {container_name}")
                            return existing
                        elif existing.status in ["exited", "created"]:
                            logger.info(f"Starting existing container {container_name}")
                            await asyncio.wait_for(
                                loop.run_in_executor(None, existing.start), timeout=20
                            )
                            # Wait for it to be running
                            for _ in range(10):
                                await asyncio.sleep(0.5)
                                await loop.run_in_executor(None, existing.reload)
                                if existing.status == "running":
                                    logger.info(
                                        f"Existing container {container_name} is running"
                                    )
                                    return existing
                            raise DockerError(
                                f"Failed to start existing container {container_name}"
                            )
                        else:
                            logger.info(
                                f"Container {container_name} in unexpected state {existing.status}, creating new"
                            )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout while checking/starting existing container; proceeding to create new"
                    )
                except NotFound:
                    logger.debug(f"Container {container_name} not found, creating new")
                except (docker.errors.APIError, docker.errors.DockerException) as e:
                    logger.warning(
                        f"Error checking existing container: {e}, creating new"
                    )
            # Convert to absolute path
            workspace_dir = workspace_dir.resolve()

            # Extract strategy index from container name for volume naming
            # Container name format: pitaya_{run_id}_s{sidx}_k{khash}[_rXXXX]
            name_parts = container_name.split("_")
            if len(name_parts) >= 4:
                sidx = name_parts[-2].lstrip("s")
            else:
                sidx = "0"

            # Named volume for agent home as per spec (GHASH)
            import hashlib

            eff_sgk = session_group_key or (
                task_key or instance_id or container_name or ""
            )
            # Scope: default run; allow global only when explicitly enabled via param
            allow_global = bool(allow_global_session_volume)
            if allow_global:
                payload = {
                    "session_group_key": eff_sgk,
                    "plugin": (plugin_name or ""),
                    "model": (resolved_model_id or ""),
                }
                enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
                ghash = hashlib.sha256(
                    enc.encode("utf-8", errors="ignore")
                ).hexdigest()[:8]
                volume_name = f"pitaya_home_g{ghash}"
            else:
                payload = {"session_group_key": eff_sgk}
                enc = json.dumps(payload, separators=(",", ":"), sort_keys=True)
                ghash = hashlib.sha256(
                    enc.encode("utf-8", errors="ignore")
                ).hexdigest()[:8]
                # Scope volumes per-run and per-strategy to avoid concurrent copy-up races
                volume_name = f"pitaya_home_{(run_id or 'norun')}_s{sidx}_g{ghash}"
            # KHASH based on durable task key if provided
            khash = (
                hashlib.sha256(
                    (task_key or "").encode("utf-8", errors="ignore")
                ).hexdigest()[:8]
                if task_key
                else ""
            )

            # Platform detection for SELinux flag
            import platform
            import os

            selinux_mode = ""
            if platform.system() == "Linux" and os.path.exists("/sys/fs/selinux"):
                selinux_mode = "z"

            # Review workspace RO mode: fixed default 'ro'
            review_mode = "ro"

            # Decide workspace mount strategy
            mounts: List[Mount] = []
            volumes: Dict[str, Dict[str, str]] = {}
            ws_source = normalize_path_for_docker(workspace_dir)
            ws_ro = import_policy == "never" and review_mode == "ro"
            if selinux_mode:
                # Use volumes mapping to apply ':z' (and 'ro' when needed)
                mode = "z,ro" if ws_ro else "z"
                volumes[ws_source] = {"bind": "/workspace", "mode": mode}
            else:
                mounts.append(
                    Mount(
                        target="/workspace",
                        source=ws_source,
                        type="bind",
                        read_only=ws_ro,
                    )
                )
            # Named volume for node home (session persistence)
            mounts.append(Mount(target="/home/node", source=volume_name, type="volume"))
            try:
                logger.info(
                    f"Mounts prepared: workspace={mounts[0].source} -> /workspace (ro={mounts[0].read_only}), home=volume:{volume_name} -> /home/node"
                )
                if event_callback:
                    event_callback(
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

            # Container configuration following spec section 3.4 (with hardening)
            config: Dict[str, Any] = {
                "image": image,
                "name": container_name,
                "command": "sleep infinity",  # Keep container running
                "detach": True,
                "labels": {
                    "pitaya": "true",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id or "",
                    "strategy_execution_id": strategy_execution_id or "",
                    "strategy_index": sidx,
                    "task_key": task_key or "",
                    "session_group_key": session_group_key or "",
                    "pitaya.last_active_ts": datetime.now(timezone.utc).isoformat(),
                    "instance_id": instance_id or "",
                },
                # Use explicit Mounts; avoid old volumes/binds ambiguity
                "mounts": mounts,
                # tmpfs mount for /tmp (fixed sensible default)
                "tmpfs": {"/tmp": "rw,size=512m"},
                "working_dir": "/workspace",
                "read_only": True,  # Lock down filesystem except mounts
                "user": "node",  # Run as non-root node user
                "environment": {
                    "PYTHONUNBUFFERED": "1",
                    "GIT_AUTHOR_NAME": "AI Agent",
                    "GIT_AUTHOR_EMAIL": "agent@pitaya.local",
                    "GIT_COMMITTER_NAME": "AI Agent",
                    "GIT_COMMITTER_EMAIL": "agent@pitaya.local",
                    # Normative envs for in-container awareness/debugging
                    "TASK_KEY": task_key or "",
                    "SESSION_GROUP_KEY": session_group_key or "",
                    "KHASH": khash,
                    "GHASH": ghash,
                },
                # Resource limits will be applied via supported keys below
                "mem_limit": f"{memory_gb}g",
                "memswap_limit": f"{memory_swap_gb}g",
                "auto_remove": False,
            }
            # Apply workspace volumes mapping when needed (SELinux)
            if volumes:
                config["volumes"] = volumes

            # Security hardening: drop capabilities, block new privileges, set pids/nofile limits
            try:
                config["cap_drop"] = ["ALL"]
            except Exception:
                pass
            try:
                # Docker default seccomp + no-new-privileges
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

            # Enforce network egress policy
            try:
                eg = str(network_egress).lower()
                if eg == "offline":
                    # Align with spec literal: use network_mode='none'
                    # Ensure network_disabled is not set to avoid conflicts
                    if "network_disabled" in config:
                        try:
                            del config["network_disabled"]
                        except Exception:
                            pass
                    config["network_mode"] = "none"
                # For proxy/online we rely on default bridge; proxy envs handled below
            except Exception:
                pass

            # Merge extra env vars (e.g., auth) into environment pre-creation
            if extra_env:
                try:
                    config["environment"].update(extra_env)
                except Exception:
                    pass
            # Provide normative env hints inside the container for debugging/compliance
            try:
                import hashlib

                eff_sgk = session_group_key or (
                    task_key or instance_id or container_name
                )
                # KHASH short8 over TASK_KEY
                durable = task_key or (instance_id or "")
                khash = hashlib.sha256(
                    str(durable).encode("utf-8", errors="ignore")
                ).hexdigest()[:8]
                # GHASH run-scope short8 over JCS({"session_group_key": EFFECTIVE_SGK})
                payload = json.dumps(
                    {"session_group_key": eff_sgk},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                ghash = hashlib.sha256(
                    payload.encode("utf-8", errors="ignore")
                ).hexdigest()[:8]
                config["environment"].update(
                    {
                        "TASK_KEY": str(durable),
                        "SESSION_GROUP_KEY": str(eff_sgk),
                        "KHASH": khash,
                        "GHASH": ghash,
                    }
                )
            except Exception:
                pass
            # Proxy egress support: pass host proxy envs when requested
            try:
                if str(network_egress).lower() == "proxy":
                    import os as _os

                    for k in (
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "NO_PROXY",
                        "http_proxy",
                        "https_proxy",
                        "no_proxy",
                    ):
                        if k in _os.environ:
                            config["environment"][k] = _os.environ[k]
            except Exception:
                pass

            # Apply CPU limit using nano_cpus when possible
            try:
                config["nano_cpus"] = int(max(1, int(cpu_count)) * 1_000_000_000)
            except Exception:
                pass

            # Add Anthropic authentication based on auth_config (plugin-specific)
            if auth_config:
                if auth_config.oauth_token:
                    config["environment"][
                        "CLAUDE_CODE_OAUTH_TOKEN"
                    ] = auth_config.oauth_token
                elif auth_config.api_key:
                    config["environment"]["ANTHROPIC_API_KEY"] = auth_config.api_key
                    if auth_config.base_url:
                        config["environment"][
                            "ANTHROPIC_BASE_URL"
                        ] = auth_config.base_url
            else:
                # Fallback to environment variables if no auth_config
                if oauth_token := os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
                    config["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
                elif api_key := os.environ.get("ANTHROPIC_API_KEY"):
                    config["environment"]["ANTHROPIC_API_KEY"] = api_key
                    if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
                        config["environment"]["ANTHROPIC_BASE_URL"] = base_url

            # Add session ID for agent-tool resumability (spec section 3.4)
            if session_id:
                config["environment"]["CLAUDE_CODE_SESSION_ID"] = session_id

            # Allow plugin to adjust container configuration pre-creation
            try:
                if plugin is not None and hasattr(plugin, "prepare_container"):
                    config = await plugin.prepare_container(
                        config, session_id=session_id
                    )
            except Exception:
                # Continue with base config if plugin hook fails
                pass

            # Enforce mount policy after plugin hook: keep existing mounts prepared above.
            # We do not alter the Docker SDK Mount objects here to avoid losing required mounts
            # due to SDK attribute differences. We only harden obvious risky flags.
            try:
                # tmpfs: ensure only /tmp is present when tmpfs exists
                if isinstance(config.get("tmpfs"), dict):
                    tmap = config.get("tmpfs") or {}
                    if set(tmap.keys()) != {"/tmp"}:
                        tmp_only = {}
                        if "/tmp" in tmap:
                            tmp_only["/tmp"] = tmap["/tmp"]
                        config["tmpfs"] = tmp_only
                # Disallow devices/privileged/cap_add additions
                for k in ("devices", "device_requests"):
                    if k in config:
                        try:
                            del config[k]
                        except Exception:
                            pass
                if bool(config.get("privileged")):
                    config["privileged"] = False
                if config.get("cap_add"):
                    config["cap_add"] = []
                # Ensure no-new-privileges remains
                try:
                    secopts = set(config.get("security_opt") or [])
                    secopts.add("no-new-privileges:true")
                    config["security_opt"] = list(secopts)
                except Exception:
                    pass
            except Exception:
                # If enforcement fails, continue with original safe mounts
                pass

            # Summarize environment without leaking secrets
            try:
                env = config.get("environment", {}) or {}
                logger.debug(
                    "Container config ready: env_keys=%d, cpu=%.2f, mem=%s, swap=%s",
                    len(env),
                    (config.get("nano_cpus", 0) or 0) / 1_000_000_000,
                    config.get("mem_limit"),
                    config.get("memswap_limit"),
                )
                if event_callback:
                    event_callback(
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

            # Create container with a bounded timeout to avoid hangs
            logger.info(
                f"Creating Docker container via API: name={container_name}, image={image}"
            )
            if event_callback:
                event_callback(
                    {
                        "type": "instance.container_create_attempt",
                        "data": {"container_name": container_name, "image": image},
                    }
                )
            create_start = time.monotonic()
            create_future = loop.run_in_executor(
                None, lambda: self.client.containers.create(**config)
            )
            try:
                container = await asyncio.wait_for(create_future, timeout=20)
                logger.info(
                    "Docker create returned id=%s (%.2fs)",
                    container.id[:12] if container else "unknown",
                    time.monotonic() - create_start,
                )
            except asyncio.TimeoutError:
                if event_callback:
                    event_callback(
                        {
                            "type": "instance.container_create_timeout",
                            "data": {"container_name": container_name, "timeout_s": 20},
                        }
                    )
                raise DockerError(
                    f"Docker create timed out after 20s for {container_name}. Check Docker Desktop and volume sharing."
                )
            except (docker.errors.APIError, docker.errors.DockerException) as e:
                # Handle name conflicts gracefully to avoid races during parallel creation
                msg = str(e)
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                is_conflict = False
                try:
                    is_conflict = (
                        (status_code == 409)
                        or ("already in use" in msg)
                        or ("Conflict" in msg)
                    )
                except Exception:
                    is_conflict = False

                if is_conflict:
                    if event_callback:
                        event_callback(
                            {
                                "type": "instance.container_create_conflict",
                                "data": {
                                    "container_name": container_name,
                                    "error": msg,
                                },
                            }
                        )
                    # Attempt to adopt the existing container by this name. This addresses races where
                    # a parallel attempt (or a previous retry) created the container just before us.
                    adopt_start = time.monotonic()
                    for i in range(6):  # Try up to ~3s total
                        try:
                            get_future2 = loop.run_in_executor(
                                None, lambda: self.client.containers.get(container_name)
                            )
                            existing2 = await asyncio.wait_for(get_future2, timeout=5)
                            # Ensure it's started
                            try:
                                await asyncio.wait_for(
                                    loop.run_in_executor(None, existing2.reload),
                                    timeout=5,
                                )
                            except Exception:
                                pass
                            if existing2.status != "running":
                                try:
                                    await asyncio.wait_for(
                                        loop.run_in_executor(None, existing2.start),
                                        timeout=15,
                                    )
                                except Exception:
                                    pass
                                # Give it a moment to reach running
                                for _ in range(10):
                                    await asyncio.sleep(0.2)
                                    try:
                                        await loop.run_in_executor(
                                            None, existing2.reload
                                        )
                                    except Exception:
                                        pass
                                    if existing2.status == "running":
                                        break
                            # Report adoption
                            logger.info(
                                "Adopted existing container %s after conflict (%.2fs)",
                                container_name,
                                time.monotonic() - adopt_start,
                            )
                            if event_callback:
                                event_callback(
                                    {
                                        "type": "instance.container_adopted",
                                        "data": {
                                            "container_name": container_name,
                                            "container_id": getattr(
                                                existing2, "id", ""
                                            )[:12],
                                        },
                                    }
                                )
                            container = existing2
                            break
                        except NotFound:
                            # Container may not be visible yet; brief backoff and retry
                            await asyncio.sleep(0.5)
                            continue
                        except (docker.errors.APIError, docker.errors.DockerException):
                            await asyncio.sleep(0.5)
                            continue

                    if not container:
                        # Could not adopt; surface a clearer error
                        if event_callback:
                            event_callback(
                                {
                                    "type": "instance.container_create_failed",
                                    "data": {
                                        "container_name": container_name,
                                        "error": msg,
                                    },
                                }
                            )
                        raise DockerError(
                            f"Docker create name conflict for {container_name} but adoption failed: {e}"
                        )
                else:
                    if event_callback:
                        event_callback(
                            {
                                "type": "instance.container_create_failed",
                                "data": {
                                    "container_name": container_name,
                                    "error": str(e),
                                },
                            }
                        )
                    raise DockerError(f"Docker create failed for {container_name}: {e}")

            # Start container (bounded timeout) unless already running
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
                    if event_callback:
                        event_callback(
                            {
                                "type": "instance.container_start_timeout",
                                "data": {
                                    "container_name": container_name,
                                    "timeout_s": 15,
                                },
                            }
                        )
                    raise DockerError(
                        f"Docker start timed out after 15s for {container_name}."
                    )
                except (docker.errors.APIError, docker.errors.DockerException) as e:
                    if event_callback:
                        event_callback(
                            {
                                "type": "instance.container_start_failed",
                                "data": {
                                    "container_name": container_name,
                                    "error": str(e),
                                },
                            }
                        )
                    raise DockerError(f"Docker start failed for {container_name}: {e}")

            # Wait for container to be running
            for i in range(20):
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, container.reload)
                logger.debug(f"Container status check {i}: {container.status}")
                if container.status == "running":
                    break
            else:
                raise DockerError(
                    f"Container {container_name} failed to reach running state"
                )

            logger.info(
                f"Created container {container_name} (ID: {container.id[:12]}) in %.2fs",
                time.monotonic() - phase_start,
            )
            if event_callback:
                event_callback(
                    {
                        "type": "instance.container_created",
                        "data": {
                            "container_name": container_name,
                            "container_id": container.id[:12],
                        },
                    }
                )
            return container

        except DockerException as e:
            raise DockerError(f"Failed to create container: {e}")

    async def execute_command(
        self,
        container: Container,
        command: List[str],
        plugin: Any,  # RunnerPlugin
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_seconds: int = 3600,
        max_turns: Optional[int] = None,
        stream_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute AI tool command in container and parse output.

        Args:
            container: Container to execute in
            command: Command to run
            plugin: Plugin instance for parsing output
            event_callback: Callback for parsed events
            timeout_seconds: Maximum execution time

        Returns:
            Result data including session_id, metrics, final_message
        """
        try:
            # Convert command to shell string
            cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
            try:
                _pname = getattr(plugin, "name", "tool")
            except Exception:
                _pname = "tool"
            logger.info(f"Executing {_pname}: {cmd_str}")

            # Ensure Codex home exists if running Codex (some builds canonicalize CODEX_HOME and require presence)
            try:
                if str(getattr(plugin, "name", "")) == "codex":
                    # Best-effort, ignore errors
                    _mk = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.client.api.exec_create(
                            container.id,
                            "sh -lc 'mkdir -p /home/node/.codex'",
                            stdout=False,
                            stderr=False,
                            tty=False,
                            workdir="/workspace",
                        ),
                    )
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: container.client.api.exec_start(
                                _mk["Id"], detach=True
                            ),
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            # Create exec instance
            # IMPORTANT: pass the command as a list to preserve argument boundaries
            # (joining into a single string can cause the tool to misparse long prompts)
            loop = asyncio.get_event_loop()
            exec_instance = await loop.run_in_executor(
                None,
                lambda: container.client.api.exec_create(
                    container.id,
                    command,  # pass argv list, not joined string
                    stdout=True,
                    stderr=True,
                    tty=False,
                    workdir="/workspace",
                ),
            )
            # Optional raw stream tee to file (writes everything verbatim)
            raw_f = None
            if stream_log_path:
                try:
                    p = Path(stream_log_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    raw_f = p.open("a", encoding="utf-8", errors="replace")
                    ts = datetime.now(timezone.utc).isoformat()
                    # Header with minimal context; avoid leaking secrets
                    try:
                        cname = getattr(container, "name", None) or "unknown"
                        cid = (getattr(container, "id", None) or "unknown")[:12]
                    except Exception:
                        cname = "unknown"
                        cid = "unknown"
                    header = (
                        f"=== EXEC START {ts} ===\n"
                        f"container_id={cid} name={cname}\n"
                        f"workdir=/workspace\n"
                        f"command={' '.join(command)}\n"
                        f"plugin={getattr(plugin, 'name', 'tool')}\n"
                    )
                    raw_f.write(header)
                    raw_f.write(f"exec_id={exec_instance.get('Id')}\n")
                    raw_f.flush()
                except Exception:
                    raw_f = None

            # Start exec and get output stream
            output_stream = await loop.run_in_executor(
                None,
                lambda: container.client.api.exec_start(
                    exec_instance["Id"], stream=True
                ),
            )

            # Parse output with timeout
            parser_state: Dict[str, Any] = {}  # Plugin-specific parser state
            start_time = asyncio.get_event_loop().time()

            raw_lines: list[str] = []  # capture non-JSON output for diagnostics

            async def parse_stream():
                # Wrap blocking iterator to make it async
                turns_seen = 0
                last_activity = asyncio.get_event_loop().time()

                async def idle_watcher():
                    # Emit periodic progress if no events for a while
                    while True:
                        await asyncio.sleep(5)
                        now = asyncio.get_event_loop().time()
                        idle = now - last_activity
                        if idle >= 10:
                            try:
                                if event_callback:
                                    event_callback(
                                        {
                                            "type": "instance.progress",
                                            "data": {
                                                "phase": "model_wait",
                                                "idle_seconds": int(idle),
                                            },
                                        }
                                    )
                            except Exception:
                                pass

                watcher = asyncio.create_task(idle_watcher())
                async for chunk in self._async_iter(output_stream):
                    # Check timeout
                    if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            f"Agent execution exceeded {timeout_seconds}s timeout"
                        )

                    # Decode chunk
                    raw_chunk = chunk
                    if isinstance(chunk, bytes):
                        try:
                            # Tee raw bytes to file before any processing
                            if raw_f is not None:
                                raw_f.write(raw_chunk.decode("utf-8", errors="replace"))
                                raw_f.flush()
                        except Exception:
                            pass
                        chunk = chunk.decode("utf-8", errors="replace")
                    else:
                        # Chunk already str
                        try:
                            if raw_f is not None:
                                raw_f.write(str(chunk))
                                raw_f.flush()
                        except Exception:
                            pass

                    # Parse lines
                    for line in chunk.strip().split("\n"):
                        if not line:
                            continue

                        try:
                            # Parse line using plugin
                            parsed = await plugin.parse_events(line, parser_state)
                            if parsed:
                                # Emit event
                                if event_callback:
                                    event_callback(parsed)

                                # Allow other tasks to run
                                await asyncio.sleep(0)
                                last_activity = asyncio.get_event_loop().time()

                                # Enforce max_turns if configured
                                if max_turns is not None and isinstance(max_turns, int):
                                    try:
                                        if (
                                            str(parsed.get("type", "")).lower()
                                            == "turn_complete"
                                        ):
                                            turns_seen += 1
                                            if turns_seen >= max_turns:
                                                # Stop parsing further events
                                                return
                                    except Exception:
                                        pass
                            else:
                                # Non-JSON or unparsed line; emit as diagnostic
                                try:
                                    msg = line[:1000]
                                    raw_lines.append(msg)
                                    if len(raw_lines) > 200:
                                        raw_lines.pop(0)
                                    if event_callback:
                                        event_callback(
                                            {
                                                "type": "log",
                                                "timestamp": datetime.now(
                                                    timezone.utc
                                                ).isoformat(),
                                                "stream": "stdout",
                                                "message": msg,
                                            }
                                        )
                                except Exception:
                                    pass

                        except json.JSONDecodeError:
                            # Non-JSON output: emit as diagnostic log event and capture tail
                            try:
                                msg = line[:1000]
                                raw_lines.append(msg)
                                if len(raw_lines) > 200:
                                    raw_lines.pop(0)
                                if event_callback:
                                    event_callback(
                                        {
                                            "type": "log",
                                            "timestamp": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "stream": "stdout",
                                            "message": msg,
                                        }
                                    )
                                last_activity = asyncio.get_event_loop().time()
                            except Exception:
                                pass
                try:
                    watcher.cancel()
                except Exception:
                    pass

            # Run parser with timeout
            try:
                await asyncio.wait_for(parse_stream(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Agent execution exceeded {timeout_seconds}s timeout"
                )

            # Check exec result
            exec_info = await loop.run_in_executor(
                None, lambda: container.client.api.exec_inspect(exec_instance["Id"])
            )

            if exec_info["ExitCode"] != 0:
                # Include last few lines of output to aid debugging
                tail = "\n".join(raw_lines[-10:]) if raw_lines else ""
                msg = f"Command exited with code {exec_info['ExitCode']}"
                if tail:
                    msg += f"\nLast output:\n{tail}"
                try:
                    if raw_f is not None:
                        ts = datetime.now(timezone.utc).isoformat()
                        raw_f.write(
                            f"\n=== EXEC END {ts} exit={exec_info.get('ExitCode', 'nonzero')} ===\n"
                        )
                        raw_f.flush()
                except Exception:
                    pass
                raise DockerError(msg)

            # Extract final result using plugin
            result_data = await plugin.extract_result(parser_state)

            logger.info("Command execution completed successfully")
            try:
                if raw_f is not None:
                    ts = datetime.now(timezone.utc).isoformat()
                    raw_f.write(
                        f"\n=== EXEC END {ts} exit={exec_info.get('ExitCode', 0)} ===\n"
                    )
                    raw_f.flush()
            except Exception:
                pass
            return result_data

        except TimeoutError:
            raise
        except (docker.errors.APIError, docker.errors.DockerException, OSError) as e:
            raise DockerError(f"Failed to execute agent tool: {e}")
        finally:
            try:
                if raw_f is not None:
                    raw_f.close()
            except Exception:
                pass

    async def stop_container(
        self, container: Container, timeout: int | None = None
    ) -> None:
        """
        Stop a container gracefully with SIGTERM, then SIGKILL if needed.

        Args:
            container: Container to stop
            timeout: Seconds to wait for graceful shutdown before force kill. If None, uses 1s default.
        """
        try:
            loop = asyncio.get_event_loop()
            # Docker's stop() sends SIGTERM, waits timeout seconds, then SIGKILL
            eff_timeout = int(timeout) if timeout is not None else 1
            await loop.run_in_executor(
                None, lambda: container.stop(timeout=max(0, eff_timeout))
            )
            logger.info(f"Stopped container {container.name} gracefully")
        except NotFound:
            logger.debug(f"Container {container.name} not found")
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Failed to stop container {container.name}: {e}")

    async def verify_container_tools(
        self, container: Container, tools: list[str]
    ) -> None:
        """Verify required tools are available in the container.

        Raises DockerError if any tool is missing.
        """
        loop = asyncio.get_event_loop()
        missing = []
        for tool in tools:
            found = False
            # Try invoking the tool directly with --version (fast and reliable)
            for candidate in (tool, f"/usr/local/bin/{tool}"):
                try:
                    exec_id = await loop.run_in_executor(
                        None,
                        lambda: container.client.api.exec_create(
                            container.id,
                            f"{candidate} --version",
                            stdout=True,
                            stderr=True,
                        ),
                    )
                    await loop.run_in_executor(
                        None,
                        lambda: container.client.api.exec_start(
                            exec_id["Id"], stream=False
                        ),
                    )
                    inspect = await loop.run_in_executor(
                        None,
                        lambda: container.client.api.exec_inspect(exec_id["Id"]),
                    )
                    if inspect.get("ExitCode", 1) == 0:
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                missing.append(tool)
        if missing:
            raise DockerError(
                f"Required tools missing in container: {', '.join(missing)}"
            )

    async def cleanup_container(
        self,
        container: Container,
        force: bool = True,
        *,
        remove_home_volume: bool = False,
    ) -> None:
        """
        Remove a container after stopping it.

        Args:
            container: Container to remove
            force: Force removal even if running
            remove_home_volume: Also remove the named session volume mounted at /home/node
        """
        try:
            loop = asyncio.get_event_loop()

            # Capture the mounted /home volume name before removal (if requested)
            home_volume_name: Optional[str] = None
            try:
                await loop.run_in_executor(None, container.reload)
                mounts = (container.attrs or {}).get("Mounts", []) or []
                for m in mounts:
                    try:
                        mtype = str(m.get("Type") or m.get("type") or "").lower()
                        if mtype != "volume":
                            continue
                        dest = (
                            m.get("Destination")
                            or m.get("Target")
                            or m.get("Mountpoint")
                            or ""
                        )
                        if str(dest) == "/home/node":
                            # Prefer Name (for named volumes) and fall back to Source
                            name = m.get("Name") or m.get("name") or m.get("Source")
                            if name:
                                home_volume_name = str(name)
                                break
                    except Exception:
                        continue
            except Exception:
                # Best effort: continue even if we cannot inspect mounts
                pass

            # Stop container gracefully first if it's running
            await loop.run_in_executor(None, container.reload)
            if container.status == "running":
                await self.stop_container(container)

            # Now remove the container
            await loop.run_in_executor(None, lambda: container.remove(force=force))
            logger.info(f"Removed container {container.name}")

            # Optionally remove the run-scoped session volume
            if remove_home_volume and home_volume_name:
                try:
                    name = home_volume_name
                    # Only remove Pitaya run-scoped volumes. Keep global session volumes.
                    if name.startswith("pitaya_home_") and not re.fullmatch(
                        r"pitaya_home_g[0-9a-f]{8}", name
                    ):
                        # Resolve volume and remove
                        vol = await loop.run_in_executor(
                            None, lambda: self.client.volumes.get(name)
                        )
                        await loop.run_in_executor(None, lambda: vol.remove(force=True))
                        logger.info(f"Removed session volume {name}")
                    else:
                        logger.debug(
                            f"Skipping volume removal for {name} (not run-scoped or not Pitaya)"
                        )
                except Exception as ve:
                    logger.warning(
                        f"Failed to remove session volume {home_volume_name}: {ve}"
                    )
        except NotFound:
            logger.debug(f"Container {container.name} already removed")
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Failed to remove container {container.name}: {e}")

    # Orphan container cleanup removed — containers are removed immediately on completion

    def get_container(self, name: str) -> Optional[Container]:
        """Return container by name or None if missing."""
        try:
            return self.client.containers.get(name)
        except Exception:
            return None

    # Unused volume cleanup helpers removed
