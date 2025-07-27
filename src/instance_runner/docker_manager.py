"""
Docker container management for instance execution.

Handles container lifecycle, resource limits, volume mounts, and cleanup.
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import docker

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

# Monkey-patch urllib3's HTTPResponse to fix the flush issue
try:
    import urllib3.response

    _original_close = urllib3.response.HTTPResponse.close  # type: ignore

    def _patched_close(self):  # type: ignore
        try:
            _original_close(self)
        except (ValueError, OSError):
            # Ignore "I/O operation on closed file" errors
            pass

    urllib3.response.HTTPResponse.close = _patched_close  # type: ignore
except Exception:
    # If patching fails, just continue
    pass

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker containers for instance execution."""

    def __init__(self):
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
        except DockerException as e:
            raise DockerError(f"Failed to connect to Docker daemon: {e}")

        # Flag to track if cleanup has been performed
        self._cleanup_performed = False

    def close(self) -> None:
        """Close the Docker client connection."""
        try:
            self.client.close()
        except Exception:
            # Ignore errors during close
            pass

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

        Per spec section 3.2, this performs orphan cleanup on startup.
        """
        if not self._cleanup_performed:
            try:
                logger.info("Performing orphan container cleanup on startup")
                cleaned_count = await self.cleanup_orphaned_containers()
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} orphaned containers")
                self._cleanup_performed = True
            except (DockerError, docker.errors.APIError) as e:
                logger.warning(f"Failed to cleanup orphaned containers on startup: {e}")
                # Don't fail initialization if cleanup fails

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
        image: str = "claude-code:latest",
        session_id: Optional[str] = None,
        auth_config: Optional[AuthConfig] = None,
        reuse_container: bool = True,
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
            session_id: Claude session to resume
            auth_config: Authentication configuration
            reuse_container: Whether to reuse existing container

        Returns:
            Created or reused container instance
        """
        try:
            # Check for existing container if reuse_container=True
            if reuse_container:
                try:
                    loop = asyncio.get_event_loop()
                    existing = await loop.run_in_executor(
                        None, lambda: self.client.containers.get(container_name)
                    )

                    # Reload to get current status
                    await loop.run_in_executor(None, existing.reload)

                    if existing.status == "running":
                        logger.info(f"Reusing running container {container_name}")
                        return existing
                    elif existing.status in ["exited", "created"]:
                        logger.info(f"Starting existing container {container_name}")
                        await loop.run_in_executor(None, existing.start)
                        # Wait for it to be running
                        for _ in range(10):
                            await asyncio.sleep(0.5)
                            await loop.run_in_executor(None, existing.reload)
                            if existing.status == "running":
                                return existing
                        raise DockerError(
                            f"Failed to start existing container {container_name}"
                        )
                    else:
                        logger.info(
                            f"Container {container_name} in unexpected state {existing.status}, creating new"
                        )
                except NotFound:
                    logger.debug(f"Container {container_name} not found, creating new")
                except (docker.errors.APIError, docker.errors.DockerException) as e:
                    logger.warning(
                        f"Error checking existing container: {e}, creating new"
                    )
            # Convert to absolute path
            workspace_dir = workspace_dir.resolve()

            # Extract indices from container name for volume naming
            # Container name format: orchestrator_{run_id}_s{sidx}_i{iidx}
            name_parts = container_name.split("_")
            if len(name_parts) >= 4:
                sidx = name_parts[-2].lstrip("s")
                iidx = name_parts[-1].lstrip("i")
            else:
                # Fallback
                sidx = "0"
                iidx = "0"

            # Named volume for Claude home as per spec
            # Even without run_id, maintain the naming pattern
            volume_name = f"orc_home_{run_id or 'norun'}_s{sidx}_i{iidx}"

            # Platform detection for SELinux flag
            import platform
            import os

            selinux_mode = ""
            if platform.system() == "Linux" and os.path.exists("/sys/fs/selinux"):
                selinux_mode = "z"

            # Container configuration following spec section 3.4
            config: Dict[str, Any] = {
                "image": image,
                "name": container_name,
                "command": "sleep infinity",  # Keep container running
                "detach": True,
                "labels": {
                    "orchestrator": "true",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id or "",
                    "strategy_exec": sidx,
                    "instance_index": iidx,
                    "instance_status": "running",  # Will be updated on completion
                },
                "volumes": [
                    # Workspace mount with conditional SELinux flag
                    # Normalize path for Docker on Windows
                    f"{normalize_path_for_docker(workspace_dir)}:/workspace{':' + selinux_mode if selinux_mode else ''}",
                    # Named volume for /home/node
                    f"{volume_name}:/home/node",
                ],
                # tmpfs mount for /tmp
                "tmpfs": {"/tmp": "rw,size=256m"},
                "working_dir": "/workspace",
                "read_only": True,  # Lock down filesystem except mounts
                "user": "node",  # Run as non-root node user
                "environment": {
                    "PYTHONUNBUFFERED": "1",
                    "GIT_AUTHOR_NAME": "AI Agent",
                    "GIT_AUTHOR_EMAIL": "agent@orchestrator.local",
                    "GIT_COMMITTER_NAME": "AI Agent",
                    "GIT_COMMITTER_EMAIL": "agent@orchestrator.local",
                },
                "cpu_count": cpu_count,
                "mem_limit": f"{memory_gb}g",
                "memswap_limit": f"{memory_swap_gb}g",
                "auto_remove": False,  # Keep containers for debugging/resume
            }

            # Add Claude authentication based on auth_config
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

            # Add session ID for Claude Code resumability (spec section 3.4)
            if session_id:
                config["environment"]["CLAUDE_CODE_SESSION_ID"] = session_id

            # Create container
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None, lambda: self.client.containers.create(**config)
            )

            # Start container
            await loop.run_in_executor(None, container.start)

            # Wait for container to be running
            for _ in range(10):
                await asyncio.sleep(0.5)
                await loop.run_in_executor(None, container.reload)
                if container.status == "running":
                    break
            else:
                raise DockerError(f"Container {container_name} failed to start")

            logger.info(f"Created container {container_name} (ID: {container.id[:12]})")
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
            logger.info(f"Executing Claude Code: {cmd_str}")

            # Create exec instance
            loop = asyncio.get_event_loop()
            exec_instance = await loop.run_in_executor(
                None,
                lambda: container.client.api.exec_create(
                    container.id,
                    cmd_str,
                    stdout=True,
                    stderr=True,
                    tty=False,
                    workdir="/workspace",
                ),
            )

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

            async def parse_stream():
                # Wrap blocking iterator to make it async
                async for chunk in self._async_iter(output_stream):
                    # Check timeout
                    if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            f"Claude Code execution exceeded {timeout_seconds}s timeout"
                        )

                    # Decode chunk
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="replace")

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

                        except json.JSONDecodeError:
                            # Some lines might not be JSON (e.g., error messages)
                            logger.debug(f"Non-JSON output: {line}")

            # Run parser with timeout
            try:
                await asyncio.wait_for(parse_stream(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Claude Code execution exceeded {timeout_seconds}s timeout"
                )

            # Check exec result
            exec_info = await loop.run_in_executor(
                None, lambda: container.client.api.exec_inspect(exec_instance["Id"])
            )

            if exec_info["ExitCode"] != 0:
                raise DockerError(f"Command exited with code {exec_info['ExitCode']}")

            # Extract final result using plugin
            result_data = await plugin.extract_result(parser_state)

            logger.info("Command execution completed successfully")
            return result_data

        except TimeoutError:
            raise
        except (docker.errors.APIError, docker.errors.DockerException, OSError) as e:
            raise DockerError(f"Failed to execute Claude Code: {e}")

    async def stop_container(self, container: Container, timeout: int = 10) -> None:
        """
        Stop a container gracefully with SIGTERM, then SIGKILL if needed.

        Args:
            container: Container to stop
            timeout: Seconds to wait for graceful shutdown before force kill
        """
        try:
            loop = asyncio.get_event_loop()
            # Docker's stop() sends SIGTERM, waits timeout seconds, then SIGKILL
            await loop.run_in_executor(None, lambda: container.stop(timeout=timeout))
            logger.info(f"Stopped container {container.name} gracefully")
        except NotFound:
            logger.debug(f"Container {container.name} not found")
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Failed to stop container {container.name}: {e}")

    async def cleanup_container(self, container: Container, force: bool = True) -> None:
        """
        Remove a container after stopping it.

        Args:
            container: Container to remove
            force: Force removal even if running
        """
        try:
            loop = asyncio.get_event_loop()

            # Stop container gracefully first if it's running
            await loop.run_in_executor(None, container.reload)
            if container.status == "running":
                await self.stop_container(container)

            # Now remove the container
            await loop.run_in_executor(None, lambda: container.remove(force=force))
            logger.info(f"Removed container {container.name}")
        except NotFound:
            logger.debug(f"Container {container.name} already removed")
        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Failed to remove container {container.name}: {e}")

    async def cleanup_orphaned_containers(
        self, failed_retention_hours: int = 24, success_retention_hours: int = 2
    ) -> int:
        """
        Clean up old orchestrator containers.

        Per specification:
        - Failed instances retained for 24 hours
        - Successful instances retained for 2 hours

        Args:
            failed_retention_hours: Keep failed containers for this long
            success_retention_hours: Keep successful containers for this long

        Returns:
            Number of containers removed
        """
        try:
            loop = asyncio.get_event_loop()
            containers = await loop.run_in_executor(
                None,
                lambda: self.client.containers.list(
                    all=True,  # Include stopped containers
                    filters={"label": "orchestrator=true"},
                ),
            )

            removed_count = 0
            current_time = datetime.now(timezone.utc)

            for container in containers:
                # Parse creation time from label
                try:
                    created_str = container.labels.get("created_at", "")
                    if not created_str:
                        continue

                    created_time = datetime.fromisoformat(created_str)
                    age_hours = (current_time - created_time).total_seconds() / 3600

                    # Check container status to determine retention period
                    # Read status from host file
                    status = "unknown"
                    try:
                        status_file = Path(
                            f"/tmp/orchestrator_status/{container.id[:12]}.status"
                        )
                        if status_file.exists():
                            status = status_file.read_text().strip()
                            # Clean up old status file
                            status_file.unlink()
                    except (OSError, IOError):
                        # If we can't read status, use default
                        pass

                    if status == "failed":
                        max_age = failed_retention_hours
                    elif status == "success":
                        max_age = success_retention_hours
                    else:
                        # Default to failed retention for unknown status (safer)
                        max_age = failed_retention_hours

                    if age_hours > max_age:
                        await self.cleanup_container(container)
                        removed_count += 1
                        logger.debug(
                            f"Removed {status} container {container.name} (age: {age_hours:.1f}h)"
                        )

                except (
                    docker.errors.APIError,
                    docker.errors.DockerException,
                    OSError,
                ) as e:
                    logger.error(f"Error checking container {container.name}: {e}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} orphaned containers")

            return removed_count

        except (docker.errors.APIError, docker.errors.DockerException) as e:
            logger.error(f"Failed to cleanup orphaned containers: {e}")
            return 0

    async def update_container_status(self, container: Container, status: str) -> None:
        """
        Track container completion status for cleanup differentiation.

        Since Docker doesn't allow updating labels after creation and containers
        will be stopped, we'll store the status in a file on the host.

        Args:
            container: Container to update
            status: New status (success/failed)
        """
        try:
            # Store status in a host file indexed by container ID
            status_dir = Path("/tmp/orchestrator_status")
            status_dir.mkdir(exist_ok=True)

            status_file = status_dir / f"{container.id[:12]}.status"
            status_file.write_text(f"{status}\n")

            logger.debug(f"Recorded container {container.name} status as: {status}")

        except (OSError, IOError) as e:
            logger.warning(f"Could not record container status: {e}")
