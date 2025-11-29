"""Main orchestration component coordinating Pitaya runs."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..shared import (
    AuthConfig,
    ContainerLimits,
    InstanceResult,
    RetryConfig,
    InstanceStatus,
)
from .event_bus import EventBus
from .instance_manager import InstanceManager
from .resume_manager import resume_run as resume_run_helper
from .results_writer import save_results as save_results_helper
from .state import InstanceInfo, RunState, StateManager
from .strategy_runner import run_strategy as run_strategy_helper

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates multiple AI coding instances according to strategies."""

    def __init__(
        self,
        max_parallel_instances: Optional[int] = None,
        max_parallel_startup: Optional[int] = None,
        state_dir: Path = Path("./pitaya_state"),
        logs_dir: Path = Path("./logs"),
        container_limits: Optional[ContainerLimits] = None,
        retry_config: Optional[RetryConfig] = None,
        auth_config: Optional[AuthConfig] = None,
        snapshot_interval: int = 30,
        event_buffer_size: int = 10000,
        runner_timeout_seconds: int = 3600,
        default_network_egress: str = "online",
        branch_namespace: str = "flat",
        allow_overwrite_protected_refs: bool = False,
        allow_global_session_volume: bool = False,
        default_plugin_name: str = "claude-code",
        default_model_alias: str = "sonnet",
        default_docker_image: Optional[str] = None,
        default_agent_cli_args: Optional[List[str]] = None,
        force_commit: bool = False,
        randomize_queue_order: bool = False,
        explicit_max_parallel: bool = False,
        default_workspace_include_branches: Optional[List[str]] = None,
    ):
        self.max_parallel_instances: Optional[int] = max_parallel_instances
        self.max_parallel_startup: Optional[int] = max_parallel_startup
        self.state_dir = state_dir
        self.logs_dir = logs_dir
        self.container_limits = container_limits or ContainerLimits()
        self.retry_config = retry_config or RetryConfig()
        self.auth_config = auth_config
        self.snapshot_interval = snapshot_interval
        self.event_buffer_size = event_buffer_size
        self.runner_timeout_seconds = max(1, int(runner_timeout_seconds))
        self.default_network_egress = str(default_network_egress or "online").lower()
        self.default_docker_image = default_docker_image
        try:
            self.default_agent_cli_args: List[str] = list(default_agent_cli_args or [])
        except Exception:
            self.default_agent_cli_args = []
        self.force_commit = bool(force_commit)
        self._explicit_max_parallel = bool(explicit_max_parallel)
        self.branch_namespace = str(branch_namespace or "flat").lower()
        self.allow_overwrite_protected_refs = bool(allow_overwrite_protected_refs)
        self.allow_global_session_volume = bool(allow_global_session_volume)
        self.default_plugin_name = str(default_plugin_name or "claude-code")
        self.default_model_alias = str(default_model_alias or "sonnet")
        try:
            self.default_workspace_include_branches: Optional[List[str]] = (
                list(default_workspace_include_branches)
                if default_workspace_include_branches
                else None
            )
        except Exception:
            self.default_workspace_include_branches = None

        self._pending_redaction_patterns: Optional[List[str]] = None
        self._shutdown = False
        self._initialized = False
        self._force_import = False
        self._randomize_queue_order = bool(randomize_queue_order)
        self.randomize_queue_order = self._randomize_queue_order

        self.event_bus: Optional[EventBus] = None
        self.state_manager: Optional[StateManager] = None
        self.instance_manager = InstanceManager(
            self, randomize_queue_order=self._randomize_queue_order
        )
        self.repo_path: Optional[Path] = None

        if self.auth_config:
            logger.info("Pitaya initialized with auth_config (secrets redacted)")
        else:
            logger.warning("Pitaya initialized with NO auth_config")

    def set_pending_redaction_patterns(self, patterns: Optional[List[str]]) -> None:
        """Store regex patterns to redact when the event bus is created."""
        try:
            self._pending_redaction_patterns = list(patterns or [])
        except Exception:
            self._pending_redaction_patterns = []

    async def initialize(self) -> None:
        """Initialize state manager, event bus, and instance executors."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        await self._check_disk_space()

        self.state_manager = StateManager(
            self.state_dir, event_bus=None, snapshot_interval=self.snapshot_interval
        )
        if not self.event_bus:
            self.event_bus = EventBus(max_events=self.event_buffer_size)

        if self.max_parallel_instances is None or int(self.max_parallel_instances) <= 0:
            try:
                import os as _os

                cpu = int(_os.cpu_count() or 2)
            except Exception:
                cpu = 2
            self.max_parallel_instances = max(1, cpu)
        if self.max_parallel_startup is None or int(self.max_parallel_startup) <= 0:
            self.max_parallel_startup = max(
                1, min(10, int(self.max_parallel_instances or 1))
            )

        await self.instance_manager.initialize(
            int(self.max_parallel_instances or 1),
            int(self.max_parallel_startup or self.max_parallel_instances or 1),
        )
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown Pitaya cleanly."""
        logger.info("Shutting down Pitaya")
        self._shutdown = True

        if self.state_manager and self.state_manager.current_state:
            try:
                for instance in list(
                    self.state_manager.current_state.instances.values()
                ):
                    if instance.state in (
                        InstanceStatus.RUNNING,
                        InstanceStatus.QUEUED,
                    ):
                        self.state_manager.update_instance_state(
                            instance_id=instance.instance_id,
                            state=InstanceStatus.INTERRUPTED,
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to mark running instances as interrupted: %s", exc
                )

        await self.instance_manager.shutdown()

        if self.state_manager and self.state_manager.current_state:
            running_instances = [
                inst
                for inst in self.state_manager.current_state.instances.values()
                if inst.state in (InstanceStatus.RUNNING, InstanceStatus.INTERRUPTED)
            ]
            if running_instances:
                logger.info(
                    "Stopping %s running containers in parallel...",
                    len(running_instances),
                )
                from ..instance_runner.docker_manager import DockerManager

                async def _stop(inst: InstanceInfo) -> None:
                    try:
                        docker_mgr = DockerManager()
                        container = docker_mgr.get_container(inst.container_name)
                        if container:
                            await docker_mgr.stop_container(container, timeout=1)
                            logger.info("Stopped container %s", inst.container_name)
                    except Exception as exc:
                        logger.warning(
                            "Failed to stop container %s: %s", inst.container_name, exc
                        )

                await asyncio.gather(
                    *[_stop(inst) for inst in running_instances], return_exceptions=True
                )

        if self.event_bus:
            try:
                self.event_bus.flush_pending()
            except Exception:
                pass
            self.event_bus.close()

        if self.state_manager:
            await self.state_manager.save_snapshot()
            try:
                await self.state_manager.stop_periodic_snapshots()
            except Exception:
                pass

    async def run_strategy(
        self,
        strategy_name: str,
        prompt: str,
        repo_path: Path,
        base_branch: str = "main",
        strategy_config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        runs: int = 1,
    ) -> List[InstanceResult]:
        return await run_strategy_helper(
            self,
            strategy_name=strategy_name,
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
            strategy_config=strategy_config,
            run_id=run_id,
            runs=runs,
        )

    async def spawn_instance(
        self,
        *,
        prompt: str,
        repo_path: Path,
        base_branch: str,
        strategy_name: str,
        strategy_execution_id: str,
        instance_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> str:
        return await self.instance_manager.spawn_instance(
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
            strategy_name=strategy_name,
            strategy_execution_id=strategy_execution_id,
            instance_index=instance_index,
            metadata=metadata,
            key=key,
        )

    async def wait_for_instances(
        self,
        instance_ids: List[str],
    ) -> Dict[str, InstanceResult]:
        return await self.instance_manager.wait_for_instances(instance_ids)

    def subscribe(
        self,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> Callable[[], None]:
        if not self.event_bus:
            raise RuntimeError("Event bus not initialized")
        return self.event_bus.subscribe(event_type, callback)

    def get_current_state(self) -> Optional[RunState]:
        return self.state_manager.get_current_state() if self.state_manager else None

    async def get_events_since(
        self,
        offset: int = 0,
        limit: int = 1000,
        run_id: Optional[str] = None,
        event_types: Optional[Set[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not self.event_bus:
            return [], 0
        if run_id:
            try:
                event_log_path = self.logs_dir / run_id / "events.jsonl"
                if self.event_bus.persist_path != event_log_path:
                    try:
                        self.event_bus.close()
                    except Exception:
                        pass
                    self.event_bus.persist_path = event_log_path
                    self.event_bus._open_persist_file()
            except Exception:
                pass
        return self.event_bus.get_events_since(
            offset=offset, limit=limit, event_types=event_types, timestamp=timestamp
        )

    async def resume_run(self, run_id: str) -> List[InstanceResult]:
        return await resume_run_helper(self, run_id)

    async def save_results(self, run_id: str, results: List[InstanceResult]) -> None:
        await save_results_helper(self, run_id, results)

    async def _check_disk_space(self) -> None:
        """Log available disk space (informational only)."""
        try:
            import shutil

            stat = shutil.disk_usage(Path.cwd())
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = (stat.used / stat.total) * 100
            logger.info(
                "Disk space: %.1fGB free of %.1fGB (%.1f%% used)",
                free_gb,
                total_gb,
                used_percent,
            )
        except (OSError, ImportError) as exc:
            logger.warning("Could not check disk space: %s", exc)
