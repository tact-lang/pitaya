"""
Main orchestration component that coordinates multiple AI coding instances.

This is the central component that manages strategies, parallel execution,
events, and state. It depends only on the Instance Runner's public API.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ..instance_runner import run_instance
from ..shared import (
    AuthConfig,
    ContainerLimits,
    InstanceResult,
    RetryConfig,
)

from .event_bus import EventBus
from .state import InstanceInfo, RunState, StateManager
from ..shared import InstanceStatus
from .strategies import AVAILABLE_STRATEGIES
from .strategy_context import StrategyContext
from ..exceptions import (
    OrchestratorError,
    DockerError,
    GitError,
    StrategyError,
    ValidationError,
)

if TYPE_CHECKING:
    from .http_server import OrchestratorHTTPServer

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates multiple AI coding instances according to strategies.

    Core responsibilities:
    - Execute strategies that spawn and coordinate instances
    - Manage parallel execution with resource limits
    - Own the event bus for component communication
    - Track current state of all instances and strategies
    - Generate container and branch names with proper conventions
    - Handle instance failures according to strategy logic
    """

    def __init__(
        self,
        max_parallel_instances: int = 20,
        state_dir: Path = Path("./orchestrator_state"),
        logs_dir: Path = Path("./logs"),
        container_limits: Optional[ContainerLimits] = None,
        retry_config: Optional[RetryConfig] = None,
        auth_config: Optional[AuthConfig] = None,
        snapshot_interval: int = 30,
        event_buffer_size: int = 10000,
        container_retention_failed_hours: int = 24,
        container_retention_success_hours: int = 2,
    ):
        """
        Initialize orchestrator.

        Args:
            max_parallel_instances: Maximum instances running concurrently
            state_dir: Directory for state persistence
            logs_dir: Directory for logs and events
            container_limits: Resource limits for containers
            retry_config: Retry configuration for instances
            auth_config: Authentication configuration for AI tools
        """
        self.max_parallel_instances = max_parallel_instances
        self.state_dir = state_dir
        self.logs_dir = logs_dir
        self.container_limits = container_limits or ContainerLimits()
        self.retry_config = retry_config or RetryConfig()
        self.auth_config = auth_config
        self.snapshot_interval = snapshot_interval
        self.event_buffer_size = event_buffer_size
        self.container_retention_failed_hours = container_retention_failed_hours
        self.container_retention_success_hours = container_retention_success_hours

        # Log auth config for debugging
        if self.auth_config:
            logger.info(
                f"Orchestrator initialized with auth_config: oauth_token={'***' if self.auth_config.oauth_token else None}, api_key={'***' if self.auth_config.api_key else None}"
            )
        else:
            logger.warning("Orchestrator initialized with NO auth_config")

        # Core components
        self.event_bus: Optional[EventBus] = None
        self.state_manager: Optional[StateManager] = None

        # Execution tracking
        self._active_instances: Set[str] = set()
        self._instance_queue: asyncio.Queue = asyncio.Queue()
        self._instance_futures: Dict[str, asyncio.Future] = {}
        self._shutdown = False

        # Resource pool semaphore
        self._resource_pool = asyncio.Semaphore(max_parallel_instances)

        # Background tasks
        self._executor_tasks: List[asyncio.Task] = []
        # Map instance_id -> container_id[:12] for last-active tracking
        self._inst_container_id: Dict[str, str] = {}

        # HTTP server (optional)
        self.http_server: Optional["OrchestratorHTTPServer"] = None
        self._initialized: bool = False
        self._force_import: bool = False  # set per-run from strategy_config

    async def initialize(self) -> None:
        """Initialize orchestrator components."""
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Check disk space (20GB minimum as per spec)
        await self._check_disk_space()

        # Initialize components
        self.state_manager = StateManager(
            self.state_dir, event_bus=None, snapshot_interval=self.snapshot_interval
        )  # Will set event_bus later

        # Initialize a default event bus for early subscriptions
        if not self.event_bus:
            self.event_bus = EventBus(max_events=self.event_buffer_size)

        # Clean up orphaned containers from previous runs
        await self.cleanup_orphaned_containers()

        # Adjust parallelism adaptively if beneficial
        try:
            import os
            host_cpu = max(1, os.cpu_count() or 1)
            per_container = max(1, int(self.container_limits.cpu_count))
            adaptive = max(2, min(20, host_cpu // per_container))
            if self.max_parallel_instances == 20 and adaptive != 20:
                logger.info(f"Adaptive parallelism: host_cpu={host_cpu}, per_container={per_container} -> setting max_parallel_instances={adaptive}")
                self.max_parallel_instances = adaptive
                self._resource_pool = asyncio.Semaphore(self.max_parallel_instances)
        except Exception as e:
            logger.debug(f"Adaptive parallelism calc failed: {e}")

        # Start multiple background executors for true parallel execution
        # Start a reasonable number of executors (min of max_parallel and 10)
        num_executors = min(self.max_parallel_instances, 10)
        for i in range(num_executors):
            task = asyncio.create_task(self._instance_executor())
            self._executor_tasks.append(task)

        logger.info(
            f"Orchestrator initialized with {self.max_parallel_instances} max parallel instances and {num_executors} executor tasks"
        )
        self._initialized = True

    async def start_http_server(self, port: int) -> None:
        """Start HTTP server for multi-UI support."""
        from .http_server import OrchestratorHTTPServer

        self.http_server = OrchestratorHTTPServer(self, port)
        # Run in separate thread to match spec wording
        self.http_server.start_threaded()
        logger.info(f"HTTP server started (threaded) on port {port}")

    async def shutdown(self) -> None:
        """Shutdown orchestrator cleanly."""
        logger.info("Shutting down orchestrator")

        # Signal shutdown
        self._shutdown = True

        # Mark any running instances as interrupted before cancelling tasks
        if self.state_manager and self.state_manager.current_state:
            try:
                for instance in list(self.state_manager.current_state.instances.values()):
                    if instance.state == InstanceStatus.RUNNING:
                        self.state_manager.update_instance_state(
                            instance_id=instance.instance_id,
                            state=InstanceStatus.INTERRUPTED,
                        )
            except Exception as e:
                logger.warning(f"Failed to mark running instances as interrupted: {e}")

        # Cancel all executor tasks
        for task in self._executor_tasks:
            task.cancel()

        # Wait for all tasks to complete
        if self._executor_tasks:
            await asyncio.gather(*self._executor_tasks, return_exceptions=True)

        # Stop all running containers in parallel for fast shutdown
        if self.state_manager and self.state_manager.current_state:
            running_instances = [
                instance
                for instance in self.state_manager.current_state.instances.values()
                if instance.state in (InstanceStatus.RUNNING, InstanceStatus.INTERRUPTED)
            ]

            if running_instances:
                logger.info(
                    f"Stopping {len(running_instances)} running containers in parallel..."
                )

                # Import here to avoid circular dependency
                from ..instance_runner.docker_manager import DockerManager

                async def stop_container_for_instance(instance_info: InstanceInfo):
                    """Stop container for a single instance."""
                    try:
                        docker_mgr = DockerManager()
                        container = docker_mgr.get_container(
                            instance_info.container_name
                        )
                        if container:
                            await docker_mgr.stop_container(container)
                            logger.info(
                                f"Stopped container {instance_info.container_name}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to stop container {instance_info.container_name}: {e}"
                        )

                # Stop all containers in parallel
                await asyncio.gather(
                    *[stop_container_for_instance(inst) for inst in running_instances],
                    return_exceptions=True,
                )

                logger.info("All running containers stopped")

        # Stop HTTP server if running
        if self.http_server:
            try:
                self.http_server.stop_threaded()
            except Exception:
                pass

        # Close event bus
        if self.event_bus:
            self.event_bus.close()

        # Save final state
        if self.state_manager:
            await self.state_manager.save_snapshot()

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
        """
        Execute a strategy with given configuration.

        This is the main entry point for running strategies. It handles
        all the complexity of parallel execution and result aggregation.

        Args:
            strategy_name: Name of strategy to execute
            prompt: Instruction for AI agents
            repo_path: Repository to work on
            base_branch: Starting branch
            strategy_config: Strategy-specific configuration
            run_id: Optional run ID (auto-generated if not provided)
            runs: Number of parallel strategy executions (default: 1)

        Returns:
            List of final results from all strategy executions
        """
        logger.info(
            f"run_strategy called: strategy={strategy_name}, prompt={prompt[:50]}..."
        )

        # Generate run ID if not provided
        if not run_id:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"

        # Initialize event bus for this run (if not already initialized)
        if not self.event_bus:
            event_log_path = self.logs_dir / run_id / "events.jsonl"
            self.event_bus = EventBus(max_events=self.event_buffer_size, persist_path=event_log_path, run_id=run_id)
        else:
            # If event bus exists, update persistence path
            event_log_path = self.logs_dir / run_id / "events.jsonl"
            event_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.event_bus._persist_file:
                self.event_bus.close()
            self.event_bus.persist_path = event_log_path
            self.event_bus._open_persist_file()

        # Set event bus on state manager
        self.state_manager.event_bus = self.event_bus

        # Initialize state
        state = self.state_manager.initialize_run(
            run_id=run_id,
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
        )

        # Per-run force_import setting from strategy_config
        self._force_import = bool((strategy_config or {}).get("force_import", False))

        # Start periodic snapshots
        await self.state_manager.start_periodic_snapshots()

        # Emit run started event
        self.event_bus.emit("run.started", {"run_id": run_id, "strategy": strategy_name, "prompt": prompt, "repo_path": str(repo_path), "base_branch": base_branch})

        try:
            # Get strategy class
            if strategy_name not in AVAILABLE_STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            logger.info(f"Found strategy class for {strategy_name}")
            strategy_class = AVAILABLE_STRATEGIES[strategy_name]

            # Execute multiple strategy runs in parallel
            logger.info(f"Runs requested: {runs}")
            if runs > 1:
                # Run multiple strategies in parallel
                logger.info("Executing multiple strategy runs in parallel")
                strategy_tasks = []
                for run_idx in range(runs):
                    strategy = strategy_class()
                    strategy.set_config_overrides(strategy_config or {})
                    config = strategy.create_config()

                    # Register strategy execution
                    strategy_id = str(uuid.uuid4())
                    self.state_manager.register_strategy(
                        strategy_id=strategy_id,
                        strategy_name=strategy_name,
                        config=strategy_config or {},
                    )

                    # Execute strategy
                    self.event_bus.emit("strategy.started", {"strategy_id": strategy_id, "strategy_name": strategy_name, "config": strategy_config, "run_index": run_idx + 1, "total_runs": runs})
                    # Canonical
                    self.event_bus.emit_canonical(
                        type="strategy.started",
                        run_id=run_id,
                        strategy_execution_id=strategy_id,
                        payload={"name": strategy_name, "params": strategy_config or {}},
                    )

                    # Create async task for this strategy execution
                    async def execute_single_strategy(strat, strat_id, cfg):
                        try:
                            # Create strategy context
                            ctx = StrategyContext(
                                orchestrator=self,
                                strategy_name=strat.name,
                                strategy_execution_id=strat_id,
                            )
                            # Store repo_path for context to use
                            self.repo_path = repo_path

                            res = await strat.execute(
                                prompt=prompt,
                                base_branch=base_branch,
                                ctx=ctx,
                            )

                            # Determine strategy state based on results per spec semantics
                            state_value = "completed"
                            if res and all((not r.success) for r in res):
                                state_value = "failed"
                            self.state_manager.update_strategy_state(
                                strategy_id=strat_id,
                                state=state_value,
                                results=res,
                            )

                            self.event_bus.emit("strategy.completed", {"strategy_id": strat_id, "result_count": len(res), "branch_names": [r.branch_name for r in res if r.branch_name]})
                            self.event_bus.emit_canonical(
                                type="strategy.completed",
                                run_id=run_id,
                                strategy_execution_id=strat_id,
                                payload={"status": "success"},
                            )

                            return res
                        except (StrategyError, OrchestratorError) as e:
                            self.state_manager.update_strategy_state(
                                strategy_id=strat_id,
                                state="failed",
                            )
                            self.event_bus.emit(
                                "strategy.failed",
                                {
                                    "strategy_id": strat_id,
                                    "error": str(e),
                                },
                            )
                            raise

                    task = asyncio.create_task(
                        execute_single_strategy(strategy, strategy_id, config)
                    )
                    strategy_tasks.append(task)

                # Wait for all strategies to complete
                all_results = await asyncio.gather(
                    *strategy_tasks, return_exceptions=True
                )

                # Flatten results, handling exceptions
                final_results = []
                for result in all_results:
                    if isinstance(result, Exception):
                        logger.error(f"Strategy execution failed: {result}")
                    elif isinstance(result, list):
                        final_results.extend(result)

                # Save results to disk
                await self._save_results(run_id, final_results)

                return final_results

            else:
                # Single strategy execution (original code)
                logger.info("Executing single strategy run")
                strategy = strategy_class()
                strategy.set_config_overrides(strategy_config or {})
                config = strategy.create_config()

                # Register strategy execution
                strategy_id = str(uuid.uuid4())
                self.state_manager.register_strategy(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    config=strategy_config or {},
                )

                # Execute strategy
                logger.info(f"Emitting strategy.started event for {strategy_id}")
                self.event_bus.emit("strategy.started", {"strategy_id": strategy_id, "strategy_name": strategy_name, "config": strategy_config})
                self.event_bus.emit_canonical(
                    type="strategy.started",
                    run_id=run_id,
                    strategy_execution_id=strategy_id,
                    payload={"name": strategy_name, "params": strategy_config or {}},
                )

                logger.info(f"Calling strategy.execute for {strategy_name}")
                # Create strategy context
                ctx = StrategyContext(
                    orchestrator=self,
                    strategy_name=strategy.name,
                    strategy_execution_id=strategy_id,
                )
                # Store repo_path for context to use
                self.repo_path = repo_path

                results = await strategy.execute(
                    prompt=prompt,
                    base_branch=base_branch,
                    ctx=ctx,
                )

                # Determine strategy state based on results per spec semantics
                state_value = "completed"
                if results and all((not r.success) for r in results):
                    state_value = "failed"
                self.state_manager.update_strategy_state(
                    strategy_id=strategy_id,
                    state=state_value,
                    results=results,
                )

                self.event_bus.emit("strategy.completed", {"strategy_id": strategy_id, "result_count": len(results), "branch_names": [r.branch_name for r in results if r.branch_name]})
                self.event_bus.emit_canonical(
                    type="strategy.completed",
                    run_id=run_id,
                    strategy_execution_id=strategy_id,
                    payload={"status": "success"},
                )

                # Save results to disk
                await self._save_results(run_id, results)

                return results

        except (StrategyError, OrchestratorError, asyncio.CancelledError) as e:
            logger.exception(f"Strategy execution failed: {e}")
            self.event_bus.emit(
                "run.failed",
                {
                    "run_id": run_id,
                    "error": str(e),
                },
            )
            raise

        finally:
            # Stop periodic snapshots
            await self.state_manager.stop_periodic_snapshots()

            # Mark run complete
            if state:
                state.completed_at = datetime.now(timezone.utc)
                await self.state_manager.save_snapshot()

            self.event_bus.emit(
                "run.completed",
                {
                    "run_id": run_id,
                    "duration_seconds": (
                        (state.completed_at - state.started_at).total_seconds()
                        if state
                        else 0
                    ),
                    "total_cost": state.total_cost if state else 0,
                    "total_tokens": state.total_tokens if state else 0,
                },
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
        """
        Spawn a new instance (called by strategies).

        This method handles all the complexity of naming, queueing,
        and tracking instances. Strategies just call this and wait.

        Args:
            prompt: Instruction for the instance
            repo_path: Repository to work on
            base_branch: Starting branch
            strategy_name: Name of calling strategy
            strategy_execution_id: ID of the strategy execution
            instance_index: Index of this instance within the strategy
            metadata: Strategy-specific metadata

        Returns:
            Instance ID for tracking
        """
        logger.info(
            f"spawn_instance called: strategy={strategy_name}, prompt={prompt[:30]}..."
        )

        # Check disk space before starting (spec: validate before instances)
        try:
            await self._check_disk_space()
        except Exception:
            # _check_disk_space already logs validation errors; re-raise to stop spawn
            raise

        # Generate instance ID
        if key:
            import hashlib
            instance_id = hashlib.sha256(f"{self.state_manager.current_state.run_id}|{strategy_execution_id}|{key}".encode("utf-8")).hexdigest()[:8]
        else:
            instance_id = str(uuid.uuid4())[:8]

        # Get strategy execution index from state
        strategy_exec = self.state_manager.current_state.strategies.get(
            strategy_execution_id
        )
        if strategy_exec:
            # Find the strategy index by counting previous strategies
            sidx = (
                len(
                    [
                        s
                        for s in self.state_manager.current_state.strategies.values()
                        if s.started_at < strategy_exec.started_at
                    ]
                )
                + 1
            )
        else:
            sidx = 1

        # Generate names according to spec
        # Extract timestamp from run_id (format: "run_YYYYMMDD_HHMMSS")
        run_timestamp = self.state_manager.current_state.run_id.replace("run_", "")
        if key:
            import hashlib
            khash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
            container_name = f"orchestrator_{run_timestamp}_s{sidx}_k{khash}"
            branch_name = f"{strategy_name}_{run_timestamp}_k{khash}"
        else:
            container_name = f"orchestrator_{run_timestamp}_s{sidx}_i{instance_index}"
            branch_name = f"{strategy_name}_{run_timestamp}_{sidx}_{instance_index}"

        # Register instance only if this durable key wasn't seen before
        if instance_id not in self.state_manager.current_state.instances:
            self.state_manager.register_instance(
                instance_id=instance_id,
                strategy_name=strategy_name,
                prompt=prompt,
                base_branch=base_branch,
                branch_name=branch_name,
                container_name=container_name,
                metadata=metadata,
            )

        # Add instance to strategy
        if (
            strategy_execution_id
            and strategy_execution_id in self.state_manager.current_state.strategies
        ):
            inst_list = self.state_manager.current_state.strategies[strategy_execution_id].instance_ids
            if instance_id not in inst_list:
                inst_list.append(instance_id)

        # Only enqueue if not already enqueued/known
        if instance_id not in self._instance_futures:
            future = asyncio.Future()
            self._instance_futures[instance_id] = future
            logger.info(f"Queueing instance {instance_id} for execution")
            await self._instance_queue.put(instance_id)
            self.event_bus.emit("instance.queued", {"instance_id": instance_id, "strategy": strategy_name, "branch_name": branch_name}, instance_id=instance_id)
            # Canonical task.scheduled already emitted in StrategyContext.run; no-op for spawn_instance without key

        logger.info(f"Instance {instance_id} queued successfully")
        return instance_id

    async def wait_for_instances(
        self,
        instance_ids: List[str],
    ) -> Dict[str, InstanceResult]:
        """
        Wait for multiple instances to complete.

        Args:
            instance_ids: List of instance IDs to wait for

        Returns:
            Dictionary mapping instance ID to result
        """
        # Wait for all futures
        futures = [self._instance_futures[id] for id in instance_ids]
        results = await asyncio.gather(*futures)

        # Map results
        return {
            instance_id: result for instance_id, result in zip(instance_ids, results)
        }

    async def _instance_executor(self) -> None:
        """
        Background task that executes queued instances.

        This implements the parallel execution engine with resource
        pool management and FIFO scheduling.
        """
        logger.info("Instance executor started")

        while not self._shutdown:
            try:
                # Get next instance from queue (with timeout to check shutdown)
                try:
                    logger.debug("Waiting for instance from queue...")
                    instance_id = await asyncio.wait_for(
                        self._instance_queue.get(), timeout=1.0
                    )
                    logger.info(f"Got instance {instance_id} from queue")
                except asyncio.TimeoutError:
                    continue

                # Acquire resource slot
                async with self._resource_pool:
                    if self._shutdown:
                        break

                    # Mark as active
                    self._active_instances.add(instance_id)

                    # Execute instance
                    try:
                        await self._execute_instance(instance_id)
                    finally:
                        self._active_instances.remove(instance_id)

            except OrchestratorError as e:
                logger.exception(f"Error in instance executor: {e}")
            except asyncio.CancelledError:
                # Expected during shutdown or task cancellation; do not log as error
                break

    async def _execute_instance(self, instance_id: str) -> None:
        """Execute a single instance."""
        info = self.state_manager.current_state.instances.get(instance_id)
        if not info:
            logger.error(f"Instance {instance_id} not found in state")
            return

        # Update state to running
        self.state_manager.update_instance_state(
            instance_id=instance_id,
            state=InstanceStatus.RUNNING,
        )

        # Emit instance.started event for TUI and canonical mapping when key available
        self.event_bus.emit("instance.started", {"strategy": info.strategy_name, "prompt": info.prompt, "model": info.metadata.get("model", "sonnet"), "branch_name": info.branch_name}, instance_id=instance_id)
        task_key = (info.metadata or {}).get("key")
        if task_key:
            self.event_bus.emit_canonical(
                type="task.started",
                run_id=self.state_manager.current_state.run_id,
                strategy_execution_id=next((sid for sid, strat in self.state_manager.current_state.strategies.items() if instance_id in strat.instance_ids), None),
                key=task_key,
                payload={"key": task_key, "instance_id": instance_id, "container_name": info.container_name, "model": info.metadata.get("model", "sonnet")},
            )

        # Create event callback (forward and capture session_id for resume)
        def event_callback(event: Dict[str, Any]) -> None:
            data = event.get("data", {})
            sid = data.get("session_id")
            if sid:
                try:
                    self.state_manager.update_instance_session_id(instance_id, sid)
                except Exception:
                    pass
            # Track container id for last-active updates
            try:
                if event.get("type") == "instance.container_created":
                    cid = data.get("container_id")
                    if cid:
                        self._inst_container_id[instance_id] = str(cid)
            except Exception:
                pass
            # Update last-active file for cleanup logic
            try:
                cid = self._inst_container_id.get(instance_id)
                if cid:
                    from pathlib import Path as _P
                    p = _P(f"/tmp/orchestrator_status/{cid}.active")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
            # Forward runner-level events; persist for real-time UI until canonical task.* stream is used
            self.event_bus.emit(
                event_type=event.get("type", "instance.event"),
                data=data,
                instance_id=instance_id,
                persist=True,
            )

        # Start a lightweight heartbeat that reports recent activity for this instance
        heartbeat_task: Optional[asyncio.Task] = None
        try:
            if self.event_bus:
                heartbeat_task = asyncio.create_task(self._heartbeat_monitor(instance_id))
            # Extract strategy execution ID from instance metadata
            strategy_execution_id = None
            for sid, strat in self.state_manager.current_state.strategies.items():
                if instance_id in strat.instance_ids:
                    strategy_execution_id = sid
                    break

            # Run the instance
            result = await run_instance(
                prompt=info.prompt,
                repo_path=self.state_manager.current_state.repo_path,
                base_branch=info.base_branch,
                branch_name=info.branch_name,
                run_id=self.state_manager.current_state.run_id,
                strategy_execution_id=strategy_execution_id,
                instance_id=instance_id,
                task_key=task_key,
                container_name=info.container_name,
                model=info.metadata.get("model", "sonnet"),
                session_id=(info.session_id or (info.metadata or {}).get("resume_session_id")),
                event_callback=event_callback,
                timeout_seconds=3600,  # Default 1 hour, could be made configurable
                container_limits=self.container_limits,
                auth_config=self.auth_config,
                retry_config=self.retry_config,
                import_policy=(info.metadata or {}).get("import_policy", "auto"),
                import_conflict_policy=(info.metadata or {}).get("import_conflict_policy", "fail"),
                skip_empty_import=bool((info.metadata or {}).get("skip_empty_import", True)),
                network_egress=(info.metadata or {}).get("network_egress"),
                max_turns=(info.metadata or {}).get("max_turns"),
            )

            # Populate strategy-specific metadata on the result per spec
            try:
                if result and hasattr(result, "metadata"):
                    result.metadata = result.metadata or {}
                    result.metadata.update(info.metadata or {})
                    if strategy_execution_id:
                        result.metadata["strategy_execution_id"] = strategy_execution_id
            except Exception:
                pass

            # Update state with result (treat canceled as interrupted)
            if result.success:
                new_state = InstanceStatus.COMPLETED
            else:
                if getattr(result, "status", None) == "canceled" or getattr(result, "error_type", None) == "canceled":
                    new_state = InstanceStatus.INTERRUPTED
                else:
                    new_state = InstanceStatus.FAILED

            self.state_manager.update_instance_state(
                instance_id=instance_id,
                state=new_state,
                result=result,
            )

            # Emit canonical terminal events when key is present
            if task_key:
                if result.success:
                    artifact = {
                        "type": "branch",
                        "branch_planned": info.branch_name,
                        "branch_final": result.branch_name,
                        "base": info.base_branch,
                        "has_changes": result.has_changes,
                    }
                    self.event_bus.emit_canonical(
                        type="task.completed",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=strategy_execution_id,
                        key=task_key,
                        payload={"key": task_key, "instance_id": instance_id, "artifact": artifact, "metrics": result.metrics or {}, "final_message": result.final_message or ""},
                    )
                elif new_state == InstanceStatus.INTERRUPTED:
                    self.event_bus.emit_canonical(
                        type="task.interrupted",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=strategy_execution_id,
                        key=task_key,
                        payload={"key": task_key, "instance_id": instance_id},
                    )
                else:
                    self.event_bus.emit_canonical(
                        type="task.failed",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=strategy_execution_id,
                        key=task_key,
                        payload={"key": task_key, "instance_id": instance_id, "error_type": result.error_type or "unknown", "message": result.error or ""},
                    )

            # Resolve future
            self._instance_futures[instance_id].set_result(result)

        except asyncio.CancelledError:
            # Treat orchestrator-initiated cancellation as interruption, not failure
            logger.info(f"Instance {instance_id} canceled by shutdown; marking interrupted")
            interrupt_result = InstanceResult(
                success=False,
                error="canceled",
                error_type="canceled",
                status="canceled",
            )
            self.state_manager.update_instance_state(
                instance_id=instance_id,
                state=InstanceStatus.INTERRUPTED,
                result=interrupt_result,
            )
            self._instance_futures[instance_id].set_result(interrupt_result)
        except (DockerError, GitError, OrchestratorError) as e:
            logger.exception(f"Instance {instance_id} execution failed: {e}")

            # Create error result
            error_result = InstanceResult(
                success=False,
                error=str(e),
                error_type="orchestration",
            )

            # Update state
            self.state_manager.update_instance_state(
                instance_id=instance_id,
                state=InstanceStatus.FAILED,
                result=error_result,
            )

            # Resolve future with error
            self._instance_futures[instance_id].set_result(error_result)
        except Exception as e:
            # Catch-all to ensure the system does not hang on unexpected errors
            logger.exception(f"Instance {instance_id} crashed with unexpected error: {e}")

            error_result = InstanceResult(
                success=False,
                error=str(e),
                error_type="unexpected",
            )

            # Update state
            self.state_manager.update_instance_state(
                instance_id=instance_id,
                state=InstanceStatus.FAILED,
                result=error_result,
            )

            # Resolve future with error
            self._instance_futures[instance_id].set_result(error_result)
        finally:
            # Stop heartbeat
            if heartbeat_task:
                heartbeat_task.cancel()

    async def _heartbeat_monitor(self, instance_id: str, interval: float = 2.0) -> None:
        """Emit periodic debug heartbeats with the last event type/time for the instance.

        Helps identify stalls by showing how long since the last event was observed.
        """
        try:
            while True:
                last_type = None
                last_ts = None
                if self.event_bus:
                    try:
                        # Scan recent events in reverse for this instance
                        for ev in reversed(self.event_bus.events):
                            if ev.get("instance_id") == instance_id:
                                last_type = ev.get("type")
                                last_ts = ev.get("timestamp")
                                break
                    except Exception:
                        pass

                if last_type and last_ts:
                    logger.debug(
                        f"Heartbeat: instance {instance_id} last_event={last_type} at {last_ts}"
                    )
                else:
                    logger.debug(f"Heartbeat: instance {instance_id} awaiting first event")

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    def subscribe(
        self,
        event_type: Optional[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> Callable[[], None]:
        """
        Subscribe to orchestration events.

        Args:
            event_type: Event type to subscribe to (None for all)
            callback: Function to call when event occurs

        Returns:
            Unsubscribe function
        """
        if not self.event_bus:
            raise RuntimeError("Event bus not initialized")

        return self.event_bus.subscribe(event_type, callback)

    def get_current_state(self) -> Optional[RunState]:
        """Get current run state snapshot."""
        return self.state_manager.get_current_state() if self.state_manager else None

    async def get_events_since(
        self,
        offset: int = 0,
        limit: int = 1000,
        run_id: Optional[str] = None,
        event_types: Optional[Set[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get events since a given offset.

        Args:
            offset: Byte offset to start from
            limit: Maximum number of events to return
            run_id: Filter to specific run (not implemented yet)
            event_types: Filter to specific event types

        Returns:
            Tuple of (events list, next offset)
        """
        if not self.event_bus:
            return [], 0

        # If run_id specified, point event bus to that run's events file
        if run_id:
            try:
                event_log_path = self.logs_dir / run_id / "events.jsonl"
                if self.event_bus.persist_path != event_log_path:
                    # Close existing file and open new
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

    async def cleanup_orphaned_containers(self) -> None:
        """Clean up orphaned containers from previous runs."""
        try:
            from ..instance_runner.docker_manager import DockerManager

            docker_manager = DockerManager()

            # Clean up containers with differentiated retention times
            cleaned = await docker_manager.cleanup_orphaned_containers(
                failed_retention_hours=self.container_retention_failed_hours,
                success_retention_hours=self.container_retention_success_hours,
            )

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} orphaned containers")

        except (DockerError, OSError) as e:
            logger.warning(f"Failed to cleanup orphaned containers: {e}")
            # Don't fail initialization if cleanup fails

    async def resume_run(
        self, run_id: str, force_fresh: bool = False
    ) -> List[InstanceResult]:
        """
        Resume an interrupted run by recovering state and continuing execution.

        Per spec section 7.4:
        - Load saved state from snapshot
        - Identify interrupted instances
        - Check container existence
        - Resume based on plugin capabilities
        - Continue strategy execution

        Args:
            run_id: The run ID to resume
            force_fresh: If True, re-run incomplete instances with fresh containers

        Returns:
            List of instance results from resumed execution
        """
        logger.info(f"Resuming run {run_id}")

        # Initialize if not already done
        if not self._initialized:
            await self.initialize()

        # Load saved state and recover from events
        # Ensure event bus is configured to this run's events file
        event_log_path = self.logs_dir / run_id / "events.jsonl"
        if not self.event_bus:
            self.event_bus = EventBus(max_events=self.event_buffer_size, persist_path=event_log_path)
        else:
            # Repoint persist path to this run
            self.event_bus.persist_path = event_log_path
            self.event_bus._open_persist_file()
        self.state_manager.event_bus = self.event_bus

        saved_state = await self.state_manager.load_and_recover_state(run_id)
        if not saved_state:
            raise ValueError(f"No saved state found for run {run_id}")

        # Build tasks according to resume policy
        from ..shared import InstanceStatus as _IS
        from ..instance_runner.plugins import AVAILABLE_PLUGINS
        plugin = AVAILABLE_PLUGINS["claude-code"]()
        resume_tasks: list[asyncio.Task] = []
        cannot_resume_count = 0

        # Docker client to check container existence
        from ..instance_runner.docker_manager import DockerManager
        docker_manager = DockerManager()

        for iid, info in saved_state.instances.items():
            if info.state in (_IS.RUNNING, _IS.INTERRUPTED):
                if force_fresh:
                    resume_tasks.append(
                        asyncio.create_task(self._run_instance_from_saved_state(info))
                    )
                else:
                    can_resume = False
                    try:
                        if info.session_id and plugin.capabilities.supports_resume:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: docker_manager.client.containers.get(info.container_name),
                            )
                            can_resume = True
                    except Exception:
                        can_resume = False

                    if can_resume:
                        async def do_resume(inst) -> InstanceResult:
                            return await run_instance(
                                prompt=inst.prompt,
                                repo_path=saved_state.repo_path,
                                base_branch=inst.base_branch,
                                branch_name=inst.branch_name,
                                run_id=saved_state.run_id,
                                strategy_execution_id=None,
                                instance_id=inst.instance_id,
                                container_name=inst.container_name,
                                model=inst.metadata.get("model", "sonnet"),
                                session_id=inst.session_id,
                                event_callback=lambda e: self.event_bus.emit(e.get("type", "instance.event"), e.get("data", {}), instance_id=inst.instance_id),
                                timeout_seconds=3600,
                                container_limits=self.container_limits,
                                auth_config=self.auth_config,
                                reuse_container=True,
                                finalize=True,
                                import_policy=(inst.metadata or {}).get("import_policy", "auto"),
                                import_conflict_policy=(inst.metadata or {}).get("import_conflict_policy", "fail"),
                                skip_empty_import=bool((inst.metadata or {}).get("skip_empty_import", True)),
                            )
                        resume_tasks.append(asyncio.create_task(do_resume(info)))
                    else:
                        self.state_manager.update_instance_state(
                            instance_id=info.instance_id,
                            state=_IS.FAILED,
                            result=InstanceResult(
                                success=False,
                                error="cannot_resume",
                                error_type="cannot_resume",
                                branch_name=info.branch_name,
                                status="failed",
                            ),
                        )
                        cannot_resume_count += 1

        # Wait for any re-run tasks
        resumed_results = await asyncio.gather(*resume_tasks, return_exceptions=True) if resume_tasks else []
        all_results = [r for r in resumed_results if r and not isinstance(r, Exception)]

        # Finalize state and save results
        self.state_manager.current_state.completed_at = datetime.now(timezone.utc)
        await self.state_manager.save_snapshot()
        await self._save_results(run_id, all_results)

        self.event_bus.emit(
            "run.completed",
            {"run_id": run_id, "resumed": True, "cannot_resume": cannot_resume_count, "total_results": len(all_results)},
        )

        return all_results

    async def _run_instance_from_saved_state(self, instance_info) -> InstanceResult:
        """
        Re-run an instance from saved state with a fresh container.

        Args:
            instance_info: Saved instance information

        Returns:
            InstanceResult from the fresh run
        """
        # Extract the necessary information from saved state
        # Note: This assumes the saved state has the required fields

        # Use the instance runner to re-run with fresh container
        from ..instance_runner import run_instance

        try:
            result = await run_instance(
                prompt=instance_info.prompt,
                repo_path=self.state_manager.current_state.repo_path,
                base_branch=instance_info.base_branch,
                branch_name=instance_info.branch_name,
                instance_id=instance_info.instance_id,
                run_id=self.state_manager.current_state.run_id,
                strategy_execution_id=None,
                event_callback=lambda e: self.event_bus.emit(
                    e.get("type", "instance.event"), e.get("data", {}), instance_id=instance_info.instance_id
                ),
                container_name=instance_info.container_name,
                container_limits=self.container_limits,
                retry_config=self.retry_config,
                auth_config=self.auth_config,
                import_policy=(instance_info.metadata or {}).get("import_policy", "auto"),
                import_conflict_policy=(instance_info.metadata or {}).get("import_conflict_policy", "fail"),
                skip_empty_import=bool((instance_info.metadata or {}).get("skip_empty_import", True)),
            )

            # Update state
            self.state_manager.update_instance_state(
                instance_id=instance_info.instance_id, state=_IS.COMPLETED, result=result
            )

            return result

        except (DockerError, GitError, OrchestratorError) as e:
            logger.error(f"Failed to re-run instance {instance_info.instance_id}: {e}")

            # Mark as failed
            self.state_manager.update_instance_state(
                instance_id=instance_info.instance_id, state=_IS.FAILED, result=InstanceResult(success=False, error=str(e), error_type="rerun_failed", branch_name=instance_info.branch_name, status="failed")
            )

            # Return a failed result
            return InstanceResult(
                success=False,
                error=str(e),
                error_type="rerun_failed",
                branch_name=instance_info.branch_name,
                status="failed",
            )

    async def _check_disk_space(self) -> None:
        """
        Check available disk space per spec section 8.1.
        Requires at least 20GB free space for Docker images and temporary workspaces.
        """
        try:
            import shutil

            # Get disk usage statistics for the current directory
            stat = shutil.disk_usage(Path.cwd())

            # Convert to GB
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = (stat.used / stat.total) * 100

            logger.info(
                f"Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB ({used_percent:.1f}% used)"
            )

            # Check minimum requirement (20GB as per spec)
            if free_gb < 20:
                raise ValidationError(
                    f"Insufficient disk space: {free_gb:.1f}GB free, need at least 20GB "
                    f"for Docker images and temporary workspaces"
                )

        except (OSError, ImportError) as e:
            logger.warning(f"Could not check disk space: {e}")
            # Don't fail if we can't check - let operations fail naturally

    async def _save_results(self, run_id: str, results: List[InstanceResult]) -> None:
        """
        Save run results to disk per spec section 7.5.

        Creates:
        - ./results/run_*/summary.json - Machine-readable run summary
        - ./results/run_*/branches.txt - Simple list of branch names
        - ./results/run_*/metrics.csv - Time-series metrics (per spec)
        - ./results/run_*/instance_metrics.csv - Instance-level snapshot metrics
        """
        try:
            # Create results directory
            results_dir = Path("./results") / run_id
            results_dir.mkdir(parents=True, exist_ok=True)

            # Get current state
            state = self.state_manager.current_state
            if not state:
                logger.warning(
                    f"No state found for run {run_id}, skipping results save"
                )
                return

            # Prepare summary data
            summary_data = {
                "run_id": run_id,
                "status": "completed",
                "started_at": state.started_at.isoformat(),
                "completed_at": (
                    state.completed_at.isoformat() if state.completed_at else None
                ),
                "duration_seconds": (
                    (state.completed_at - state.started_at).total_seconds()
                    if state.completed_at
                    else None
                ),
                "prompt": state.prompt,
                "repo_path": str(state.repo_path),
                "base_branch": state.base_branch,
                "total_instances": state.total_instances,
                "completed_instances": state.completed_instances,
                "failed_instances": state.failed_instances,
                "total_cost": state.total_cost,
                "total_tokens": state.total_tokens,
                "strategies": {},
                "results": [],
            }

            # Add strategy information
            for strat_id, strat_info in state.strategies.items():
                summary_data["strategies"][strat_id] = {
                    "name": strat_info.strategy_name,
                    "state": strat_info.state,
                    "config": strat_info.config,
                    "started_at": strat_info.started_at.isoformat(),
                    "completed_at": (
                        strat_info.completed_at.isoformat()
                        if strat_info.completed_at
                        else None
                    ),
                    "result_count": (
                        len(strat_info.results) if strat_info.results else 0
                    ),
                }

            # Add results information
            branches = []

            # Build a reverse lookup from result to instance_id
            result_to_instance = {}
            for instance_id, info in state.instances.items():
                if info.result:
                    result_to_instance[id(info.result)] = instance_id

            for result in results:
                # Extract commit statistics
                commit_stats = result.commit_statistics or {}

                # Get instance_id from state mapping
                instance_id = result_to_instance.get(id(result), "unknown")

                result_data = {
                    "instance_id": instance_id,
                    "session_id": result.session_id,
                    "branch_name": result.branch_name,
                    "status": result.status,
                    "success": result.success,
                    "error": result.error,
                    "has_changes": result.has_changes,
                    "duration_seconds": result.duration_seconds,
                    "cost": (
                        result.metrics.get("total_cost", 0.0) if result.metrics else 0.0
                    ),
                    "tokens": (
                        result.metrics.get("total_tokens", 0) if result.metrics else 0
                    ),
                    "input_tokens": (
                        result.metrics.get("input_tokens", 0) if result.metrics else 0
                    ),
                    "output_tokens": (
                        result.metrics.get("output_tokens", 0) if result.metrics else 0
                    ),
                    # Compatibility keys per spec naming
                    "cost_usd": (
                        result.metrics.get("total_cost", 0.0) if result.metrics else 0.0
                    ),
                    "tokens_in": (
                        result.metrics.get("input_tokens", 0) if result.metrics else 0
                    ),
                    "tokens_out": (
                        result.metrics.get("output_tokens", 0) if result.metrics else 0
                    ),
                    "commit_count": commit_stats.get("commit_count", 0),
                    "lines_added": commit_stats.get("insertions", 0),
                    "lines_deleted": commit_stats.get("deletions", 0),
                    "container_name": result.container_name,
                    "started_at": result.started_at,
                    "completed_at": result.completed_at,
                }
                summary_data["results"].append(result_data)

                if result.branch_name:
                    branches.append(result.branch_name)

            # Save summary.json
            summary_path = results_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2, default=str)

            # Save branches.txt
            if branches:
                branches_path = results_dir / "branches.txt"
                with open(branches_path, "w") as f:
                    f.write("\n".join(branches) + "\n")

            # Save per-instance snapshot metrics as instance_metrics.csv
            metrics_path = results_dir / "instance_metrics.csv"
            with open(metrics_path, "w") as f:
                # Write header
                f.write("instance_id,branch_name,status,duration_seconds,cost,tokens,input_tokens,output_tokens,commit_count,lines_added,lines_deleted,has_changes\n")

                # Write data rows
                for result in results:
                    # Extract commit statistics
                    commit_stats = result.commit_statistics or {}

                    # Get instance_id from state mapping
                    instance_id = result_to_instance.get(id(result), "unknown")

                    row = [
                        (
                            instance_id[:8] if instance_id != "unknown" else "unknown"
                        ),  # Shortened ID
                        result.branch_name or "",
                        result.status,
                        (
                            f"{result.duration_seconds:.1f}"
                            if result.duration_seconds
                            else "0"
                        ),
                        (
                            f"{result.metrics.get('total_cost', 0):.2f}"
                            if result.metrics
                            else "0.00"
                        ),
                        (
                            str(result.metrics.get("total_tokens", 0))
                            if result.metrics
                            else "0"
                        ),
                        (
                            str(result.metrics.get("input_tokens", 0))
                            if result.metrics
                            else "0"
                        ),
                        (
                            str(result.metrics.get("output_tokens", 0))
                            if result.metrics
                            else "0"
                        ),
                        str(commit_stats.get("commit_count", 0)),
                        str(commit_stats.get("insertions", 0)),
                        str(commit_stats.get("deletions", 0)),
                        "yes" if result.has_changes else "no",
                    ]
                    f.write(",".join(row) + "\n")

            # Generate time-series metrics.csv per spec from events log
            try:
                events_file = self.logs_dir / run_id / "events.jsonl"
                ts_path = results_dir / "metrics.csv"
                running: set[str] = set()
                completed_set: set[str] = set()
                failed_set: set[str] = set()
                inst_tokens: Dict[str, int] = {}
                inst_cost: Dict[str, float] = {}
                if events_file.exists():
                    with open(events_file, "r") as ef, open(ts_path, "w") as tf:
                        tf.write("timestamp,active_instances,completed_instances,failed_instances,total_cost,total_tokens,event_type\n")
                        for line in ef:
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            et = ev.get("type", "")
                            iid = ev.get("instance_id")
                            data = ev.get("data", {})
                            ts = ev.get("timestamp", "")
                            if et == "instance.started" and iid:
                                running.add(iid)
                            elif et == "instance.completed" and iid:
                                running.discard(iid)
                                completed_set.add(iid)
                            elif et == "instance.failed" and iid:
                                running.discard(iid)
                                failed_set.add(iid)
                            elif et == "instance.claude_turn_complete" and iid:
                                tm = data.get("turn_metrics", {})
                                inst_tokens[iid] = inst_tokens.get(iid, 0) + int(tm.get("tokens", 0) or 0)
                                inst_cost[iid] = inst_cost.get(iid, 0.0) + float(tm.get("cost", 0.0) or 0.0)
                            elif et == "instance.claude_completed" and iid:
                                m = data.get("metrics", {})
                                if m:
                                    inst_tokens[iid] = int(m.get("total_tokens", inst_tokens.get(iid, 0)))
                                    inst_cost[iid] = float(m.get("total_cost", inst_cost.get(iid, 0.0)))
                            total_cost = sum(inst_cost.values())
                            total_tokens = sum(inst_tokens.values())
                            tf.write(f"{ts},{len(running)},{len(completed_set)},{len(failed_set)},{total_cost:.4f},{total_tokens},{et}\n")
            except Exception as e:
                logger.debug(f"Failed generating time-series metrics: {e}")

            # Create strategy output directory
            strategy_dir = results_dir / "strategy_output"
            strategy_dir.mkdir(exist_ok=True)

            # Save strategy-specific outputs
            for strat_id, strat_info in state.strategies.items():
                if strat_info.results:
                    # Save strategy-specific data like scores and feedback
                    strategy_file = (
                        strategy_dir / f"{strat_info.strategy_name}_{strat_id}.json"
                    )
                    strategy_data = {
                        "strategy_id": strat_id,
                        "strategy_name": strat_info.strategy_name,
                        "config": strat_info.config,
                        "results": [],
                    }

                    for result in strat_info.results:
                        if result and hasattr(result, "metrics") and result.metrics:
                            # Include metrics like score, feedback for scoring strategies
                            result_entry = {
                                "branch_name": result.branch_name,
                                "success": result.success,
                                "metrics": result.metrics,
                            }
                            strategy_data["results"].append(result_entry)

                    with open(strategy_file, "w") as f:
                        json.dump(strategy_data, f, indent=2)

                    # Best-of-N helpers: best_branch.txt and scores.json
                    try:
                        if strat_info.strategy_name == "best-of-n" and strat_info.results:
                            selected_branch = None
                            scores: Dict[str, float] = {}
                            for res in strat_info.results:
                                # res may be InstanceResult or dict depending on context
                                branch = getattr(res, "branch_name", None) or (
                                    res.get("branch_name") if isinstance(res, dict) else None
                                )
                                metrics = getattr(res, "metrics", None) or (
                                    res.get("metrics") if isinstance(res, dict) else {}
                                )
                                if metrics:
                                    if metrics.get("selected") and branch:
                                        selected_branch = branch
                                    if branch and metrics.get("score") is not None:
                                        try:
                                            scores[branch] = float(metrics.get("score"))
                                        except Exception:
                                            pass
                            if selected_branch:
                                with open(strategy_dir / "best_branch.txt", "w") as bf:
                                    bf.write(f"{selected_branch}\n")
                            if scores:
                                with open(strategy_dir / "scores.json", "w") as sf:
                                    json.dump(scores, sf, indent=2)
                    except Exception:
                        pass

            # Optional Markdown export (summary-focused)
            try:
                md_path = results_dir / "summary.md"
                with open(md_path, "w") as f:
                    f.write(f"# Orchestrator Run Summary: {run_id}\n\n")
                    f.write(f"Prompt: {state.prompt}\n\n")
                    f.write(f"Repository: {state.repo_path}\n\n")
                    f.write(
                        f"Total Instances: {state.total_instances} | Completed: {state.completed_instances} | Failed: {state.failed_instances}\n\n"
                    )
                    f.write(
                        f"Total Cost: ${state.total_cost:.2f} | Total Tokens: {state.total_tokens}\n\n"
                    )
                    if branches:
                        f.write("## Branches\n\n")
                        for b in branches:
                            f.write(f"- {b}\n")
                        f.write("\n")
                    if results:
                        f.write("## Instances\n\n")
                        for r in results:
                            status = "✅" if r.success else "❌"
                            dur = (
                                f"{r.duration_seconds:.0f}s" if r.duration_seconds else "-"
                            )
                            cost = r.metrics.get("total_cost", 0.0) if r.metrics else 0.0
                            tokens = r.metrics.get("total_tokens", 0) if r.metrics else 0
                            f.write(
                                f"- {status} {r.branch_name or 'no-branch'} • {dur} • ${cost:.2f} • {tokens} tokens\n"
                            )
            except (OSError, IOError, ValueError):
                # Non-fatal if markdown export fails
                pass

            logger.info(f"Saved results to {results_dir}")

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to save results: {e}")
            # Don't fail the run if results can't be saved
