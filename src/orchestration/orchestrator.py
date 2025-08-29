"""
Main orchestration component that coordinates multiple AI coding agents.

Centers Pitaya's strategy engine: manages pluggable/custom strategies,
parallel execution, events, and state. Depends only on the Instance Runner's
public API.
"""

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
from .strategies.loader import load_strategy
from .strategy_context import StrategyContext
from ..exceptions import (
    OrchestratorError,
    DockerError,
    GitError,
    StrategyError,
)


logger = logging.getLogger(__name__)


class RandomAsyncQueue:
    """Random-pick async queue with a minimal asyncio.Queue-like interface."""

    def __init__(self) -> None:
        self._items: List[Any] = []
        self._cv: asyncio.Condition = asyncio.Condition()

    async def put(self, item: Any) -> None:
        async with self._cv:
            self._items.append(item)
            self._cv.notify(1)

    async def get(self) -> Any:
        async with self._cv:
            while not self._items:
                await self._cv.wait()
            idx = random.randrange(len(self._items))
            return self._items.pop(idx)

    def qsize(self) -> int:
        return len(self._items)


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
        # Optional global Docker image override
        self.default_docker_image = default_docker_image
        # Optional default agent CLI passthrough args
        try:
            self.default_agent_cli_args: List[str] = list(default_agent_cli_args or [])
        except Exception:
            self.default_agent_cli_args = []
        # Runner behavior: force a commit if workspace has changes
        self.force_commit: bool = bool(force_commit)
        # Whether operator explicitly set max_parallel (override guards/policies)
        self._explicit_max_parallel = bool(explicit_max_parallel)
        # models.yaml mapping removed; no checksum handshake

        # Log auth config for debugging
        if self.auth_config:
            logger.info(
                f"Pitaya initialized with auth_config: oauth_token={'***' if self.auth_config.oauth_token else None}, api_key={'***' if self.auth_config.api_key else None}"
            )
        else:
            logger.warning("Pitaya initialized with NO auth_config")

        # Core components
        self.event_bus: Optional[EventBus] = None
        self.state_manager: Optional[StateManager] = None

        # Execution tracking
        self._active_instances: Set[str] = set()
        self._randomize_queue_order = bool(randomize_queue_order)
        self._instance_queue = (
            RandomAsyncQueue() if self._randomize_queue_order else asyncio.Queue()
        )
        self._instance_futures: Dict[str, asyncio.Future] = {}
        self._shutdown = False

        # Resource pool semaphore (initialized after adaptive calc in initialize())
        self._resource_pool: asyncio.Semaphore = asyncio.Semaphore(1)
        # Separate startup semaphore to throttle expensive workspace prep
        self._startup_pool: asyncio.Semaphore = asyncio.Semaphore(1)
        # Branch namespace strategy: 'flat' (default) or 'hierarchical'
        self.branch_namespace = str(branch_namespace or "flat").lower()
        self.allow_overwrite_protected_refs = bool(allow_overwrite_protected_refs)
        self.allow_global_session_volume = bool(allow_global_session_volume)
        # Default plugin/model for strategy tasks (strategy-agnostic selection)
        self.default_plugin_name = str(default_plugin_name or "claude-code")
        self.default_model_alias = str(default_model_alias or "sonnet")

        # Multi-resource admission (CPU, memory, disk guard)
        self._admission_lock = asyncio.Lock()
        self._admission_cv = asyncio.Condition(self._admission_lock)
        self._cpu_in_use = 0
        self._mem_in_use_gb = 0
        try:
            import os as _os

            self._host_cpu = max(1, _os.cpu_count() or 1)
        except Exception:
            self._host_cpu = 1
        try:
            # Total physical memory in GB (best-effort)
            import os as _os

            if hasattr(_os, "sysconf"):
                pages = _os.sysconf("SC_PHYS_PAGES") if hasattr(_os, "sysconf") else 0
                page_size = (
                    _os.sysconf("SC_PAGE_SIZE") if hasattr(_os, "sysconf") else 4096
                )
                total_bytes = int(pages) * int(page_size)
            else:
                total_bytes = 0
            self._host_mem_gb = max(
                1, int(total_bytes / (1024**3)) if total_bytes else 8
            )
        except Exception:
            self._host_mem_gb = 8
        # Fixed sensible defaults (no env overrides)
        self._mem_guard_pct = 0.8
        self._disk_min_free_gb = 10
        self._pack_max_slope_mib_per_min = 256

        # Background tasks
        self._executor_tasks: List[asyncio.Task] = []

        # Server extension support removed
        self._initialized: bool = False
        self._force_import: bool = False  # set per-run from strategy_config

    async def initialize(self) -> None:
        """Initialize orchestrator components."""
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Disk space check removed: keep informational logging only
        await self._check_disk_space()

        # Initialize components
        self.state_manager = StateManager(
            self.state_dir, event_bus=None, snapshot_interval=self.snapshot_interval
        )  # Will set event_bus later

        # Initialize a default event bus for early subscriptions
        if not self.event_bus:
            self.event_bus = EventBus(max_events=self.event_buffer_size)

        # Orphaned container cleanup removed; containers are deleted immediately on success/failure

        # Resolve parallelism defaults: CPU-based for total; startup=min(10, total)
        if self.max_parallel_instances is None or int(self.max_parallel_instances) <= 0:
            try:
                import os as _os

                cpu = int(_os.cpu_count() or 2)
            except Exception:
                cpu = 2
            self.max_parallel_instances = max(1, cpu)
            logger.info(
                f"Parallelism(default): max_parallel_instances={self.max_parallel_instances} (cpu-based)"
            )
        if self.max_parallel_startup is None or int(self.max_parallel_startup) <= 0:
            self.max_parallel_startup = max(
                1, min(10, int(self.max_parallel_instances or 1))
            )
            logger.info(
                f"Parallelism(default): max_parallel_startup={self.max_parallel_startup} (min(10,total))"
            )

        # Initialize resource pool semaphore now that parallelism is resolved
        self._resource_pool = asyncio.Semaphore(int(self.max_parallel_instances or 1))
        # Initialize startup pool (caps concurrent workspace preparations)
        self._startup_pool = asyncio.Semaphore(
            int(self.max_parallel_startup or self.max_parallel_instances or 1)
        )

        # Start multiple background executors equal to max_parallel_instances
        num_executors = int(self.max_parallel_instances or 1)
        for i in range(num_executors):
            task = asyncio.create_task(self._instance_executor())
            self._executor_tasks.append(task)

        logger.info(
            f"Pitaya initialized with max_parallel_instances={self.max_parallel_instances} "
            f"max_parallel_startup={self.max_parallel_startup} and {num_executors} executor tasks"
        )
        self._initialized = True

    # Server extension support removed

    async def shutdown(self) -> None:
        """Shutdown Pitaya cleanly."""
        logger.info("Shutting down Pitaya")

        # Signal shutdown
        self._shutdown = True

        # Mark any running or queued instances as interrupted before cancelling tasks
        interrupted_ids: list[str] = []
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
                        interrupted_ids.append(instance.instance_id)
                        # Ensure any pending future is resolved to unblock waiters
                        try:
                            fut = self._instance_futures.get(instance.instance_id)
                            if fut and not fut.done():
                                interrupt_result = InstanceResult(
                                    success=False,
                                    error="canceled",
                                    error_type="canceled",
                                    status="canceled",
                                )
                                fut.set_result(interrupt_result)
                        except Exception:
                            pass
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
                if instance.state
                in (InstanceStatus.RUNNING, InstanceStatus.INTERRUPTED)
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
                            # Use a short timeout to avoid the 10s default wait on PID1 SIGTERM
                            # Fixed fast stop timeout
                            tmo = 1
                            await docker_mgr.stop_container(
                                container, timeout=max(0, tmo)
                            )
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

        # Server extension support removed

        # Note: canonical task.interrupted is emitted by StateManager.update_instance_state

        # Per spec: emit strategy.completed {status: "canceled", reason: "operator_interrupt"}
        # for each strategy execution that has not yet produced any successful
        # child task, and flush pending events before exit.
        try:
            if (
                self.state_manager
                and self.state_manager.current_state
                and self.event_bus
            ):
                for sid, strat in list(
                    self.state_manager.current_state.strategies.items()
                ):
                    # Skip strategies already terminal in state
                    if getattr(strat, "state", None) in ("completed", "failed"):
                        continue
                    # Determine if any child instance succeeded
                    any_success = False
                    try:
                        for iid in strat.instance_ids or []:
                            info = self.state_manager.current_state.instances.get(iid)
                            if not info:
                                continue
                            if getattr(info, "result", None) and getattr(
                                info.result, "success", False
                            ):
                                any_success = True
                                break
                    except Exception:
                        any_success = False
                    if not any_success:
                        # Update strategy state to a terminal state for snapshot purposes
                        try:
                            self.state_manager.update_strategy_state(
                                strategy_id=sid, state="failed"
                            )
                        except Exception:
                            pass
                        # Emit canonical completion with canceled status and operator reason
                        try:
                            self.event_bus.emit_canonical(
                                type="strategy.completed",
                                run_id=self.state_manager.current_state.run_id,
                                strategy_execution_id=sid,
                                payload={
                                    "status": "canceled",
                                    "reason": "operator_interrupt",
                                },
                            )
                        except Exception:
                            pass
                # Synchronously flush pending canonical events before exiting
                try:
                    self.event_bus.flush_pending()
                except Exception:
                    pass
        except Exception:
            pass

        # Close event bus
        if self.event_bus:
            self.event_bus.close()

        # Save final state
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
            import uuid as _uuid

            short8 = _uuid.uuid4().hex[:8]
            run_id = f"run_{timestamp}_{short8}"

        # Initialize event bus for this run (if not already initialized)
        if not self.event_bus:
            event_log_path = self.logs_dir / run_id / "events.jsonl"
            self.event_bus = EventBus(
                max_events=self.event_buffer_size,
                persist_path=event_log_path,
                run_id=run_id,
            )
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
        self.event_bus.emit(
            "run.started",
            {
                "run_id": run_id,
                "strategy": strategy_name,
                "prompt": prompt,
                "repo_path": str(repo_path),
                "base_branch": base_branch,
            },
        )

        try:
            # Resolve strategy class: built-in name or file.py[:Class]
            effective_strategy_name = strategy_name
            if strategy_name in AVAILABLE_STRATEGIES:
                strategy_class = AVAILABLE_STRATEGIES[strategy_name]
            else:
                # Support file.py[:Class] and module.path[:Class] forms
                strategy_class = load_strategy(strategy_name)
                try:
                    effective_strategy_name = strategy_class().name
                except Exception:
                    effective_strategy_name = strategy_name

            logger.info(f"Resolved strategy class for {effective_strategy_name}")

            # Removed hardcoded capability gating by strategy name; selection is plugin-agnostic

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
                        strategy_name=getattr(
                            strategy, "name", effective_strategy_name
                        ),
                        config=strategy_config or {},
                    )

                    # Execute strategy
                    self.event_bus.emit(
                        "strategy.started",
                        {
                            "strategy_id": strategy_id,
                            "strategy_name": getattr(
                                strategy, "name", effective_strategy_name
                            ),
                            "config": strategy_config,
                            "run_index": run_idx + 1,
                            "total_runs": runs,
                        },
                    )
                    # Canonical
                    self.event_bus.emit_canonical(
                        type="strategy.started",
                        run_id=run_id,
                        strategy_execution_id=strategy_id,
                        payload={
                            "name": getattr(strategy, "name", effective_strategy_name),
                            "params": strategy_config or {},
                        },
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

                            self.event_bus.emit(
                                "strategy.completed",
                                {
                                    "strategy_id": strat_id,
                                    "result_count": len(res),
                                    "branch_names": [
                                        r.branch_name for r in res if r.branch_name
                                    ],
                                },
                            )
                            # Canonical completion with normative status semantics
                            any_success = any(
                                getattr(r, "success", False) for r in (res or [])
                            )
                            # On operator shutdown with no success, classify as canceled
                            if not any_success and self._shutdown:
                                status = "canceled"
                                payload = {
                                    "status": status,
                                    "reason": "operator_interrupt",
                                }
                            else:
                                status = "success" if any_success else "failed"
                                payload = {"status": status}
                                if status == "failed":
                                    payload["reason"] = "no_successful_tasks"
                            self.event_bus.emit_canonical(
                                type="strategy.completed",
                                run_id=run_id,
                                strategy_execution_id=strat_id,
                                payload=payload,
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
                # Single strategy execution
                logger.info("Executing single strategy run")
                strategy = strategy_class()
                strategy.set_config_overrides(strategy_config or {})
                config = strategy.create_config()

                # Register strategy execution
                strategy_id = str(uuid.uuid4())
                self.state_manager.register_strategy(
                    strategy_id=strategy_id,
                    strategy_name=getattr(strategy, "name", effective_strategy_name),
                    config=strategy_config or {},
                )

                # Execute strategy
                logger.info(f"Emitting strategy.started event for {strategy_id}")
                self.event_bus.emit(
                    "strategy.started",
                    {
                        "strategy_id": strategy_id,
                        "strategy_name": getattr(
                            strategy, "name", effective_strategy_name
                        ),
                        "config": strategy_config,
                    },
                )
                self.event_bus.emit_canonical(
                    type="strategy.started",
                    run_id=run_id,
                    strategy_execution_id=strategy_id,
                    payload={
                        "name": getattr(strategy, "name", effective_strategy_name),
                        "params": strategy_config or {},
                    },
                )

                logger.info(
                    f"Calling strategy.execute for {getattr(strategy, 'name', effective_strategy_name)}"
                )
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

                self.event_bus.emit(
                    "strategy.completed",
                    {
                        "strategy_id": strategy_id,
                        "result_count": len(results),
                        "branch_names": [
                            r.branch_name for r in results if r.branch_name
                        ],
                    },
                )
                any_success = any(getattr(r, "success", False) for r in (results or []))
                if not any_success and self._shutdown:
                    status = "canceled"
                    payload = {"status": status, "reason": "operator_interrupt"}
                else:
                    status = "success" if any_success else "failed"
                    payload = {"status": status}
                    if status == "failed":
                        payload["reason"] = "no_successful_tasks"
                self.event_bus.emit_canonical(
                    type="strategy.completed",
                    run_id=run_id,
                    strategy_execution_id=strategy_id,
                    payload=payload,
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
            f"spawn_instance: strategy={strategy_name} sid={strategy_execution_id} key={(metadata or {}).get('key','-')}"
        )

        # Check disk space before starting (spec: validate before instances)
        try:
            await self._check_disk_space()
        except Exception:
            # _check_disk_space already logs validation errors; re-raise to stop spawn
            raise

        # Generate instance ID (normative stable 16-hex over JCS-like canonical JSON)
        import hashlib
        import json as _json

        def _drop_nulls(o):
            if isinstance(o, dict):
                return {k: _drop_nulls(v) for k, v in o.items() if v is not None}
            elif isinstance(o, list):
                return [_drop_nulls(v) for v in o if v is not None]
            return o

        def _short16_from(obj: dict) -> str:
            enc = _json.dumps(
                _drop_nulls(obj),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            return hashlib.sha256(enc.encode("utf-8")).hexdigest()[:16]

        if key:
            instance_id = _short16_from(
                {
                    "run_id": self.state_manager.current_state.run_id,
                    "strategy_execution_id": strategy_execution_id,
                    "key": key,
                }
            )
        else:
            synthetic = f"i:{instance_index}"
            instance_id = _short16_from(
                {
                    "run_id": self.state_manager.current_state.run_id,
                    "strategy_execution_id": strategy_execution_id,
                    "key": synthetic,
                }
            )

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

        # Generate names according to spec using full run_id (run_YYYYMMDD_HHMMSS_<short8>)
        run_id_val = self.state_manager.current_state.run_id
        # Strategy segment sanitization: allow only [A-Za-z0-9._-], strip others; fallback to 'unknown'
        try:
            import re as _re

            _san = _re.compile(r"[^A-Za-z0-9._-]+")
            _strategy_segment = _san.sub("-", str(strategy_name or "").strip())
            _strategy_segment = _strategy_segment.strip("-/._") or "unknown"
        except Exception:
            _strategy_segment = str(strategy_name or "unknown")
        # Use durable key when provided, else derive from instance_id for stability
        durable_key = key or str(instance_id)
        # Namespace durable key by strategy execution to avoid collisions across parallel runs
        khash = hashlib.sha256(
            f"{strategy_execution_id}|{durable_key}".encode("utf-8")
        ).hexdigest()[:8]
        container_name = f"pitaya_{run_id_val}_s{sidx}_k{khash}"
        # Enforce hierarchical namespace per spec with pitaya namespace
        branch_name = f"pitaya/{_strategy_segment}/{run_id_val}/k{khash}"
        logger.debug(
            f"spawn_instance: iid={instance_id} container={container_name} branch={branch_name}"
        )

        # Defensive: clamp branch name length to <=200 chars and validate
        try:
            if len(branch_name) > 200:
                # Prefer trimming the middle run_id segment if needed
                head = f"pitaya/{_strategy_segment}/"
                tail = branch_name.split("/")[-1]
                room = 200 - (len(head) + 1 + len(tail))
                if room > 8:  # keep at least some of run_id
                    short_run = run_id_val[:room]
                    branch_name = f"{head}{short_run}/{tail}"
                else:
                    # Last resort: fallback strategy segment
                    branch_name = f"pitaya/unknown/{tail}"[-200:]
        except Exception:
            pass

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
            logger.debug(f"spawn_instance: registered iid={instance_id}")

        # Add instance to strategy
        if (
            strategy_execution_id
            and strategy_execution_id in self.state_manager.current_state.strategies
        ):
            inst_list = self.state_manager.current_state.strategies[
                strategy_execution_id
            ].instance_ids
            if instance_id not in inst_list:
                inst_list.append(instance_id)

        # Ensure a future exists for this instance and enqueue if needed
        info = self.state_manager.current_state.instances.get(instance_id)
        if info is None:
            # Should not happen: registered above
            info = InstanceInfo(
                instance_id=instance_id,
                strategy_name=strategy_name,
                prompt=prompt,
                base_branch=base_branch,
                branch_name=branch_name,
                container_name=container_name,
                state=InstanceStatus.QUEUED,
                metadata=metadata or {},
            )
            self.state_manager.current_state.instances[instance_id] = info

        if instance_id not in self._instance_futures:
            # If the instance is already terminal, create an already-completed future
            if info.state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
                fut = asyncio.Future()
                # Synthesize minimal result if missing (e.g., on resume after reload)
                if not info.result:
                    from ..shared import InstanceResult as _IR

                    info.result = _IR(
                        success=(info.state == InstanceStatus.COMPLETED),
                        branch_name=info.branch_name,
                        has_changes=False,
                        metrics={},
                        session_id=info.session_id,
                        status=(
                            "success"
                            if info.state == InstanceStatus.COMPLETED
                            else "failed"
                        ),
                    )
                fut.set_result(info.result)
                self._instance_futures[instance_id] = fut
                logger.debug(
                    f"spawn_instance: iid={instance_id} already terminal ({info.state.value}); returning completed future"
                )
                # Emit canonical terminal event only if missing, to avoid duplicates on resume
                try:
                    task_key = (info.metadata or {}).get("key")
                    if task_key and self.event_bus:
                        # Detect if a terminal canonical event for this instance is already present
                        seen_terminal = False
                        try:
                            term_types = {
                                "task.completed",
                                "task.failed",
                                "task.interrupted",
                            }
                            events, _ = self.event_bus.get_events_since(
                                offset=0, event_types=term_types
                            )
                            for ev in events or []:
                                payload = ev.get("payload") or {}
                                iid = payload.get("instance_id") or ev.get(
                                    "instance_id"
                                )
                                if str(iid) == str(instance_id):
                                    seen_terminal = True
                                    break
                        except Exception:
                            seen_terminal = False
                        if not seen_terminal:
                            sid_env = None
                            for (
                                sid,
                                strat,
                            ) in self.state_manager.current_state.strategies.items():
                                if instance_id in strat.instance_ids:
                                    sid_env = sid
                                    break
                            if info.state == InstanceStatus.COMPLETED:
                                artifact = {
                                    "type": "branch",
                                    "branch_planned": info.branch_name,
                                    "branch_final": info.result.branch_name,
                                    "base": info.base_branch,
                                    "commit": getattr(info.result, "commit", None)
                                    or "",
                                    "has_changes": bool(
                                        getattr(info.result, "has_changes", False)
                                    ),
                                    "duplicate_of_branch": getattr(
                                        info.result, "duplicate_of_branch", None
                                    ),
                                    "dedupe_reason": getattr(
                                        info.result, "dedupe_reason", None
                                    ),
                                }
                                # Final message truncation per spec (backfill path)
                                try:
                                    max_bytes = 65536
                                except Exception:
                                    max_bytes = 65536
                                full_msg = (
                                    getattr(info.result, "final_message", None) or ""
                                )
                                msg_bytes = full_msg.encode("utf-8", errors="ignore")
                                truncated_flag = False
                                rel_path = ""
                                out_msg = full_msg
                                if max_bytes > 0 and len(msg_bytes) > max_bytes:
                                    truncated_flag = True
                                    out_msg = msg_bytes[:max_bytes].decode(
                                        "utf-8", errors="ignore"
                                    )
                                    try:
                                        run_id = self.state_manager.current_state.run_id
                                        run_logs = self.logs_dir / run_id
                                        dest_dir = run_logs / "final_messages"
                                        dest_dir.mkdir(parents=True, exist_ok=True)
                                        fname = f"{instance_id}.txt"
                                        with open(
                                            dest_dir / fname,
                                            "w",
                                            encoding="utf-8",
                                            errors="ignore",
                                        ) as fh:
                                            fh.write(full_msg)
                                        rel_path = f"final_messages/{fname}"
                                    except Exception:
                                        truncated_flag = False
                                        rel_path = ""
                                self.event_bus.emit_canonical(
                                    type="task.completed",
                                    run_id=self.state_manager.current_state.run_id,
                                    strategy_execution_id=sid_env,
                                    key=task_key,
                                    payload={
                                        "key": task_key,
                                        "instance_id": instance_id,
                                        "artifact": artifact,
                                        "metrics": info.result.metrics or {},
                                        "final_message": out_msg,
                                        "final_message_truncated": truncated_flag,
                                        "final_message_path": rel_path,
                                    },
                                )
                            elif info.state == InstanceStatus.FAILED:
                                self.event_bus.emit_canonical(
                                    type="task.failed",
                                    run_id=self.state_manager.current_state.run_id,
                                    strategy_execution_id=sid_env,
                                    key=task_key,
                                    payload={
                                        "key": task_key,
                                        "instance_id": instance_id,
                                        "error_type": info.result.error_type
                                        or "unknown",
                                        "message": info.result.error or "",
                                    },
                                )
                except Exception as _e:
                    logger.debug(
                        f"spawn_instance: terminal backfill emit check failed: {_e}"
                    )
            elif info.state == InstanceStatus.INTERRUPTED:
                # Create a pending future; resume_run will enqueue it if resumable
                self._instance_futures[instance_id] = asyncio.Future()
                logger.debug(
                    f"spawn_instance: iid={instance_id} interrupted; pending future created (enqueue deferred to resume)"
                )
            else:
                # QUEUED or RUNNING: create future and ensure it's enqueued
                future = asyncio.Future()
                self._instance_futures[instance_id] = future
                # If not already queued/running in executor, enqueue
                logger.info(f"spawn_instance: queue iid={instance_id} for execution")
                await self._instance_queue.put(instance_id)
                self.event_bus.emit(
                    "instance.queued",
                    {
                        "instance_id": instance_id,
                        "strategy": strategy_name,
                        "branch_name": branch_name,
                    },
                    instance_id=instance_id,
                )
                # Canonical task.scheduled already emitted in StrategyContext.run; no-op here

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
                    logger.debug(
                        "executor: waiting for instance from queue (qsize=%s)",
                        self._instance_queue.qsize(),
                    )
                    instance_id = await asyncio.wait_for(
                        self._instance_queue.get(), timeout=1.0
                    )
                    logger.info(f"executor: dequeued instance {instance_id}")
                except asyncio.TimeoutError:
                    continue

                # Acquire resource slot
                async with self._resource_pool:
                    try:
                        logger.debug(
                            "executor: acquired slot (active=%d/%d)",
                            len(self._active_instances),
                            self.max_parallel_instances,
                        )
                    except Exception:
                        pass
                    if self._shutdown:
                        break

                    # Mark as active
                    self._active_instances.add(instance_id)

                    # Execute instance
                    try:
                        # Admission wait: CPU+memory tokens and disk guard (use per-task overrides when present)
                        info = self.state_manager.current_state.instances.get(
                            instance_id
                        )
                        cpu_need = (
                            max(
                                1,
                                int(
                                    (info.metadata or {}).get(
                                        "container_cpu", self.container_limits.cpu_count
                                    )
                                ),
                            )
                            if info
                            else max(1, int(self.container_limits.cpu_count))
                        )
                        mem_need = (
                            max(
                                1,
                                int(
                                    (info.metadata or {}).get(
                                        "container_memory_gb",
                                        self.container_limits.memory_gb,
                                    )
                                ),
                            )
                            if info
                            else max(1, int(self.container_limits.memory_gb))
                        )
                        await self._admission_wait(cpu_need, mem_need)
                        # Emit canonical debug with key
                        try:
                            info = self.state_manager.current_state.instances.get(
                                instance_id
                            )
                            key = (info.metadata or {}).get("key") if info else None
                            if key and self.event_bus:
                                self.event_bus.emit_canonical(
                                    type="strategy.debug",
                                    run_id=self.state_manager.current_state.run_id,
                                    strategy_execution_id=next(
                                        (
                                            sid
                                            for sid, strat in self.state_manager.current_state.strategies.items()
                                            if instance_id in strat.instance_ids
                                        ),
                                        None,
                                    ),
                                    key=key,
                                    payload={
                                        "op": "executor_start",
                                        "instance_id": instance_id,
                                    },
                                )
                        except Exception:
                            pass
                        await self._execute_instance(instance_id)
                    finally:
                        # Release admission tokens
                        try:
                            info = self.state_manager.current_state.instances.get(
                                instance_id
                            )
                            cpu_rel = (
                                max(
                                    1,
                                    int(
                                        (info.metadata or {}).get(
                                            "container_cpu",
                                            self.container_limits.cpu_count,
                                        )
                                    ),
                                )
                                if info
                                else max(1, int(self.container_limits.cpu_count))
                            )
                            mem_rel = (
                                max(
                                    1,
                                    int(
                                        (info.metadata or {}).get(
                                            "container_memory_gb",
                                            self.container_limits.memory_gb,
                                        )
                                    ),
                                )
                                if info
                                else max(1, int(self.container_limits.memory_gb))
                            )
                            await self._admission_release(cpu_rel, mem_rel)
                        except Exception:
                            pass
                        self._active_instances.remove(instance_id)

            except OrchestratorError as e:
                logger.exception(f"Error in instance executor: {e}")
            except asyncio.CancelledError:
                # Expected during shutdown or task cancellation; do not log as error
                break

    async def _admission_wait(self, cpu_need: int, mem_need_gb: int) -> None:
        """No-op admission: do not gate on CPU/memory/disk; honor only max_parallel semaphore."""
        return

    async def _admission_release(self, cpu: int, mem_gb: int) -> None:
        return

    # Legacy GC removed: workspaces are cleaned immediately per instance

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
        # Note: canonical task.started is emitted below with full payload (container_name/model)
        logger.debug(
            f"_execute_instance: iid={instance_id} -> RUNNING model={(info.metadata or {}).get('model','-')} key={(info.metadata or {}).get('key','-')}"
        )

        # Emit instance.started event for TUI and canonical mapping when key available
        self.event_bus.emit(
            "instance.started",
            {
                "strategy": info.strategy_name,
                "prompt": info.prompt,
                "model": info.metadata.get("model", "sonnet"),
                "branch_name": info.branch_name,
            },
            instance_id=instance_id,
        )
        task_key = (info.metadata or {}).get("key")
        if task_key:
            self.event_bus.emit_canonical(
                type="task.started",
                run_id=self.state_manager.current_state.run_id,
                strategy_execution_id=next(
                    (
                        sid
                        for sid, strat in self.state_manager.current_state.strategies.items()
                        if instance_id in strat.instance_ids
                    ),
                    None,
                ),
                key=task_key,
                payload={
                    "key": task_key,
                    "instance_id": instance_id,
                    "container_name": info.container_name,
                    "model": info.metadata.get("model", "sonnet"),
                },
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
            # Forward runner-level events to in-memory bus for local diagnostics
            try:
                self.event_bus.emit(
                    event_type=event.get("type", "instance.event"),
                    data=data,
                    instance_id=instance_id,
                )
            except Exception:
                pass

            # Map select instance.* events to canonical task.progress for the TUI/file
            try:
                et = str(event.get("type", ""))
                phase = None
                activity = None
                tool = None
                # Workspace and container phases
                if et == "instance.workspace_preparing":
                    phase, activity = "workspace_preparing", "Preparing workspace..."
                elif et in (
                    "instance.container_creating",
                    "instance.container_create_call",
                    "instance.container_create_attempt",
                    "instance.container_image_check",
                ):
                    phase, activity = "container_creating", "Creating container..."
                elif et == "instance.container_env_preparing":
                    phase, activity = (
                        "container_env_preparing",
                        "Preparing container env...",
                    )
                elif et == "instance.startup_waiting":
                    phase, activity = (
                        "startup_waiting",
                        "Waiting for startup slot...",
                    )
                elif et == "instance.container_env_prepared":
                    phase, activity = "container_env_prepared", "Container env ready"
                elif et == "instance.container_created":
                    phase, activity = "container_created", "Container created"
                elif et == "instance.container_adopted":
                    # Treat adoption as created for UX
                    phase, activity = "container_created", "Container adopted"
                elif et == "instance.agent_starting":
                    phase, activity = "agent_starting", "Starting Agent..."
                elif et == "instance.result_collection_started":
                    phase, activity = "result_collection", "Collecting results..."
                elif et == "instance.branch_imported":
                    phase, activity = (
                        "branch_imported",
                        f"Imported branch {data.get('branch_name','')}",
                    )
                elif et == "instance.no_changes":
                    phase, activity = "no_changes", "No changes"
                elif et == "instance.workspace_cleaned":
                    phase, activity = "cleanup", "Workspace cleaned"
                # Agent tool and message events
                elif et == "instance.agent_tool_use":
                    phase, tool = "tool_use", (
                        data.get("tool") or data.get("data", {}).get("tool")
                    )
                    activity = f"Using {tool}" if tool else "Tool use"
                elif et == "instance.agent_assistant":
                    phase, activity = "assistant", "Agent is thinking..."
                elif et == "instance.agent_system":
                    phase, activity = "system", "Agent connected"

                if phase and task_key:
                    self.event_bus.emit_canonical(
                        type="task.progress",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=next(
                            (
                                sid
                                for sid, strat in self.state_manager.current_state.strategies.items()
                                if instance_id in strat.instance_ids
                            ),
                            None,
                        ),
                        key=task_key,
                        payload={
                            "key": task_key,
                            "instance_id": instance_id,
                            "phase": phase,
                            **({"activity": activity} if activity else {}),
                            **({"tool": tool} if tool else {}),
                        },
                    )
            except Exception:
                pass

        # Start a lightweight heartbeat that reports recent activity for this instance
        heartbeat_task: Optional[asyncio.Task] = None
        try:
            if self.event_bus:
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_monitor(instance_id)
                )
            # Extract strategy execution ID from instance metadata
            strategy_execution_id = None
            for sid, strat in self.state_manager.current_state.strategies.items():
                if instance_id in strat.instance_ids:
                    strategy_execution_id = sid
                    break

            # Run the instance
            logger.info(
                f"_execute_instance: starting run_instance iid={instance_id} container={info.container_name}"
            )
            # Effective per-task container limits
            eff_cpu = max(
                1,
                int(
                    (info.metadata or {}).get(
                        "container_cpu", self.container_limits.cpu_count
                    )
                ),
            )
            eff_mem = max(
                1,
                int(
                    (info.metadata or {}).get(
                        "container_memory_gb", self.container_limits.memory_gb
                    )
                ),
            )
            result = await run_instance(
                # Always pass the original task prompt; runner decides per-attempt
                # whether to use a continuation prompt when resuming a session.
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
                session_id=(
                    info.session_id or (info.metadata or {}).get("resume_session_id")
                ),
                operator_resume=bool(
                    (info.metadata or {}).get("operator_resume", False)
                ),
                event_callback=event_callback,
                startup_semaphore=self._startup_pool,
                timeout_seconds=self.runner_timeout_seconds,
                container_limits=ContainerLimits(
                    cpu_count=eff_cpu, memory_gb=eff_mem, memory_swap_gb=eff_mem
                ),
                auth_config=self.auth_config,
                retry_config=self.retry_config,
                plugin_name=(info.metadata or {}).get("plugin_name", "claude-code"),
                docker_image=(info.metadata or {}).get("docker_image")
                or self.default_docker_image,
                import_policy=(info.metadata or {}).get("import_policy", "auto"),
                import_conflict_policy=(info.metadata or {}).get(
                    "import_conflict_policy", "fail"
                ),
                skip_empty_import=bool(
                    (info.metadata or {}).get("skip_empty_import", True)
                ),
                network_egress=(info.metadata or {}).get("network_egress"),
                max_turns=(info.metadata or {}).get("max_turns"),
                reuse_container=bool(
                    (info.metadata or {}).get("reuse_container", True)
                ),
                allow_overwrite_protected_refs=self.allow_overwrite_protected_refs,
                allow_global_session_volume=self.allow_global_session_volume,
                agent_cli_args=(info.metadata or {}).get("agent_cli_args"),
                force_commit=self.force_commit,
            )
            logger.info(
                f"_execute_instance: run_instance finished iid={instance_id} success={result.success} status={getattr(result,'status',None)}"
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
                if (
                    getattr(result, "status", None) == "canceled"
                    or getattr(result, "error_type", None) == "canceled"
                ):
                    new_state = InstanceStatus.INTERRUPTED
                else:
                    new_state = InstanceStatus.FAILED

            self.state_manager.update_instance_state(
                instance_id=instance_id,
                state=new_state,
                result=result,
            )
            logger.debug(
                f"_execute_instance: iid={instance_id} terminal={new_state.value}"
            )

            # Emit canonical terminal events when key is present
            if task_key:
                if result.success:
                    artifact = {
                        "type": "branch",
                        "branch_planned": info.branch_name,
                        "branch_final": result.branch_name,
                        "base": info.base_branch,
                        "commit": getattr(result, "commit", None) or "",
                        "has_changes": result.has_changes,
                        "duplicate_of_branch": getattr(
                            result, "duplicate_of_branch", None
                        ),
                        "dedupe_reason": getattr(result, "dedupe_reason", None),
                    }
                    # Final message truncation per spec
                    try:
                        max_bytes = 65536
                    except Exception:
                        max_bytes = 65536
                    full_msg = result.final_message or ""
                    msg_bytes = full_msg.encode("utf-8", errors="ignore")
                    truncated_flag = False
                    rel_path = ""
                    out_msg = full_msg
                    if max_bytes > 0 and len(msg_bytes) > max_bytes:
                        truncated_flag = True
                        out_msg = msg_bytes[:max_bytes].decode("utf-8", errors="ignore")
                        # Write full message to run-relative path under logs/<run_id>/
                        try:
                            run_id = self.state_manager.current_state.run_id
                            run_logs = self.logs_dir / run_id
                            dest_dir = run_logs / "final_messages"
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"{instance_id}.txt"
                            with open(
                                dest_dir / fname, "w", encoding="utf-8", errors="ignore"
                            ) as fh:
                                fh.write(full_msg)
                            rel_path = f"final_messages/{fname}"
                        except Exception:
                            truncated_flag = False
                            rel_path = ""
                    logger.debug(
                        f"emit task.completed key={task_key} iid={instance_id} branch={result.branch_name} has_changes={result.has_changes}"
                    )
                    self.event_bus.emit_canonical(
                        type="task.completed",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=strategy_execution_id,
                        key=task_key,
                        payload={
                            "key": task_key,
                            "instance_id": instance_id,
                            "artifact": artifact,
                            "metrics": result.metrics or {},
                            "final_message": out_msg,
                            "final_message_truncated": truncated_flag,
                            "final_message_path": rel_path,
                        },
                    )
                elif new_state != InstanceStatus.INTERRUPTED:
                    logger.debug(
                        f"emit task.failed key={task_key} iid={instance_id} err={result.error_type}:{result.error}"
                    )
                    # Map internal error types to the spec's closed set
                    _map = {
                        "docker": "docker",
                        "git": "git",
                        "timeout": "timeout",
                        "auth": "auth",
                        "session_corrupted": "session_corrupted",
                        "claude": "api",
                        "orchestration": "unknown",
                        "validation": "unknown",
                        "system": "unknown",
                        "unexpected": "unknown",
                    }
                    etype = (result.error_type or "unknown").lower()
                    mapped_type = _map.get(
                        etype,
                        (
                            etype
                            if etype
                            in {
                                "docker",
                                "api",
                                "network",
                                "git",
                                "timeout",
                                "session_corrupted",
                                "auth",
                                "unknown",
                            }
                            else "unknown"
                        ),
                    )
                    # Offline egress failures → classify as network per spec
                    try:
                        if (info.metadata or {}).get(
                            "network_egress"
                        ) == "offline" and mapped_type != "canceled":
                            mapped_type = "network"
                    except Exception:
                        pass
                    self.event_bus.emit_canonical(
                        type="task.failed",
                        run_id=self.state_manager.current_state.run_id,
                        strategy_execution_id=strategy_execution_id,
                        key=task_key,
                        payload={
                            "key": task_key,
                            "instance_id": instance_id,
                            "error_type": mapped_type,
                            "message": result.error or "",
                            "network_egress": (info.metadata or {}).get(
                                "network_egress"
                            ),
                        },
                    )
                else:
                    # For interruptions, StateManager.update_instance_state emits canonical task.interrupted.
                    # Avoid duplicate emission here.
                    pass

            # Resolve future
            self._instance_futures[instance_id].set_result(result)

        except asyncio.CancelledError:
            # Treat orchestrator-initiated cancellation as interruption, not failure
            logger.info(
                f"Instance {instance_id} canceled by shutdown; marking interrupted"
            )
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
            # If shutdown in progress, classify as interrupted rather than failed
            if getattr(self, "_shutdown", False):
                logger.info(
                    f"Instance {instance_id} canceled during shutdown (init/runner error): {e}"
                )
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
            else:
                logger.exception(f"Instance {instance_id} execution failed: {e}")
                error_result = InstanceResult(
                    success=False,
                    error=str(e),
                    error_type="orchestration",
                )
                self.state_manager.update_instance_state(
                    instance_id=instance_id,
                    state=InstanceStatus.FAILED,
                    result=error_result,
                )
                self._instance_futures[instance_id].set_result(error_result)
        except Exception as e:
            # Catch-all; treat as interrupted when shutting down to avoid misclassifying user cancels as failures
            if getattr(self, "_shutdown", False):
                logger.info(
                    f"Instance {instance_id} canceled during shutdown (unexpected error): {e}"
                )
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
            else:
                logger.exception(
                    f"Instance {instance_id} crashed with unexpected error: {e}"
                )
                error_result = InstanceResult(
                    success=False,
                    error=str(e),
                    error_type="unexpected",
                )
                self.state_manager.update_instance_state(
                    instance_id=instance_id,
                    state=InstanceStatus.FAILED,
                    result=error_result,
                )
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
                    logger.debug(
                        f"Heartbeat: instance {instance_id} awaiting first event"
                    )

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

    # Removed: Orchestrator-level orphan cleanup. Containers are removed immediately
    # on success/failure/timeout at the runner level.

    async def resume_run(self, run_id: str) -> List[InstanceResult]:
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
            self.event_bus = EventBus(
                max_events=self.event_buffer_size, persist_path=event_log_path
            )
        else:
            # Repoint persist path to this run
            self.event_bus.persist_path = event_log_path
            self.event_bus._open_persist_file()
        self.state_manager.event_bus = self.event_bus

        saved_state = await self.state_manager.load_and_recover_state(run_id)
        if not saved_state:
            raise ValueError(f"No saved state found for run {run_id}")

        # Prime repo_path for StrategyContext.spawn_instance on re-entry
        try:
            self.repo_path = saved_state.repo_path
        except Exception:
            pass
        try:
            # Log instance state counts for diagnostics
            counts = {
                "queued": 0,
                "running": 0,
                "interrupted": 0,
                "completed": 0,
                "failed": 0,
            }
            for iid, info in self.state_manager.current_state.instances.items():
                k = info.state.value
                counts[k] = counts.get(k, 0) + 1
            logger.info(
                f"resume_run: instances total={len(self.state_manager.current_state.instances)} counts={counts}"
            )
        except Exception:
            pass

        # Backfill missing canonical terminal events for already-terminal instances
        try:
            # Collect instance_ids that already have terminal canonical events in the log
            term_types = {"task.completed", "task.failed", "task.interrupted"}
            seen_terminals: set[str] = set()
            try:
                events, _ = self.event_bus.get_events_since(
                    offset=0, event_types=term_types
                )
            except Exception:
                events = []
            for ev in events or []:
                payload = ev.get("payload") or {}
                iid = payload.get("instance_id") or ev.get("instance_id")
                if iid:
                    seen_terminals.add(str(iid))

            # Emit synthetic terminal events for terminal instances missing from the log
            for iid, info in list(self.state_manager.current_state.instances.items()):
                try:
                    task_key = (info.metadata or {}).get("key")
                    if not task_key:
                        continue
                    if info.state.value not in ("completed", "failed", "interrupted"):
                        continue
                    if iid in seen_terminals:
                        continue
                    # Find strategy execution id for envelope
                    strategy_execution_id = None
                    for (
                        sid,
                        strat,
                    ) in self.state_manager.current_state.strategies.items():
                        if iid in strat.instance_ids:
                            strategy_execution_id = sid
                            break
                    if info.state.value == "completed":
                        res = info.result
                        if not res:
                            continue
                        artifact = {
                            "type": "branch",
                            "branch_planned": info.branch_name,
                            "branch_final": res.branch_name,
                            "base": info.base_branch,
                            "commit": getattr(res, "commit", None) or "",
                            "has_changes": bool(res.has_changes),
                            "duplicate_of_branch": getattr(
                                res, "duplicate_of_branch", None
                            ),
                            "dedupe_reason": getattr(res, "dedupe_reason", None),
                        }
                        logger.debug(
                            f"backfill: task.completed iid={iid} key={task_key}"
                        )
                        # Final message truncation per spec (resume-run backfill)
                        try:
                            max_bytes = 65536
                        except Exception:
                            max_bytes = 65536
                        full_msg = getattr(res, "final_message", None) or ""
                        msg_bytes = full_msg.encode("utf-8", errors="ignore")
                        truncated_flag = False
                        rel_path = ""
                        out_msg = full_msg
                        if max_bytes > 0 and len(msg_bytes) > max_bytes:
                            truncated_flag = True
                            out_msg = msg_bytes[:max_bytes].decode(
                                "utf-8", errors="ignore"
                            )
                            try:
                                run_logs = self.logs_dir / run_id
                                dest_dir = run_logs / "final_messages"
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                fname = f"{iid}.txt"
                                with open(
                                    dest_dir / fname,
                                    "w",
                                    encoding="utf-8",
                                    errors="ignore",
                                ) as fh:
                                    fh.write(full_msg)
                                rel_path = f"final_messages/{fname}"
                            except Exception:
                                truncated_flag = False
                                rel_path = ""
                        self.event_bus.emit_canonical(
                            type="task.completed",
                            run_id=run_id,
                            strategy_execution_id=strategy_execution_id,
                            key=task_key,
                            payload={
                                "key": task_key,
                                "instance_id": iid,
                                "artifact": artifact,
                                "metrics": res.metrics or {},
                                "final_message": out_msg,
                                "final_message_truncated": truncated_flag,
                                "final_message_path": rel_path,
                            },
                        )
                    elif info.state.value == "failed":
                        res = info.result
                        logger.debug(f"backfill: task.failed iid={iid} key={task_key}")
                        self.event_bus.emit_canonical(
                            type="task.failed",
                            run_id=run_id,
                            strategy_execution_id=strategy_execution_id,
                            key=task_key,
                            payload={
                                "key": task_key,
                                "instance_id": iid,
                                "error_type": (res.error_type if res else "unknown"),
                                "message": (res.error if res else ""),
                                "network_egress": (info.metadata or {}).get(
                                    "network_egress"
                                ),
                            },
                        )
                    else:  # interrupted
                        logger.debug(
                            f"backfill: task.interrupted iid={iid} key={task_key}"
                        )
                        self.event_bus.emit_canonical(
                            type="task.interrupted",
                            run_id=run_id,
                            strategy_execution_id=strategy_execution_id,
                            key=task_key,
                            payload={"key": task_key, "instance_id": iid},
                        )
                except Exception:
                    # best-effort backfill
                    pass
        except Exception:
            pass

        # Re-open strategies that were marked terminal due to interruption so we can re-enter
        try:
            for sid, strat in list(self.state_manager.current_state.strategies.items()):
                # If strategy was marked completed/failed but there is work to resume or continue, set to running
                # Heuristic: if any instance exists for this strategy (regardless of state), we allow re-entry
                if getattr(strat, "completed_at", None) is not None:
                    strat.completed_at = None
                    old = strat.state
                    strat.state = "running"
                    # Emit state update event
                    logger.info(
                        f"Re-opening strategy {sid} for resume re-entry (was {old})"
                    )
                    if self.event_bus:
                        self.event_bus.emit(
                            "state.strategy_updated",
                            {
                                "strategy_id": sid,
                                "old_state": old,
                                "new_state": "running",
                            },
                        )
                        # Also emit a canonical strategy.started with a 'resumed' hint for downstream consumers
                        try:
                            self.event_bus.emit_canonical(
                                type="strategy.started",
                                run_id=run_id,
                                strategy_execution_id=sid,
                                payload={
                                    "name": strat.strategy_name,
                                    "params": strat.config or {},
                                    "resumed": True,
                                },
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        # Start periodic snapshots during resume for liveness
        await self.state_manager.start_periodic_snapshots()

        # Respect resource gating by using the executor queue
        from ..shared import InstanceStatus as _IS
        from ..instance_runner.plugins import AVAILABLE_PLUGINS

        # DockerManager import removed; containers are adopted at runner level when preserved

        scheduled_ids: list[str] = []
        cannot_resume_count = 0

        # Helper to enqueue existing instance IDs
        async def _enqueue(iid: str) -> None:
            if iid not in self._instance_futures:
                self._instance_futures[iid] = asyncio.Future()
            await self._instance_queue.put(iid)

        for iid, info in saved_state.instances.items():
            try:
                # Resume or re-run only RUNNING/INTERRUPTED; start queued normally
                if info.state in (_IS.RUNNING, _IS.INTERRUPTED):
                    if False:
                        # resume-fresh removed
                        pass
                    else:
                        # Resume only if container exists and plugin supports resume
                        can_resume = False
                        try:
                            _pname = (info.metadata or {}).get(
                                "plugin_name", "claude-code"
                            )
                            _pclass = AVAILABLE_PLUGINS.get(
                                _pname
                            ) or AVAILABLE_PLUGINS.get("claude-code")
                            _plugin = _pclass()  # type: ignore[call-arg]
                        except Exception:
                            _plugin = AVAILABLE_PLUGINS["claude-code"]()
                        # Resume session across containers when session_id exists
                        if info.session_id and _plugin.capabilities.supports_resume:
                            can_resume = True
                        else:
                            can_resume = False
                        if can_resume:
                            # Ensure reuse flag remains True
                            self.state_manager.update_instance_metadata(
                                iid, {"reuse_container": True, "operator_resume": True}
                            )
                            self.state_manager.update_instance_state(iid, _IS.QUEUED)
                            await _enqueue(iid)
                            scheduled_ids.append(iid)
                            try:
                                logger.info(
                                    f"resume_run: scheduled resume iid={iid} container={info.container_name} session_id={info.session_id} plugin={(info.metadata or {}).get('plugin_name','claude-code')}"
                                )
                            except Exception:
                                logger.info(
                                    f"resume_run: scheduled resume iid={iid} container={info.container_name}"
                                )
                        else:
                            # Fresh re-run: no session to resume or container missing; schedule a new container
                            try:
                                self.state_manager.update_instance_session_id(iid, None)
                                meta_patch = {
                                    "reuse_container": False,
                                    "operator_resume": True,
                                    "resume_session_id": None,
                                }
                                # generate a new container name suffix to avoid collisions
                                import uuid as _uuid

                                new_name = (
                                    f"{info.container_name}_r{_uuid.uuid4().hex[:4]}"
                                )
                                self.state_manager.update_instance_container_name(
                                    iid, new_name
                                )
                                meta_patch["container_name_override"] = new_name
                                self.state_manager.update_instance_metadata(
                                    iid, meta_patch
                                )
                            except Exception:
                                pass
                            self.state_manager.update_instance_state(iid, _IS.QUEUED)
                            await _enqueue(iid)
                            scheduled_ids.append(iid)
                            logger.info(
                                f"resume_run: scheduled fresh iid={iid} (no resumable session)"
                            )
                elif info.state == _IS.QUEUED:
                    await _enqueue(iid)
                    scheduled_ids.append(iid)
                    logger.info(f"resume_run: scheduled queued iid={iid}")
            except Exception as e:
                logger.debug(f"Resume scheduling error for {iid}: {e}")

        # Do NOT block on all scheduled instances here; allow strategy re-entry to await
        # them and schedule downstream work (e.g., scoring) as each completes.
        if scheduled_ids:
            logger.info(
                f"resume_run: enqueued {len(scheduled_ids)} instance(s) for resume"
            )

        # Re-enter strategies immediately to allow downstream stages (e.g., scoring) to run
        # IMPORTANT: Run strategy re-entry concurrently to avoid serializing downstream work
        reentry_results: list[InstanceResult] = []
        try:
            from .strategies import AVAILABLE_STRATEGIES as _STRATS

            prompt = self.state_manager.current_state.prompt
            base_branch = self.state_manager.current_state.base_branch

            async def _reenter_one(sid: str, strat_exec) -> list[InstanceResult]:
                # Skip if already completed
                if getattr(strat_exec, "completed_at", None):
                    return []
                sname = strat_exec.strategy_name
                if sname not in _STRATS:
                    return []
                logger.info(f"Re-entering strategy {sname} sid={sid}")
                strat_cls = _STRATS[sname]
                strat = strat_cls()
                # Reuse original config captured in state
                try:
                    strat.set_config_overrides(strat_exec.config or {})
                except Exception:
                    pass
                # Build context with the same execution id
                ctx = StrategyContext(self, sname, sid)
                try:
                    res = await strat.execute(
                        prompt=prompt, base_branch=base_branch, ctx=ctx
                    )
                except Exception as e:
                    logger.error(f"Strategy re-entry failed for {sname}: {e}")
                    res = []
                # Update strategy state and emit canonical completion
                state_value = (
                    "completed"
                    if any(getattr(r, "success", False) for r in (res or []))
                    else "failed"
                )
                self.state_manager.update_strategy_state(
                    strategy_id=sid, state=state_value, results=res
                )
                status = "success" if state_value == "completed" else "failed"
                payload = {"status": status}
                if status == "failed":
                    # If no successes and any instances are interrupted (not failed), classify as canceled
                    try:
                        any_interrupted = any(
                            (
                                self.state_manager.current_state.instances.get(
                                    iid
                                ).state.value
                                == "interrupted"
                            )
                            for iid in (
                                self.state_manager.current_state.strategies.get(
                                    sid
                                ).instance_ids
                                or []
                            )
                            if iid in self.state_manager.current_state.instances
                        )
                    except Exception:
                        any_interrupted = False
                    if any_interrupted:
                        payload["status"] = "canceled"
                        payload["reason"] = "operator_interrupt"
                    else:
                        payload["reason"] = "no_successful_tasks"
                try:
                    self.event_bus.emit_canonical(
                        type="strategy.completed",
                        run_id=run_id,
                        strategy_execution_id=sid,
                        payload=payload,
                    )
                except Exception:
                    pass
                return res or []

            tasks: list[asyncio.Task] = []
            for sid, strat_exec in list(
                self.state_manager.current_state.strategies.items()
            ):
                tasks.append(asyncio.create_task(_reenter_one(sid, strat_exec)))
            if tasks:
                results_lists = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results_lists:
                    if isinstance(res, list):
                        reentry_results.extend(res)
        except Exception as e:
            logger.debug(f"Strategy re-entry skipped due to error: {e}")

        # Mark run completion time for accurate summaries and save snapshot/results
        try:
            if self.state_manager and self.state_manager.current_state:
                self.state_manager.current_state.completed_at = datetime.now(
                    timezone.utc
                )
        except Exception:
            pass
        await self.state_manager.save_snapshot()
        # Prefer strategy-level results as the authoritative output; avoid mixing instance-level results
        final_results: list[InstanceResult] = []
        try:
            for sid, strat in self.state_manager.current_state.strategies.items():
                if strat.results:
                    final_results.extend(strat.results)
        except Exception:
            pass
        # Fallback: if no strategy results were recorded (unexpected), use reentry_results
        if not final_results:
            final_results = reentry_results
        await self._save_results(run_id, final_results)

        self.event_bus.emit(
            "run.completed",
            {
                "run_id": run_id,
                "resumed": True,
                "cannot_resume": cannot_resume_count,
                "total_results": len(final_results),
            },
        )

        return final_results

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
            # Effective per-task limits if present in saved metadata
            try:
                eff_cpu = max(
                    1,
                    int(
                        (instance_info.metadata or {}).get(
                            "container_cpu", self.container_limits.cpu_count
                        )
                    ),
                )
                eff_mem = max(
                    1,
                    int(
                        (instance_info.metadata or {}).get(
                            "container_memory_gb", self.container_limits.memory_gb
                        )
                    ),
                )
            except Exception:
                eff_cpu = self.container_limits.cpu_count
                eff_mem = self.container_limits.memory_gb
            result = await run_instance(
                prompt=instance_info.prompt,
                repo_path=self.state_manager.current_state.repo_path,
                base_branch=instance_info.base_branch,
                branch_name=instance_info.branch_name,
                instance_id=instance_info.instance_id,
                run_id=self.state_manager.current_state.run_id,
                strategy_execution_id=None,
                event_callback=lambda e: self.event_bus.emit(
                    e.get("type", "instance.event"),
                    e.get("data", {}),
                    instance_id=instance_info.instance_id,
                ),
                container_name=instance_info.container_name,
                startup_semaphore=self._startup_pool,
                container_limits=ContainerLimits(
                    cpu_count=eff_cpu, memory_gb=eff_mem, memory_swap_gb=eff_mem
                ),
                retry_config=self.retry_config,
                auth_config=self.auth_config,
                docker_image=(instance_info.metadata or {}).get("docker_image")
                or self.default_docker_image,
                import_policy=(instance_info.metadata or {}).get(
                    "import_policy", "auto"
                ),
                import_conflict_policy=(instance_info.metadata or {}).get(
                    "import_conflict_policy", "fail"
                ),
                skip_empty_import=bool(
                    (instance_info.metadata or {}).get("skip_empty_import", True)
                ),
                allow_overwrite_protected_refs=self.allow_overwrite_protected_refs,
                allow_global_session_volume=self.allow_global_session_volume,
                agent_cli_args=(instance_info.metadata or {}).get("agent_cli_args"),
                force_commit=self.force_commit,
            )

            # Update state
            self.state_manager.update_instance_state(
                instance_id=instance_info.instance_id,
                state=InstanceStatus.COMPLETED,
                result=result,
            )

            return result

        except (DockerError, GitError, OrchestratorError) as e:
            logger.error(f"Failed to re-run instance {instance_info.instance_id}: {e}")

            # Mark as failed
            self.state_manager.update_instance_state(
                instance_id=instance_info.instance_id,
                state=InstanceStatus.FAILED,
                result=InstanceResult(
                    success=False,
                    error=str(e),
                    error_type="rerun_failed",
                    branch_name=instance_info.branch_name,
                    status="failed",
                ),
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
        Log available disk space (informational only).
        No enforcement; runs proceed regardless of free space.
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

            # No minimum threshold enforced.

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
            # Run is considered interrupted if any instance ended in INTERRUPTED
            try:
                any_interrupted = any(
                    (i.state == InstanceStatus.INTERRUPTED)
                    for i in self.state_manager.current_state.instances.values()
                )
            except Exception:
                any_interrupted = False
            # Derive per-instance counts directly to avoid resume double-counting
            try:
                instance_list = list(state.instances.values())
                total_instances = len(instance_list)
                from ..shared import InstanceStatus as _IS

                completed_instances = sum(
                    1 for i in instance_list if i.state == _IS.COMPLETED
                )
                failed_instances = sum(
                    1 for i in instance_list if i.state == _IS.FAILED
                )
            except Exception:
                total_instances = state.total_instances
                completed_instances = state.completed_instances
                failed_instances = state.failed_instances

            summary_data = {
                "run_id": run_id,
                "status": ("interrupted" if any_interrupted else "completed"),
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
                "total_instances": total_instances,
                "completed_instances": completed_instances,
                "failed_instances": failed_instances,
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
                f.write(
                    "instance_id,branch_name,status,duration_seconds,cost,tokens,input_tokens,output_tokens,commit_count,lines_added,lines_deleted,has_changes\n"
                )

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
                        tf.write(
                            "timestamp,active_instances,completed_instances,failed_instances,total_cost,total_tokens,event_type\n"
                        )
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
                            elif et == "instance.agent_turn_complete" and iid:
                                tm = data.get("turn_metrics", {})
                                inst_tokens[iid] = inst_tokens.get(iid, 0) + int(
                                    tm.get("tokens", 0) or 0
                                )
                                inst_cost[iid] = inst_cost.get(iid, 0.0) + float(
                                    tm.get("cost", 0.0) or 0.0
                                )
                            elif et == "instance.agent_completed" and iid:
                                m = data.get("metrics", {})
                                if m:
                                    inst_tokens[iid] = int(
                                        m.get("total_tokens", inst_tokens.get(iid, 0))
                                    )
                                    inst_cost[iid] = float(
                                        m.get("total_cost", inst_cost.get(iid, 0.0))
                                    )
                            total_cost = sum(inst_cost.values())
                            total_tokens = sum(inst_tokens.values())
                            tf.write(
                                f"{ts},{len(running)},{len(completed_set)},{len(failed_set)},{total_cost:.4f},{total_tokens},{et}\n"
                            )
            except Exception as e:
                logger.debug(f"Failed generating time-series metrics: {e}")

            # Create strategy output directory
            strategy_dir = results_dir / "strategy_output"
            strategy_dir.mkdir(exist_ok=True)

            # Save strategy outputs (summary JSON only); strategy-specific metadata is kept in run state
            for strat_id, strat_info in state.strategies.items():
                if strat_info.results:
                    strategy_file = (
                        strategy_dir / f"{strat_info.strategy_name}_{strat_id}.json"
                    )
                    strategy_data = {
                        "strategy_id": strat_id,
                        "strategy_name": strat_info.strategy_name,
                        "config": strat_info.config,
                        # Do not interpret result metadata; just persist minimal success/branch
                        "results": [
                            {
                                "branch_name": (
                                    getattr(r, "branch_name", None)
                                    if not isinstance(r, dict)
                                    else r.get("branch_name")
                                ),
                                "success": (
                                    getattr(r, "success", False)
                                    if not isinstance(r, dict)
                                    else bool(r.get("success", False))
                                ),
                            }
                            for r in (strat_info.results or [])
                        ],
                    }

                    with open(strategy_file, "w") as f:
                        json.dump(strategy_data, f, indent=2)

            # Optional Markdown export (summary-focused)
            try:
                md_path = results_dir / "summary.md"
                with open(md_path, "w") as f:
                    f.write(f"# Pitaya Run Summary: {run_id}\n\n")
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
                                f"{r.duration_seconds:.0f}s"
                                if r.duration_seconds
                                else "-"
                            )
                            cost = (
                                r.metrics.get("total_cost", 0.0) if r.metrics else 0.0
                            )
                            tokens = (
                                r.metrics.get("total_tokens", 0) if r.metrics else 0
                            )
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
