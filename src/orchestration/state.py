"""
State management for orchestration runs.

Tracks the current state of all instances, strategies, and metrics.
Supports persistence and recovery through event sourcing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..shared import InstanceResult, InstanceStatus

if TYPE_CHECKING:
    from .event_bus import EventBus

logger = logging.getLogger(__name__)


# InstanceStatus is now imported from shared.types


@dataclass
class InstanceInfo:
    """Information about a single instance."""

    instance_id: str
    strategy_name: str
    prompt: str
    base_branch: str
    branch_name: str
    container_name: str
    state: InstanceStatus
    result: Optional[InstanceResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    session_id: Optional[str] = None
    interrupted_at: Optional[datetime] = None


@dataclass
class StrategyExecution:
    """Information about a strategy execution."""

    strategy_id: str
    strategy_name: str
    config: Dict[str, Any]
    instance_ids: List[str] = field(default_factory=list)
    state: str = "running"  # running, completed, failed
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    results: List[InstanceResult] = field(default_factory=list)


@dataclass
class RunState:
    """Complete state of an orchestration run."""

    run_id: str
    prompt: str
    repo_path: Path
    base_branch: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    # Instance tracking
    instances: Dict[str, InstanceInfo] = field(default_factory=dict)

    # Strategy tracking
    strategies: Dict[str, StrategyExecution] = field(default_factory=dict)

    # Aggregate metrics
    total_cost: float = 0.0
    total_tokens: int = 0
    total_instances: int = 0
    completed_instances: int = 0
    failed_instances: int = 0

    # Event tracking
    last_event_offset: int = 0

    # Durable task registry: key -> {fingerprint:str}
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "prompt": self.prompt,
            "repo_path": str(self.repo_path),
            "base_branch": self.base_branch,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_instances": self.total_instances,
            "completed_instances": self.completed_instances,
            "failed_instances": self.failed_instances,
            "last_event_offset": self.last_event_offset,
            "instances": {
                id: {
                    "instance_id": info.instance_id,
                    "strategy_name": info.strategy_name,
                    "prompt": info.prompt,
                    "base_branch": info.base_branch,
                    "branch_name": info.branch_name,
                    "container_name": info.container_name,
                    "session_id": info.session_id,
                    "state": info.state.value,
                    "metadata": info.metadata,
                    "created_at": info.created_at.isoformat(),
                    "started_at": (
                        info.started_at.isoformat() if info.started_at else None
                    ),
                    "completed_at": (
                        info.completed_at.isoformat() if info.completed_at else None
                    ),
                    "interrupted_at": (
                        info.interrupted_at.isoformat() if info.interrupted_at else None
                    ),
                }
                for id, info in self.instances.items()
            },
            "strategies": {
                id: {
                    "strategy_id": strat.strategy_id,
                    "strategy_name": strat.strategy_name,
                    "config": strat.config,
                    "instance_ids": strat.instance_ids,
                    "state": strat.state,
                    "started_at": strat.started_at.isoformat(),
                    "completed_at": (
                        strat.completed_at.isoformat() if strat.completed_at else None
                    ),
                }
                for id, strat in self.strategies.items()
            },
            "tasks": self.tasks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        """Restore state from dictionary."""
        state = cls(
            run_id=data["run_id"],
            prompt=data["prompt"],
            repo_path=Path(data["repo_path"]),
            base_branch=data["base_branch"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_instances=data.get("total_instances", 0),
            completed_instances=data.get("completed_instances", 0),
            failed_instances=data.get("failed_instances", 0),
            last_event_offset=data.get("last_event_offset", 0),
        )

        # Restore instances
        for id, info_data in data.get("instances", {}).items():
            state.instances[id] = InstanceInfo(
                instance_id=info_data["instance_id"],
                strategy_name=info_data["strategy_name"],
                prompt=info_data["prompt"],
                base_branch=info_data["base_branch"],
                branch_name=info_data["branch_name"],
                container_name=info_data["container_name"],
                session_id=info_data.get("session_id"),
                state=InstanceStatus(info_data["state"]),
                metadata=info_data.get("metadata", {}),
                created_at=datetime.fromisoformat(info_data["created_at"]),
                started_at=(
                    datetime.fromisoformat(info_data["started_at"])
                    if info_data["started_at"]
                    else None
                ),
                completed_at=(
                    datetime.fromisoformat(info_data["completed_at"])
                    if info_data["completed_at"]
                    else None
                ),
                interrupted_at=(
                    datetime.fromisoformat(info_data["interrupted_at"]) if info_data.get("interrupted_at") else None
                ),
            )

        # Restore strategies
        for id, strat_data in data.get("strategies", {}).items():
            state.strategies[id] = StrategyExecution(
                strategy_id=strat_data["strategy_id"],
                strategy_name=strat_data["strategy_name"],
                config=strat_data["config"],
                instance_ids=strat_data["instance_ids"],
                state=strat_data["state"],
                started_at=datetime.fromisoformat(strat_data["started_at"]),
                completed_at=(
                    datetime.fromisoformat(strat_data["completed_at"])
                    if strat_data["completed_at"]
                    else None
                ),
            )

        return state



class StateManager:
    """
    Manages orchestration state with persistence.

    Primary state lives in memory for fast access. Periodic snapshots
    and event sourcing enable recovery after crashes.
    """

    def __init__(
        self,
        state_dir: Path,
        event_bus: Optional["EventBus"] = None,
        snapshot_interval: int = 30,  # seconds
    ):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state persistence
            event_bus: Event bus for emitting state change events
            snapshot_interval: How often to save snapshots
        """
        self.state_dir = state_dir
        self.event_bus = event_bus
        self.snapshot_interval = snapshot_interval
        self.current_state: Optional[RunState] = None
        self._last_snapshot = datetime.now(timezone.utc)
        self._snapshot_task: Optional[asyncio.Task] = None

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def initialize_run(
        self,
        run_id: str,
        prompt: str,
        repo_path: Path,
        base_branch: str,
    ) -> RunState:
        """Initialize state for a new run."""
        self.current_state = RunState(
            run_id=run_id,
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
        )

        # Emit state change event
        if self.event_bus:
            self.event_bus.emit(
                "state.run_initialized",
                {
                    "run_id": run_id,
                    "prompt": prompt,
                    "repo_path": str(repo_path),
                    "base_branch": base_branch,
                },
            )

        # Save initial snapshot asynchronously
        asyncio.create_task(self.save_snapshot())

        return self.current_state

    async def load_and_recover_state(self, run_id: str) -> Optional[RunState]:
        """Load state from a previous run and apply events since last snapshot.

        This implements the full recovery process:
        1. Load state.json if it exists
        2. Read events.jsonl from the saved offset
        3. Apply events to reconstruct current state
        """
        snapshot_path = self.state_dir / run_id / "state.json"
        logs_snapshot_path = None
        if self.event_bus and getattr(self.event_bus, "persist_path", None):
            logs_snapshot_path = self.event_bus.persist_path.parent / "state.json"

        # Step 1: Load snapshot if it exists (prefer logs location per spec)
        candidate_paths = [p for p in [logs_snapshot_path, snapshot_path] if p is not None]
        loaded = False
        for path in candidate_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    self.current_state = RunState.from_dict(data)
                    last_offset = self.current_state.last_event_offset
                    logger.info(
                        f"Loaded state snapshot for run {run_id} from {path} at offset {last_offset}"
                    )
                    loaded = True
                    break
                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    # Don't give up immediately; try next candidate path
                    logger.error(
                        f"Failed to load state for run {run_id} from {path}: {e}"
                    )
                    continue

        if not loaded:
            logger.warning(
                f"No snapshot found for run {run_id}, attempting recovery from events only"
            )
            # Create minimal state to replay events into
            self.current_state = RunState(
                run_id=run_id,
                prompt="",  # Will be filled from events
                repo_path=Path("."),  # Will be updated from events
                base_branch="main",  # Will be updated from events
            )
            last_offset = 0

        # Step 2 & 3: Read and apply events since last snapshot
        if self.event_bus:
            try:
                # Get events since the last applied offset
                all_events, next_offset = self.event_bus.get_events_since(
                    offset=last_offset,
                    event_types=None,
                    limit=None,
                )

                # Filter only state.* events
                state_events = [e for e in all_events if str(e.get("type", "")).startswith("state.")]

                if state_events:
                    # Apply events to reconstruct state
                    await self.rebuild_from_events(state_events)

                    # Update last offset to the returned next offset
                    self.current_state.last_event_offset = next_offset
                    logger.info(f"Applied {len(state_events)} events to recover state")
                else:
                    logger.info("No new state events to apply since snapshot")

            except Exception as e:
                logger.error(f"Failed to recover events for run {run_id}: {e}")
                # Continue with snapshot state if event recovery fails

        # Crash inference: mark RUNNING without terminal events as INTERRUPTED
        if self.current_state:
            try:
                # Determine which instances have terminal events since last_offset
                terminal_iids: set[str] = set()
                for ev in (all_events or []):
                    et = str(ev.get("type", ""))
                    iid = ev.get("instance_id")
                    if not iid:
                        continue
                    if et in ("instance.completed", "instance.failed"):
                        terminal_iids.add(iid)
                    if et == "state.instance_updated":
                        data = ev.get("data", {})
                        if data.get("new_state") in (InstanceStatus.COMPLETED.value, InstanceStatus.FAILED.value):
                            terminal_iids.add(iid)

                for iid, info in list(self.current_state.instances.items()):
                    if info.state == InstanceStatus.RUNNING and iid not in terminal_iids:
                        info.state = InstanceStatus.INTERRUPTED
                        info.interrupted_at = datetime.now(timezone.utc)
                        if self.event_bus:
                            self.event_bus.emit(
                                "state.instance_updated",
                                {
                                    "instance_id": iid,
                                    "old_state": InstanceStatus.RUNNING.value,
                                    "new_state": InstanceStatus.INTERRUPTED.value,
                                    "interrupted_at": info.interrupted_at.isoformat(),
                                },
                                instance_id=iid,
                            )
            except Exception as e:
                logger.debug(f"Crash inference check failed: {e}")

        return self.current_state

    def load_snapshot(self, run_id: str) -> Optional[RunState]:
        """Load state from a previous run (legacy method, use load_and_recover_state).

        This method is kept for backward compatibility but should not be used
        for new code. Use load_and_recover_state for full recovery.
        """
        snapshot_path = self.state_dir / run_id / "state.json"

        if not snapshot_path.exists():
            logger.error(f"No snapshot found for run {run_id}")
            return None

        try:
            with open(snapshot_path) as f:
                data = json.load(f)

            self.current_state = RunState.from_dict(data)
            logger.info(f"Loaded state for run {run_id}")
            return self.current_state

        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load state for run {run_id}: {e}")
            return None

    def register_instance(
        self,
        instance_id: str,
        strategy_name: str,
        prompt: str,
        base_branch: str,
        branch_name: str,
        container_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new instance."""
        if not self.current_state:
            raise RuntimeError("No active run state")

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

        self.current_state.instances[instance_id] = info
        self.current_state.total_instances += 1

        # Emit state change event
        if self.event_bus:
            self.event_bus.emit(
                "state.instance_registered",
                {
                    "instance_id": instance_id,
                    "strategy_name": strategy_name,
                    "branch_name": branch_name,
                    "container_name": container_name,
                    "state": InstanceStatus.QUEUED.value,
                },
                instance_id=instance_id,
            )

        asyncio.create_task(self._maybe_snapshot())

    def register_task(self, key: str, fingerprint: str, canonical_input: Optional[str] = None) -> None:
        """Register a durable task fingerprint, enforcing no conflicts.

        Stores fingerprint and the normalized input representation when provided.
        """
        if not self.current_state:
            raise RuntimeError("No active run state")
        existing = self.current_state.tasks.get(key)
        if existing and existing.get("fingerprint") != fingerprint:
            raise ValueError(f"KeyConflictDifferentFingerprint for key {key}")
        entry = {"fingerprint": fingerprint}
        if canonical_input is not None:
            entry["input"] = canonical_input
        self.current_state.tasks[key] = entry
        if self.event_bus:
            self.event_bus.emit(
                "state.task_registered",
                {"key": key, "fingerprint": fingerprint},
            )

    def update_instance_state(
        self,
        instance_id: str,
        state: InstanceStatus,
        result: Optional[InstanceResult] = None,
    ) -> None:
        """Update instance state."""
        if not self.current_state:
            raise RuntimeError("No active run state")

        info = self.current_state.instances.get(instance_id)
        if not info:
            logger.warning(f"Unknown instance: {instance_id}")
            return

        old_state = info.state
        info.state = state

        if state == InstanceStatus.RUNNING and not info.started_at:
            info.started_at = datetime.now(timezone.utc)

        elif state == InstanceStatus.INTERRUPTED:
            # Mark interruption without completing
            info.interrupted_at = datetime.now(timezone.utc)

        elif state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
            # Only set completed_at once
            if not info.completed_at:
                info.completed_at = datetime.now(timezone.utc)
            info.result = result
            # Persist session_id from result when available
            if result and result.session_id:
                info.session_id = result.session_id

            # Idempotent counters: only increment when transitioning from non-terminal
            if state == InstanceStatus.COMPLETED and old_state != InstanceStatus.COMPLETED:
                self.current_state.completed_instances += 1
            elif state == InstanceStatus.FAILED and old_state != InstanceStatus.FAILED:
                self.current_state.failed_instances += 1

            # Update metrics
            if result and result.metrics:
                # Use the correct key names from claude_parser
                cost = result.metrics.get("total_cost", 0.0)
                tokens = result.metrics.get("total_tokens", 0)
                self.current_state.total_cost += cost
                self.current_state.total_tokens += tokens

        # Emit state change event
        if self.event_bus:
            event_data = {
                "instance_id": instance_id,
                "old_state": old_state.value,
                "new_state": state.value,
            }

            if state == InstanceStatus.RUNNING:
                event_data["started_at"] = info.started_at.isoformat()
            elif state == InstanceStatus.INTERRUPTED:
                event_data["interrupted_at"] = (
                    info.interrupted_at.isoformat() if info.interrupted_at else datetime.now(timezone.utc).isoformat()
                )
            elif state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
                event_data["completed_at"] = info.completed_at.isoformat()
                if result:
                    event_data["branch_name"] = result.branch_name
                    event_data["cost"] = result.metrics.get("total_cost", 0.0)
                    event_data["tokens"] = result.metrics.get("total_tokens", 0)

            self.event_bus.emit(
                "state.instance_updated", event_data, instance_id=instance_id
            )

        asyncio.create_task(self._maybe_snapshot())

    def update_instance_session_id(self, instance_id: str, session_id: Optional[str]) -> None:
        """Update the stored session_id for a running instance."""
        if not self.current_state:
            return
        info = self.current_state.instances.get(instance_id)
        if not info:
            return
        info.session_id = session_id
        if self.event_bus:
            self.event_bus.emit(
                "state.instance_updated",
                {"instance_id": instance_id, "session_id": session_id, "new_state": info.state.value, "old_state": info.state.value},
                instance_id=instance_id,
            )

    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Register a new strategy execution."""
        if not self.current_state:
            raise RuntimeError("No active run state")

        self.current_state.strategies[strategy_id] = StrategyExecution(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            config=config,
        )

        # Emit state change event
        if self.event_bus:
            self.event_bus.emit(
                "state.strategy_registered",
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "config": config,
                },
            )

        asyncio.create_task(self._maybe_snapshot())

    def update_strategy_state(
        self,
        strategy_id: str,
        state: str,
        results: Optional[List[InstanceResult]] = None,
    ) -> None:
        """Update strategy state."""
        if not self.current_state:
            raise RuntimeError("No active run state")

        strategy = self.current_state.strategies.get(strategy_id)
        if not strategy:
            logger.warning(f"Unknown strategy: {strategy_id}")
            return

        old_state = strategy.state
        strategy.state = state

        if state in ("completed", "failed"):
            strategy.completed_at = datetime.now(timezone.utc)
            if results:
                strategy.results = results

        # Emit state change event
        if self.event_bus:
            event_data = {
                "strategy_id": strategy_id,
                "old_state": old_state,
                "new_state": state,
            }

            if state in ("completed", "failed"):
                event_data["completed_at"] = strategy.completed_at.isoformat()
                if results:
                    event_data["result_count"] = len(results)
                    event_data["branch_names"] = [
                        r.branch_name for r in results if r.branch_name
                    ]

            self.event_bus.emit("state.strategy_updated", event_data)

        asyncio.create_task(self._maybe_snapshot())

    def get_current_state(self) -> Optional[RunState]:
        """Get current run state."""
        return self.current_state

    async def save_snapshot(self) -> None:
        """Save current state snapshot."""
        if not self.current_state:
            return

        try:
            # Ensure last_event_offset reflects current event bus position for recovery
            if self.event_bus and getattr(self.event_bus, "_current_offset", None) is not None:
                try:
                    self.current_state.last_event_offset = int(self.event_bus._current_offset)
                except Exception:
                    pass
            run_dir = self.state_dir / self.current_state.run_id
            await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)

            snapshot_path = run_dir / "state.json"
            temp_path = snapshot_path.with_suffix(".tmp")

            # Write to temp file first
            state_data = self.current_state.to_dict()
            await asyncio.to_thread(self._write_json, temp_path, state_data)

            # Atomic rename - check if temp file exists before renaming
            if temp_path.exists():
                await asyncio.to_thread(temp_path.rename, snapshot_path)
                self._last_snapshot = datetime.now(timezone.utc)
                logger.debug(
                    f"Saved state snapshot for run {self.current_state.run_id}"
                )
            else:
                logger.warning(f"Temp file {temp_path} does not exist, skipping rename")

            # Also write a duplicate snapshot under the logs/<run_id>/ directory
            try:
                if self.event_bus and getattr(self.event_bus, "persist_path", None):
                    logs_run_dir = self.event_bus.persist_path.parent
                    logs_snapshot = logs_run_dir / "state.json"
                    logs_tmp = logs_snapshot.with_suffix(".tmp")
                    # Write to temp then atomically rename to avoid partial writes
                    await asyncio.to_thread(self._write_json, logs_tmp, state_data)
                    if logs_tmp.exists():
                        await asyncio.to_thread(logs_tmp.rename, logs_snapshot)
            except Exception as e:
                # Non-fatal duplication failure
                logger.debug(
                    f"Failed to duplicate state snapshot in logs directory: {e}"
                )

        except asyncio.CancelledError:
            # Expected during shutdown
            logger.debug("Snapshot save cancelled during shutdown")
            raise
        except Exception as e:
            # Log error but don't crash - state snapshots are for recovery only
            logger.error(f"Failed to save state snapshot: {e}")

    def _write_json(self, path: Path, data: dict) -> None:
        """Helper to write JSON file (for use in thread)."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    async def _maybe_snapshot(self) -> None:
        """Save snapshot if interval has passed."""
        if not self.current_state:
            return

        elapsed = (datetime.now(timezone.utc) - self._last_snapshot).total_seconds()
        if elapsed >= self.snapshot_interval:
            await self.save_snapshot()

    async def start_periodic_snapshots(self) -> None:
        """Start background task for periodic snapshots."""
        if self._snapshot_task:
            return  # Already running

        async def snapshot_loop():
            """Background task that saves snapshots every interval."""
            while True:
                try:
                    await asyncio.sleep(self.snapshot_interval)
                    if self.current_state:
                        await self.save_snapshot()
                        logger.debug(
                            f"Periodic snapshot saved for run {self.current_state.run_id}"
                        )
                except asyncio.CancelledError:
                    break
                except (OSError, json.JSONEncodeError) as e:
                    logger.error(f"Error in periodic snapshot: {e}")

        self._snapshot_task = asyncio.create_task(snapshot_loop())
        logger.info(
            f"Started periodic snapshots every {self.snapshot_interval} seconds"
        )

    async def stop_periodic_snapshots(self) -> None:
        """Stop the periodic snapshot task."""
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
            self._snapshot_task = None
            logger.info("Stopped periodic snapshots")

    def apply_event(self, event: Dict[str, Any]) -> None:
        """Apply a single event to reconstruct state.

        This method is used during recovery to rebuild state from the event log.
        It handles all state-modifying events emitted by this component.
        """
        if not self.current_state:
            logger.warning("No current state to apply event to")
            return

        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "state.run_initialized":
            # This would typically create a new state, but during replay
            # the state should already be loaded from snapshot
            pass

        elif event_type == "state.instance_registered":
            instance_id = data.get("instance_id")
            if instance_id and instance_id not in self.current_state.instances:
                info = InstanceInfo(
                    instance_id=instance_id,
                    strategy_name=data.get("strategy_name", ""),
                    prompt=data.get("prompt", ""),
                    base_branch=data.get("base_branch", ""),
                    branch_name=data.get("branch_name", ""),
                    container_name=data.get("container_name", ""),
                    state=InstanceStatus(data.get("state", "queued")),
                    metadata=data.get("metadata", {}),
                )
                self.current_state.instances[instance_id] = info
                self.current_state.total_instances += 1

        elif event_type == "state.instance_updated":
            instance_id = data.get("instance_id")
            if instance_id and instance_id in self.current_state.instances:
                info = self.current_state.instances[instance_id]
                new_state = InstanceStatus(data.get("new_state"))
                old_state = info.state

                info.state = new_state

                # Update session_id if provided in event (e.g., during runtime)
                if "session_id" in data and data.get("session_id"):
                    info.session_id = data.get("session_id")

                if new_state == InstanceStatus.RUNNING and data.get("started_at"):
                    info.started_at = datetime.fromisoformat(data["started_at"])

                elif new_state == InstanceStatus.INTERRUPTED:
                    # Record interruption time if provided
                    ts = data.get("interrupted_at")
                    if ts:
                        try:
                            info.interrupted_at = datetime.fromisoformat(ts)
                        except Exception:
                            pass

                elif new_state in (InstanceStatus.COMPLETED, InstanceStatus.FAILED):
                    if data.get("completed_at"):
                        info.completed_at = datetime.fromisoformat(data["completed_at"])

                    if new_state == InstanceStatus.COMPLETED:
                        if old_state != InstanceStatus.COMPLETED:
                            self.current_state.completed_instances += 1
                    else:
                        if old_state != InstanceStatus.FAILED:
                            self.current_state.failed_instances += 1

                    # Update metrics
                    cost = data.get("cost", 0.0)
                    tokens = data.get("tokens", 0)
                    if cost or tokens:
                        self.current_state.total_cost += cost
                        self.current_state.total_tokens += tokens

        elif event_type == "state.strategy_registered":
            strategy_id = data.get("strategy_id")
            if strategy_id and strategy_id not in self.current_state.strategies:
                self.current_state.strategies[strategy_id] = StrategyExecution(
                    strategy_id=strategy_id,
                    strategy_name=data.get("strategy_name", ""),
                    config=data.get("config", {}),
                )

        elif event_type == "state.strategy_updated":
            strategy_id = data.get("strategy_id")
            if strategy_id and strategy_id in self.current_state.strategies:
                strategy = self.current_state.strategies[strategy_id]
                strategy.state = data.get("new_state", "running")

                if data.get("completed_at"):
                    strategy.completed_at = datetime.fromisoformat(data["completed_at"])

                # Note: Strategy results aren't included in events to keep them small
                # Full results are stored in the final snapshot

    async def rebuild_from_events(self, events: List[Dict[str, Any]]) -> None:
        """Rebuild state by replaying a list of events.

        This is used during recovery to reconstruct state from the event log
        after loading a snapshot.
        """
        if not self.current_state:
            logger.error("No base state to rebuild from")
            return

        logger.info(f"Replaying {len(events)} events to rebuild state")

        for event in events:
            try:
                self.apply_event(event)
            except Exception as e:
                logger.error(f"Error applying event {event.get('type')}: {e}")
                # Continue with other events - partial recovery is better than none

        logger.info("State reconstruction from events completed")
