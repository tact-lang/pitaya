"""StateManager coordinating snapshots and event replay."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..shared import InstanceResult, InstanceStatus
from .state_models import RunState
from .state_rebuilder import apply_event, rebuild_from_events
from .state_snapshot import (
    maybe_snapshot,
    save_snapshot,
    start_periodic_snapshots,
    stop_periodic_snapshots,
)
from .state_updates import (
    register_instance,
    register_strategy,
    register_task,
    update_instance_container_name,
    update_instance_metadata,
    update_instance_session_id,
    update_instance_state,
    update_strategy_rand,
    update_strategy_state,
)

logger = logging.getLogger(__name__)


class StateManager:
    """Manages orchestration state with persistence and recovery."""

    def __init__(
        self,
        state_dir: Path,
        event_bus: Optional[Any] = None,
        snapshot_interval: int = 30,
    ):
        self.state_dir = state_dir
        self.event_bus = event_bus
        self.snapshot_interval = snapshot_interval
        self.current_state: Optional[RunState] = None
        self._last_snapshot = datetime.now(timezone.utc)
        self._snapshot_task: Optional[asyncio.Task] = None
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Run lifecycle
    # ------------------------------------------------------------------ #
    def initialize_run(
        self, run_id: str, prompt: str, repo_path: Path, base_branch: str
    ) -> RunState:
        self.current_state = RunState(
            run_id=run_id,
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
        )
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
        asyncio.create_task(save_snapshot(self))
        return self.current_state

    async def load_and_recover_state(self, run_id: str) -> Optional[RunState]:
        snapshot_path = self.state_dir / run_id / "state.json"
        logs_snapshot_path = None
        if self.event_bus and getattr(self.event_bus, "persist_path", None):
            logs_snapshot_path = self.event_bus.persist_path.parent / "state.json"
        candidate_paths = [p for p in [logs_snapshot_path, snapshot_path] if p]

        loaded = False
        last_offset = 0
        for path in candidate_paths:
            if path.exists():
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    self.current_state = RunState.from_dict(data)
                    last_offset = self.current_state.last_event_start_offset
                    logger.info(
                        "Loaded state snapshot for run %s from %s at offset %s",
                        run_id,
                        path,
                        last_offset,
                    )
                    loaded = True
                    break
                except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
                    logger.error(
                        "Failed to load state for run %s from %s: %s", run_id, path, exc
                    )

        if not loaded:
            logger.warning(
                "No snapshot found for run %s, attempting recovery from events only",
                run_id,
            )
            self.current_state = RunState(
                run_id=run_id,
                prompt="",
                repo_path=Path("."),
                base_branch="main",
            )

        if self.event_bus:
            try:
                all_events, next_offset = self.event_bus.get_events_since(
                    offset=last_offset,
                    event_types=None,
                    limit=None,
                )
                if all_events:
                    rebuild_from_events(self.current_state, all_events)
                    self.current_state.last_event_start_offset = next_offset
                    logger.info("Applied %s events to recover state", len(all_events))
            except Exception as exc:
                logger.error("Failed to recover events for run %s: %s", run_id, exc)
        return self.current_state

    # ------------------------------------------------------------------ #
    # Instance helpers
    # ------------------------------------------------------------------ #
    def register_instance(
        self,
        *,
        instance_id: str,
        strategy_name: str,
        prompt: str,
        base_branch: str,
        branch_name: str,
        container_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        register_instance(
            self,
            instance_id=instance_id,
            strategy_name=strategy_name,
            prompt=prompt,
            base_branch=base_branch,
            branch_name=branch_name,
            container_name=container_name,
            metadata=metadata,
        )
        asyncio.create_task(maybe_snapshot(self))

    def register_task(
        self, key: str, fingerprint: str, canonical_input: Optional[str] = None
    ) -> None:
        register_task(self, key, fingerprint, canonical_input)

    def update_instance_state(
        self,
        instance_id: str,
        state: InstanceStatus,
        result: Optional[InstanceResult] = None,
    ) -> None:
        update_instance_state(self, instance_id, state, result)
        asyncio.create_task(maybe_snapshot(self))

    def update_instance_session_id(
        self, instance_id: str, session_id: Optional[str]
    ) -> None:
        update_instance_session_id(self, instance_id, session_id)

    def update_instance_metadata(self, instance_id: str, patch: Dict[str, Any]) -> None:
        update_instance_metadata(self, instance_id, patch)

    def update_instance_container_name(
        self, instance_id: str, container_name: str
    ) -> None:
        update_instance_container_name(self, instance_id, container_name)

    # ------------------------------------------------------------------ #
    # Strategy helpers
    # ------------------------------------------------------------------ #
    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        config: Dict[str, Any],
    ) -> None:
        register_strategy(self, strategy_id, strategy_name, config)
        asyncio.create_task(maybe_snapshot(self))

    def update_strategy_state(
        self,
        strategy_id: str,
        state: str,
        results: Optional[List[InstanceResult]] = None,
    ) -> None:
        update_strategy_state(self, strategy_id, state, results)
        asyncio.create_task(maybe_snapshot(self))

    def update_strategy_rand(self, strategy_id: str, seq: int, value: float) -> None:
        update_strategy_rand(self, strategy_id, seq, value)
        asyncio.create_task(maybe_snapshot(self))

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #
    def get_current_state(self) -> Optional[RunState]:
        return self.current_state

    async def save_snapshot(self) -> None:
        await save_snapshot(self)

    async def start_periodic_snapshots(self) -> None:
        await start_periodic_snapshots(self)

    async def stop_periodic_snapshots(self) -> None:
        await stop_periodic_snapshots(self)

    # ------------------------------------------------------------------ #
    # Event replay
    # ------------------------------------------------------------------ #
    async def rebuild_from_events(self, events: List[Dict[str, Any]]) -> None:
        if not self.current_state:
            return
        rebuild_from_events(self.current_state, events)

    def apply_event(self, event: Dict[str, Any]) -> None:
        if not self.current_state:
            return
        apply_event(self.current_state, event)
