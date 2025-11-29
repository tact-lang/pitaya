"""Dataclasses for orchestration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..shared import InstanceResult, InstanceStatus


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
    aggregated_tokens: int = 0
    aggregated_cost: float = 0.0


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
    total_instances: int = 0


@dataclass
class RunState:
    """Complete state of an orchestration run."""

    run_id: str
    prompt: str
    repo_path: Path
    base_branch: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    instances: Dict[str, InstanceInfo] = field(default_factory=dict)
    strategies: Dict[str, StrategyExecution] = field(default_factory=dict)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_instances: int = 0
    completed_instances: int = 0
    failed_instances: int = 0
    last_event_start_offset: int = 0
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategy_rand: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
            "last_event_start_offset": self.last_event_start_offset,
            "instances": {
                iid: _instance_to_dict(info) for iid, info in self.instances.items()
            },
            "strategies": {
                sid: _strategy_to_dict(strat) for sid, strat in self.strategies.items()
            },
            "tasks": self.tasks,
            "strategy_rand": self.strategy_rand,
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
            last_event_start_offset=data.get("last_event_start_offset", 0),
        )
        for iid, info_data in data.get("instances", {}).items():
            state.instances[iid] = _instance_from_dict(info_data)
        for sid, strat_data in data.get("strategies", {}).items():
            state.strategies[sid] = _strategy_from_dict(strat_data)
        if "tasks" in data:
            state.tasks = data.get("tasks", {})
        if "strategy_rand" in data:
            state.strategy_rand = data.get("strategy_rand", {})
        return state


def _instance_to_dict(info: InstanceInfo) -> Dict[str, Any]:
    return {
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
        "started_at": info.started_at.isoformat() if info.started_at else None,
        "completed_at": info.completed_at.isoformat() if info.completed_at else None,
        "interrupted_at": (
            info.interrupted_at.isoformat() if info.interrupted_at else None
        ),
        "result": _result_to_dict(info.result) if info.result else None,
    }


def _instance_from_dict(info_data: Dict[str, Any]) -> InstanceInfo:
    result_data = info_data.get("result")
    result_obj = None
    if result_data:
        try:
            result_obj = InstanceResult(
                success=bool(result_data.get("success")),
                branch_name=result_data.get("branch_name"),
                has_changes=bool(result_data.get("has_changes")),
                metrics=result_data.get("metrics"),
                status=result_data.get("status"),
                session_id=result_data.get("session_id"),
            )
        except Exception:
            result_obj = None

    return InstanceInfo(
        instance_id=info_data["instance_id"],
        strategy_name=info_data["strategy_name"],
        prompt=info_data["prompt"],
        base_branch=info_data["base_branch"],
        branch_name=info_data["branch_name"],
        container_name=info_data["container_name"],
        state=InstanceStatus(info_data["state"]),
        metadata=info_data.get("metadata", {}),
        created_at=datetime.fromisoformat(info_data["created_at"]),
        started_at=(
            datetime.fromisoformat(info_data["started_at"])
            if info_data.get("started_at")
            else None
        ),
        completed_at=(
            datetime.fromisoformat(info_data["completed_at"])
            if info_data.get("completed_at")
            else None
        ),
        interrupted_at=(
            datetime.fromisoformat(info_data["interrupted_at"])
            if info_data.get("interrupted_at")
            else None
        ),
        result=result_obj,
        session_id=info_data.get("session_id"),
    )


def _strategy_to_dict(strat: StrategyExecution) -> Dict[str, Any]:
    return {
        "strategy_id": strat.strategy_id,
        "strategy_name": strat.strategy_name,
        "config": strat.config,
        "instance_ids": strat.instance_ids,
        "state": strat.state,
        "started_at": strat.started_at.isoformat(),
        "completed_at": strat.completed_at.isoformat() if strat.completed_at else None,
        "total_instances": getattr(strat, "total_instances", 0),
    }


def _strategy_from_dict(strat_data: Dict[str, Any]) -> StrategyExecution:
    return StrategyExecution(
        strategy_id=strat_data["strategy_id"],
        strategy_name=strat_data["strategy_name"],
        config=strat_data.get("config", {}),
        instance_ids=strat_data.get("instance_ids", []),
        state=strat_data.get("state", "running"),
        started_at=datetime.fromisoformat(strat_data["started_at"]),
        completed_at=(
            datetime.fromisoformat(strat_data["completed_at"])
            if strat_data.get("completed_at")
            else None
        ),
        total_instances=strat_data.get("total_instances", 0),
    )


def _result_to_dict(res: InstanceResult) -> Dict[str, Any]:
    return {
        "success": bool(res.success),
        "branch_name": res.branch_name,
        "has_changes": bool(res.has_changes),
        "metrics": res.metrics or {},
        "status": res.status,
        "session_id": res.session_id,
    }
