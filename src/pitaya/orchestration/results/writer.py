"""Persist run results to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pitaya.shared import InstanceResult, InstanceStatus
from pitaya.orchestration.results.utils import (
    compute_counts,
    persist_instance_metadata,
    resolve_instance_id,
    safe_dt,
    write_json,
    write_metrics_csv,
    write_summary_md,
)

logger = logging.getLogger(__name__)


def _build_summary(
    state,
    run_id: str,
    any_interrupted: bool,
    total_instances: int,
    completed_instances: int,
    failed_instances: int,
    completed_at,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": ("interrupted" if any_interrupted else "completed"),
        "started_at": state.started_at.isoformat(),
        "completed_at": (completed_at.isoformat() if completed_at else None),
        "duration_seconds": (
            (completed_at - state.started_at).total_seconds()
            if completed_at and state.started_at
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


def _build_meta(
    state, summary_data, run_logs_dir, results_dir, orchestrator
) -> Dict[str, Any]:
    return {
        "run_id": summary_data["run_id"],
        "started_at": summary_data["started_at"],
        "completed_at": summary_data["completed_at"],
        "status": summary_data["status"],
        "prompt": state.prompt,
        "repo_path": str(state.repo_path),
        "base_branch": state.base_branch,
        "orchestrator": {
            "max_parallel_instances": orchestrator.max_parallel_instances,
            "max_parallel_startup": orchestrator.max_parallel_startup,
            "branch_namespace": getattr(
                orchestrator, "branch_namespace", "hierarchical"
            ),
            "randomize_queue_order": getattr(
                orchestrator, "randomize_queue_order", False
            ),
            "snapshot_interval": orchestrator.snapshot_interval,
            "event_buffer_size": orchestrator.event_buffer_size,
        },
        "runner": {
            "timeout_seconds": orchestrator.runner_timeout_seconds,
            "default_network_egress": orchestrator.default_network_egress,
            "container_limits": {
                "cpu_count": getattr(orchestrator.container_limits, "cpu_count", None),
                "memory_gb": getattr(orchestrator.container_limits, "memory_gb", None),
            },
            "force_commit": getattr(orchestrator, "force_commit", False),
        },
        "defaults": {
            "plugin_name": getattr(orchestrator, "default_plugin_name", None),
            "model_alias": getattr(orchestrator, "default_model_alias", None),
            "docker_image": getattr(orchestrator, "default_docker_image", None),
            "agent_cli_args": getattr(orchestrator, "default_agent_cli_args", None),
        },
        "strategies": {
            sid: {
                "name": sinfo.strategy_name,
                "config": sinfo.config,
                "state": sinfo.state,
            }
            for sid, sinfo in state.strategies.items()
        },
        "paths": {
            "logs_dir": str(run_logs_dir),
            "results_dir": str(results_dir),
            "events_file": str(run_logs_dir / "events.jsonl"),
        },
        "metrics": {
            "total_cost": state.total_cost,
            "total_tokens": state.total_tokens,
            "completed_instances": summary_data["completed_instances"],
            "failed_instances": summary_data["failed_instances"],
            "total_instances": summary_data["total_instances"],
        },
    }


def _append_strategy_info(state, summary_data) -> None:
    for strat_id, strat_info in state.strategies.items():
        summary_data["strategies"][strat_id] = {
            "name": strat_info.strategy_name,
            "state": strat_info.state,
            "config": strat_info.config,
            "started_at": strat_info.started_at.isoformat(),
            "completed_at": (
                strat_info.completed_at.isoformat() if strat_info.completed_at else None
            ),
            "result_count": len(strat_info.results) if strat_info.results else 0,
        }


def _append_results(results, state, summary_data, branches: List[str]) -> None:
    for result in results:
        commit_stats = result.commit_statistics or {}
        instance_id = resolve_instance_id(result, state)
        summary_data["results"].append(
            {
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
                "metadata": getattr(result, "metadata", None) or {},
            }
        )
        if result.branch_name:
            branches.append(result.branch_name)


async def save_results(
    orchestrator, run_id: str, results: List[InstanceResult]
) -> None:
    """Persist run outputs to configured results and logs directories."""
    results_dir = orchestrator.results_dir / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = orchestrator.logs_dir / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    state = orchestrator.state_manager.current_state
    if not state:
        logger.warning("No state found for run %s, skipping results save", run_id)
        return

    try:
        any_interrupted = any(
            i.state == InstanceStatus.INTERRUPTED for i in state.instances.values()
        )
    except Exception:
        any_interrupted = False
    total_instances, completed_instances, failed_instances = compute_counts(state)

    computed_completed_at = safe_dt(state.completed_at)
    if not computed_completed_at:
        times = [
            safe_dt(getattr(r, "completed_at", None))
            for r in results
            if getattr(r, "completed_at", None)
        ]
        times = [t for t in times if t is not None]
        computed_completed_at = max(times) if times else None

    summary_data = _build_summary(
        state,
        run_id,
        any_interrupted,
        total_instances,
        completed_instances,
        failed_instances,
        computed_completed_at,
    )
    meta = _build_meta(state, summary_data, run_logs_dir, results_dir, orchestrator)
    _append_strategy_info(state, summary_data)

    branches: List[str] = []
    _append_results(results, state, summary_data, branches)

    write_json(results_dir / "summary.json", summary_data)
    if branches:
        with open(results_dir / "branches.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(branches) + "\n")

    metrics_path = results_dir / "instance_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "instance_id,branch_name,status,duration_seconds,cost,tokens,input_tokens,output_tokens,commit_count,lines_added,lines_deleted,has_changes\n"
        )
        for result in results:
            commit_stats = result.commit_statistics or {}
            instance_id = resolve_instance_id(result, state)
            row = [
                instance_id[:8] if instance_id != "unknown" else "unknown",
                result.branch_name or "",
                result.status,
                f"{result.duration_seconds:.1f}" if result.duration_seconds else "0",
                (
                    f"{result.metrics.get('total_cost', 0):.2f}"
                    if result.metrics
                    else "0.00"
                ),
                str(result.metrics.get("total_tokens", 0) if result.metrics else 0),
                str(result.metrics.get("input_tokens", 0) if result.metrics else 0),
                str(result.metrics.get("output_tokens", 0) if result.metrics else 0),
                str(commit_stats.get("commit_count", 0)),
                str(commit_stats.get("insertions", 0)),
                str(commit_stats.get("deletions", 0)),
                "yes" if result.has_changes else "no",
            ]
            f.write(",".join(row) + "\n")

    write_metrics_csv(run_logs_dir / "events.jsonl", results_dir / "metrics.csv")

    strategy_dir = results_dir / "strategy_output"
    strategy_dir.mkdir(exist_ok=True)
    for strat_id, strat_info in state.strategies.items():
        if strat_info.results:
            strategy_file = strategy_dir / f"{strat_info.strategy_name}_{strat_id}.json"
            strategy_data = {
                "strategy_id": strat_id,
                "strategy_name": strat_info.strategy_name,
                "config": strat_info.config,
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
                        "metadata": (
                            getattr(r, "metadata", None)
                            if not isinstance(r, dict)
                            else r.get("metadata")
                        ),
                    }
                    for r in (strat_info.results or [])
                ],
            }
            write_json(strategy_file, strategy_data)

    inst_dir_results = results_dir / "instances"
    inst_dir_logs = run_logs_dir / "instances"
    inst_dir_results.mkdir(exist_ok=True)
    inst_dir_logs.mkdir(exist_ok=True)
    instance_file_names: List[str] = []
    unknown_counter = 0
    for result in results:
        iid = resolve_instance_id(result, state)
        inst_meta_raw = {
            "instance_id": iid,
            "status": result.status,
            "success": result.success,
            "branch_name": result.branch_name,
            "final_message": getattr(result, "final_message", None),
            "metadata": getattr(result, "metadata", None) or {},
            "duration_seconds": result.duration_seconds,
            "metrics": result.metrics or {},
            "error": result.error,
            "container_name": result.container_name,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
        }
        base_name = (iid or "unknown").replace("/", "_")
        if base_name == "unknown":
            unknown_counter += 1
            base_name = f"idx_{unknown_counter:03d}"
        instance_file_names.append(f"{base_name}.json")
        persist_instance_metadata(
            orchestrator,
            inst_meta_raw,
            inst_dir_results,
            inst_dir_logs,
            base_name,
        )

    meta_with_index = dict(meta)
    meta_with_index["instances_dir"] = str(results_dir / "instances")
    meta_with_index["instances"] = instance_file_names
    mdj = json.dumps(meta_with_index, indent=2, default=str)
    with open(results_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(mdj)
    with open(run_logs_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(mdj)

    write_summary_md(results_dir / "summary.md", state, branches, results)
    logger.info("Saved results to %s", results_dir)
