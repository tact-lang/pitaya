"""Persist run results to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..shared import InstanceResult, InstanceStatus

logger = logging.getLogger(__name__)


def _safe_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _compute_counts(state) -> tuple[int, int, int]:
    try:
        instance_list = list(state.instances.values())
        total = len(instance_list)
        completed = sum(1 for i in instance_list if i.state == InstanceStatus.COMPLETED)
        failed = sum(1 for i in instance_list if i.state == InstanceStatus.FAILED)
        return total, completed, failed
    except Exception:
        return state.total_instances, state.completed_instances, state.failed_instances


def _resolve_instance_id(result, state) -> str:
    result_to_instance: Dict[int, str] = {}
    by_branch_session: Dict[tuple, str] = {}
    by_container: Dict[str, str] = {}
    by_started: Dict[str, str] = {}
    for instance_id, info in state.instances.items():
        if info.result:
            result_to_instance[id(info.result)] = instance_id
            bn = getattr(info.result, "branch_name", None)
            sid = getattr(info.result, "session_id", None)
            if bn or sid:
                by_branch_session[(bn, sid)] = instance_id
            cn = getattr(info, "container_name", None)
            if cn:
                by_container[cn] = instance_id
            st = getattr(info.result, "started_at", None)
            if st:
                by_started[str(st)] = instance_id

    iid = result_to_instance.get(id(result))
    if iid:
        return iid
    key = (getattr(result, "branch_name", None), getattr(result, "session_id", None))
    if key in by_branch_session:
        return by_branch_session[key]
    cn = getattr(result, "container_name", None)
    if cn and cn in by_container:
        return by_container[cn]
    st = getattr(result, "started_at", None)
    if st and str(st) in by_started:
        return by_started[str(st)]
    return "unknown"


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_summary_md(path: Path, state, branches: List[str], results: List[InstanceResult]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Pitaya Run Summary: {state.run_id}\n\n")
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
                    dur = f"{r.duration_seconds:.0f}s" if r.duration_seconds else "-"
                    cost = r.metrics.get("total_cost", 0.0) if r.metrics else 0.0
                    tokens = r.metrics.get("total_tokens", 0) if r.metrics else 0
                    f.write(
                        f"- {status} {r.branch_name or 'no-branch'} • {dur} • ${cost:.2f} • {tokens} tokens\n"
                    )
    except (OSError, IOError, ValueError):
        pass


def _write_metrics_csv(events_file: Path, dest: Path) -> None:
    try:
        running: set[str] = set()
        completed_set: set[str] = set()
        failed_set: set[str] = set()
        inst_tokens: Dict[str, int] = {}
        inst_cost: Dict[str, float] = {}
        if not events_file.exists():
            return
        with open(events_file, "r", encoding="utf-8") as ef, open(dest, "w", encoding="utf-8") as tf:
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
                    inst_tokens[iid] = inst_tokens.get(iid, 0) + int(tm.get("tokens", 0) or 0)
                    inst_cost[iid] = inst_cost.get(iid, 0.0) + float(tm.get("cost", 0.0) or 0.0)
                elif et == "instance.agent_completed" and iid:
                    m = data.get("metrics", {})
                    if m:
                        inst_tokens[iid] = int(m.get("total_tokens", inst_tokens.get(iid, 0)))
                        inst_cost[iid] = float(m.get("total_cost", inst_cost.get(iid, 0.0)))
                total_cost = sum(inst_cost.values())
                total_tokens = sum(inst_tokens.values())
                tf.write(
                    f"{ts},{len(running)},{len(completed_set)},{len(failed_set)},{total_cost:.4f},{total_tokens},{et}\n"
                )
    except Exception as exc:
        logger.debug("Failed generating time-series metrics: %s", exc)


def _persist_instance_metadata(
    orchestrator,
    inst_meta_raw: Dict[str, Any],
    results_dir: Path,
    run_logs_dir: Path,
    base_name: str,
) -> None:
    try:
        if orchestrator.event_bus and hasattr(orchestrator.event_bus, "_sanitize"):
            inst_meta = orchestrator.event_bus._sanitize(inst_meta_raw)  # type: ignore[attr-defined]
            try:
                keep = ("total_tokens", "input_tokens", "output_tokens", "turn_count")
                raw_metrics = inst_meta_raw.get("metrics", {}) or {}
                if isinstance(inst_meta.get("metrics"), dict):
                    for k in keep:
                        if k in raw_metrics:
                            inst_meta["metrics"][k] = raw_metrics.get(k)
            except Exception:
                pass
        else:
            inst_meta = inst_meta_raw
    except Exception:
        inst_meta = inst_meta_raw

    with open(results_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(inst_meta, f, indent=2, default=str)
    with open(run_logs_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(inst_meta, f, indent=2, default=str)


async def save_results(orchestrator, run_id: str, results: List[InstanceResult]) -> None:
    """Persist run outputs to ./results and ./logs."""
    results_dir = Path("./results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = orchestrator.logs_dir / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    state = orchestrator.state_manager.current_state
    if not state:
        logger.warning("No state found for run %s, skipping results save", run_id)
        return

    try:
        any_interrupted = any(i.state == InstanceStatus.INTERRUPTED for i in state.instances.values())
    except Exception:
        any_interrupted = False
    total_instances, completed_instances, failed_instances = _compute_counts(state)

    computed_completed_at = _safe_dt(state.completed_at)
    if not computed_completed_at:
        times = [
            _safe_dt(getattr(r, "completed_at", None))
            for r in results
            if getattr(r, "completed_at", None)
        ]
        times = [t for t in times if t is not None]
        computed_completed_at = max(times) if times else None

    summary_data: Dict[str, Any] = {
        "run_id": run_id,
        "status": ("interrupted" if any_interrupted else "completed"),
        "started_at": state.started_at.isoformat(),
        "completed_at": (computed_completed_at.isoformat() if computed_completed_at else None),
        "duration_seconds": (
            (computed_completed_at - state.started_at).total_seconds()
            if computed_completed_at and isinstance(state.started_at, datetime)
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

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": summary_data["started_at"],
        "completed_at": summary_data["completed_at"],
        "status": summary_data["status"],
        "prompt": state.prompt,
        "repo_path": str(state.repo_path),
        "base_branch": state.base_branch,
        "orchestrator": {
            "max_parallel_instances": orchestrator.max_parallel_instances,
            "max_parallel_startup": orchestrator.max_parallel_startup,
            "branch_namespace": getattr(orchestrator, "branch_namespace", "hierarchical"),
            "randomize_queue_order": getattr(orchestrator, "randomize_queue_order", False),
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
            "completed_instances": completed_instances,
            "failed_instances": failed_instances,
            "total_instances": total_instances,
        },
    }

    for strat_id, strat_info in state.strategies.items():
        summary_data["strategies"][strat_id] = {
            "name": strat_info.strategy_name,
            "state": strat_info.state,
            "config": strat_info.config,
            "started_at": strat_info.started_at.isoformat(),
            "completed_at": strat_info.completed_at.isoformat() if strat_info.completed_at else None,
            "result_count": len(strat_info.results) if strat_info.results else 0,
        }

    branches: List[str] = []
    instance_file_names: List[str] = []
    unknown_counter = 0

    for result in results:
        commit_stats = result.commit_statistics or {}
        instance_id = _resolve_instance_id(result, state)
        result_data = {
            "instance_id": instance_id,
            "session_id": result.session_id,
            "branch_name": result.branch_name,
            "status": result.status,
            "success": result.success,
            "error": result.error,
            "has_changes": result.has_changes,
            "duration_seconds": result.duration_seconds,
            "cost": (result.metrics.get("total_cost", 0.0) if result.metrics else 0.0),
            "tokens": (result.metrics.get("total_tokens", 0) if result.metrics else 0),
            "input_tokens": (result.metrics.get("input_tokens", 0) if result.metrics else 0),
            "output_tokens": (result.metrics.get("output_tokens", 0) if result.metrics else 0),
            "cost_usd": (result.metrics.get("total_cost", 0.0) if result.metrics else 0.0),
            "tokens_in": (result.metrics.get("input_tokens", 0) if result.metrics else 0),
            "tokens_out": (result.metrics.get("output_tokens", 0) if result.metrics else 0),
            "commit_count": commit_stats.get("commit_count", 0),
            "lines_added": commit_stats.get("insertions", 0),
            "lines_deleted": commit_stats.get("deletions", 0),
            "container_name": result.container_name,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "metadata": getattr(result, "metadata", None) or {},
        }
        summary_data["results"].append(result_data)
        if result.branch_name:
            branches.append(result.branch_name)

    _write_json(results_dir / "summary.json", summary_data)
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
            instance_id = _resolve_instance_id(result, state)
            row = [
                instance_id[:8] if instance_id != "unknown" else "unknown",
                result.branch_name or "",
                result.status,
                f"{result.duration_seconds:.1f}" if result.duration_seconds else "0",
                f"{result.metrics.get('total_cost', 0):.2f}" if result.metrics else "0.00",
                str(result.metrics.get("total_tokens", 0) if result.metrics else 0),
                str(result.metrics.get("input_tokens", 0) if result.metrics else 0),
                str(result.metrics.get("output_tokens", 0) if result.metrics else 0),
                str(commit_stats.get("commit_count", 0)),
                str(commit_stats.get("insertions", 0)),
                str(commit_stats.get("deletions", 0)),
                "yes" if result.has_changes else "no",
            ]
            f.write(",".join(row) + "\n")

    _write_metrics_csv(run_logs_dir / "events.jsonl", results_dir / "metrics.csv")

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
                        "branch_name": (getattr(r, "branch_name", None) if not isinstance(r, dict) else r.get("branch_name")),
                        "success": (getattr(r, "success", False) if not isinstance(r, dict) else bool(r.get("success", False))),
                        "metadata": (getattr(r, "metadata", None) if not isinstance(r, dict) else r.get("metadata")),
                    }
                    for r in (strat_info.results or [])
                ],
            }
            _write_json(strategy_file, strategy_data)

    inst_dir_results = results_dir / "instances"
    inst_dir_logs = run_logs_dir / "instances"
    inst_dir_results.mkdir(exist_ok=True)
    inst_dir_logs.mkdir(exist_ok=True)
    for result in results:
        iid = _resolve_instance_id(result, state)
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
        _persist_instance_metadata(
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

    _write_summary_md(results_dir / "summary.md", state, branches, results)
    logger.info("Saved results to %s", results_dir)
