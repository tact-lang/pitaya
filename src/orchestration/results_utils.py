"""Shared helpers for writing result artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def safe_dt(value: Any) -> datetime | None:
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


def compute_counts(state) -> tuple[int, int, int]:
    try:
        instance_list = list(state.instances.values())
        total = len(instance_list)
        completed = sum(1 for i in instance_list if i.state.name.lower() == "completed")
        failed = sum(1 for i in instance_list if i.state.name.lower() == "failed")
        return total, completed, failed
    except Exception:
        return state.total_instances, state.completed_instances, state.failed_instances


def resolve_instance_id(result, state) -> str:
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


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def write_summary_md(path: Path, state, branches: List[str], results) -> None:
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


def write_metrics_csv(events_file: Path, dest: Path) -> None:
    try:
        running: set[str] = set()
        completed_set: set[str] = set()
        failed_set: set[str] = set()
        inst_tokens: Dict[str, int] = {}
        inst_cost: Dict[str, float] = {}
        if not events_file.exists():
            return
        with (
            open(events_file, "r", encoding="utf-8") as ef,
            open(dest, "w", encoding="utf-8") as tf,
        ):
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
    except Exception as exc:
        logger.debug("Failed generating time-series metrics: %s", exc)


def persist_instance_metadata(
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
