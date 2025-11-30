"""Handlers for task lifecycle events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..models import InstanceDisplay, InstanceStatus


class TaskLifecycleHandlers:
    """Task lifecycle (scheduled/start/complete/fail/interrupted) handlers."""

    def _handle_task_scheduled(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        sid = event.get("strategy_execution_id")
        strategy_name = "unknown"
        if sid and self.state.current_run and sid in self.state.current_run.strategies:
            strategy_name = self.state.current_run.strategies[sid].strategy_name
        inst = self.state.current_run.instances.get(iid)
        if not inst:
            inst = InstanceDisplay(
                instance_id=iid,
                strategy_name=strategy_name,
                status=InstanceStatus.QUEUED,
                branch_name=data.get("branch_name", ""),
            )
            self.state.current_run.instances[iid] = inst
        try:
            m = data.get("model")
            if m:
                inst.model = m
        except Exception:
            pass
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        if sid and self.state.current_run and sid in self.state.current_run.strategies:
            strat = self.state.current_run.strategies[sid]
            if iid not in strat.instance_ids:
                strat.instance_ids.append(iid)
                strat.total_instances += 1
        try:
            if self.state.current_run and not self.state.current_run.started_at:
                ts = event.get("timestamp")
                self.state.current_run.started_at = self._parse_timestamp(ts)
        except Exception:
            pass

    def _handle_task_started(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.RUNNING
            try:
                ts = event.get("timestamp") or event.get("ts")
                inst.started_at = self._parse_timestamp(ts)
            except Exception:
                pass
            try:
                if self.state.current_run and not self.state.current_run.started_at:
                    self.state.current_run.started_at = (
                        inst.started_at or self._parse_timestamp(event.get("timestamp"))
                    )
            except Exception:
                pass
            sid = event.get("strategy_execution_id")
            if (
                sid
                and self.state.current_run
                and sid in self.state.current_run.strategies
            ):
                sname = self.state.current_run.strategies[sid].strategy_name
                if sname and (
                    not inst.strategy_name or inst.strategy_name == "unknown"
                ):
                    inst.strategy_name = sname
                strat = self.state.current_run.strategies[sid]
                if iid not in strat.instance_ids:
                    strat.instance_ids.append(iid)
                    strat.total_instances += 1
            model = (event.get("data", {}) or {}).get("model")
            base = (event.get("data", {}) or {}).get("base_branch")
            parts = ["started"]
            if model:
                parts.append(f"model={model}")
            if base:
                parts.append(f"base={base}")
            self._append_msg(iid, " ".join(parts))
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _handle_task_completed(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            metrics = data.get("metrics", {}) or {}
            artifact = data.get("artifact", {}) or {}
            self._finalize_task_instance_completion(inst, artifact, event)
            self._apply_task_completion_metrics(inst, metrics)
            self._capture_final_message(inst, data)
            self._append_task_completed_message(iid, metrics, artifact, data)
        self.state.current_run.total_instances = len(self.state.current_run.instances)
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass

    def _handle_task_failed(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.FAILED
            inst.error = data.get("message")
        etype = data.get("error_type")
        msg = (data.get("message") or "").strip().replace("\n", " ")
        parts = ["failed"]
        if etype:
            parts.append(f"type={etype}")
        if msg:
            parts.append(f'message="{msg[:80]}"')
        if str(data.get("network_egress")) == "offline":
            parts.append("hint=egress=offline:set runner.network_egress=online/proxy")
        self._append_msg(iid, " ".join(parts))
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass
        self.state.current_run.total_instances = len(self.state.current_run.instances)

    def _handle_task_interrupted(self, event: Dict[str, Any]) -> None:
        self._ensure_current_run()
        data = event.get("data", {})
        iid = data.get("instance_id")
        if not iid:
            return
        inst = self.state.current_run.instances.get(iid)
        if inst:
            inst.status = InstanceStatus.INTERRUPTED
            inst.last_updated = datetime.now()
        self._append_msg(iid, "interrupted")
        try:
            self.state.last_updated_instance_id = iid
        except Exception:
            pass
            self.state.current_run.active_instances = max(
                0, self.state.current_run.active_instances - 1
            )
        self.state.current_run.total_instances = len(self.state.current_run.instances)

    def _finalize_task_instance_completion(
        self,
        instance: InstanceDisplay,
        artifact: Dict[str, Any],
        event: Dict[str, Any],
    ) -> None:
        instance.status = InstanceStatus.COMPLETED
        if artifact.get("branch_final"):
            instance.branch_name = artifact.get("branch_final")
        try:
            ts = event.get("timestamp") or event.get("ts")
            instance.completed_at = self._parse_timestamp(ts) or instance.completed_at
            if instance.started_at and instance.completed_at:
                start = instance.started_at
                end = instance.completed_at
                try:
                    if end.tzinfo != start.tzinfo:
                        end = end.replace(tzinfo=start.tzinfo)
                except Exception:
                    pass
                instance.duration_seconds = max(0.0, (end - start).total_seconds())
        except Exception:
            pass

    def _apply_task_completion_metrics(
        self, instance: InstanceDisplay, metrics: Dict[str, Any]
    ) -> None:
        if not metrics:
            return
        try:
            prev_total = instance.total_tokens
            tc = metrics.get("total_cost")
            instance.cost = float(tc) if isinstance(tc, (int, float)) else instance.cost
        except Exception:
            pass

        total_tokens_val: Optional[int] = None
        try:
            tt = metrics.get("total_tokens")
            if isinstance(tt, (int, float)):
                total_tokens_val = int(tt)
                instance.total_tokens = total_tokens_val
                instance.usage_running_total = max(
                    instance.usage_running_total, total_tokens_val
                )
        except Exception:
            pass

        input_raw: Optional[int] = None
        output_raw: Optional[int] = None
        try:
            it = metrics.get("input_tokens")
            ot = metrics.get("output_tokens")
            if isinstance(it, (int, float)):
                input_raw = int(it)
                instance.usage_input_running_total = max(
                    instance.usage_input_running_total, input_raw
                )
            if isinstance(ot, (int, float)):
                output_raw = int(ot)
                instance.output_tokens = output_raw
                instance.usage_output_running_total = max(
                    instance.usage_output_running_total, output_raw
                )
        except Exception:
            pass

        fresh_input = None
        if total_tokens_val is not None and output_raw is not None:
            fresh_input = max(0, total_tokens_val - output_raw)
        elif input_raw is not None:
            fresh_input = max(0, input_raw)

        if fresh_input is not None:
            instance.input_tokens = fresh_input
            instance.usage_prompt_running_total = max(
                instance.usage_prompt_running_total, fresh_input
            )
        if input_raw is not None and fresh_input is not None:
            instance.cached_input_tokens = max(
                instance.cached_input_tokens, max(0, input_raw - fresh_input)
            )

        try:
            delta_total = instance.total_tokens - prev_total
            if delta_total > 0:
                if self.state.current_run:
                    self.state.current_run.total_tokens += delta_total
                instance.applied_run_tokens += delta_total
        except Exception:
            pass

    def _capture_final_message(
        self, instance: InstanceDisplay, data: Dict[str, Any]
    ) -> None:
        try:
            fm = data.get("final_message")
            if isinstance(fm, str):
                instance.final_message = fm
            instance.final_message_truncated = bool(data.get("final_message_truncated"))
            fmp = data.get("final_message_path")
            if isinstance(fmp, str) and fmp:
                instance.final_message_path = fmp
        except Exception:
            pass

    def _append_task_completed_message(
        self,
        instance_id: str,
        metrics: Dict[str, Any],
        artifact: Dict[str, Any],
        data: Dict[str, Any],
    ) -> None:
        branch = artifact.get("branch_final") or artifact.get("branch_planned")
        dur = metrics.get("duration_seconds") or data.get("duration_seconds")
        toks = metrics.get("total_tokens")
        parts = ["completed"]
        if dur is not None:
            parts.append(f"time={dur}")
        if toks is not None:
            parts.append(f"tok={toks}")
        if branch:
            parts.append(f"branch={branch}")
        self._append_msg(instance_id, " ".join(parts))


__all__ = ["TaskLifecycleHandlers"]
