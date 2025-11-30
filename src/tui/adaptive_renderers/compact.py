"""Compact dashboard renderer (6–30 instances)."""

from __future__ import annotations

from datetime import datetime

from rich.align import Align
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import InstanceDisplay, InstanceStatus, RunDisplay


class CompactRenderer:
    """Render compact grouped rows per strategy."""

    def _render_compact(self, run: RunDisplay, frame_now=None) -> RenderableType:
        panels = []
        try:
            strategies = list(run.strategies.values())
            instances_map = dict(run.instances)
        except Exception:
            strategies = []
            instances_map = {}

        for strat in strategies:
            items = [
                instances_map[iid] for iid in strat.instance_ids if iid in instances_map
            ]
            if not items:
                continue

            items = sorted(
                items,
                key=lambda i: (
                    self._status_order().get(i.status, 3),
                    self._aware(i.started_at),
                ),
            )

            branch_prefix = self._branch_prefix(items)

            rows = []
            ACT_W = 24
            for inst in items:
                stat_str = f"{inst.status.emoji} {inst.status.value}"
                inst_label = (inst.instance_id[:8], inst.model or "")
                act = inst.current_activity or (
                    f"🔧 {inst.last_tool_use}" if inst.last_tool_use else ""
                )
                act = self._with_stall_suffix(inst, act)
                act = act or ""
                dur = self._instance_duration(inst, frame_now)
                br = self._branch_tail(inst.branch_name or "", branch_prefix)
                tok = self._token_display(inst)
                cost = f"${inst.cost:.4f}" if inst.cost else "-"
                rows.append((stat_str, inst_label, act, dur, tok, cost, br))

            table = self._build_compact_table(rows, ACT_W)

            try:
                summary = strat.status_summary()  # type: ignore[misc]
            except Exception:
                try:
                    summary = getattr(strat, "status_summary", "")
                    summary = summary() if callable(summary) else str(summary)
                except Exception:
                    summary = ""
            panel = Panel(
                table,
                title=f"{strat.strategy_name} {summary}",
                border_style="blue",
                padding=(0, 1),
            )
            panels.append(Padding(panel, (0, 0, 1, 0)))

        if not panels:
            return Panel(Align.center(Text("No instances", style="dim")), style="dim")
        return Group(*panels)

    def _status_order(self) -> dict[InstanceStatus, int]:
        return {
            InstanceStatus.RUNNING: 0,
            InstanceStatus.QUEUED: 1,
            InstanceStatus.COMPLETED: 2,
            InstanceStatus.FAILED: 2,
            InstanceStatus.INTERRUPTED: 2,
        }

    def _branch_prefix(self, items: list[InstanceDisplay]) -> str:
        try:
            branches = [i.branch_name for i in items if i.branch_name]
            if not branches:
                return ""
            segs = [b.split("/") for b in branches]
            min_len = min(len(s) for s in segs)
            common = []
            for idx in range(min_len):
                token = segs[0][idx]
                if all(s[idx] == token for s in segs):
                    common.append(token)
                else:
                    break
            prefix = "/".join(common)
            if prefix and all(
                b == prefix or b.startswith(prefix + "/") for b in branches
            ):
                return prefix
        except Exception:
            pass
        return ""

    def _branch_tail(self, branch: str, prefix: str) -> str:
        if not branch:
            return ""
        try:
            if prefix and branch.startswith(prefix + "/"):
                trimmed = branch[len(prefix) :]
            else:
                trimmed = branch
            BRANCH_FULL_MAX = 32
            return branch if len(branch) <= BRANCH_FULL_MAX else trimmed
        except Exception:
            return branch

    def _with_stall_suffix(self, inst: InstanceDisplay, act: str) -> str:
        try:
            if (
                inst.status == InstanceStatus.RUNNING
                and inst.last_updated is not None
                and act
            ):
                now = (
                    datetime.now(inst.last_updated.tzinfo)
                    if inst.last_updated.tzinfo
                    else datetime.now()
                )
                age = (now - inst.last_updated).total_seconds()
                if age >= 30:
                    return f"{act} (stalled {int(age)}s)"
        except Exception:
            pass
        return act

    def _instance_duration(self, inst: InstanceDisplay, frame_now=None) -> str:
        dur = "-"
        try:
            secs = inst.duration_seconds or 0
            if inst.status == InstanceStatus.RUNNING and inst.started_at:
                now = (
                    frame_now
                    if frame_now is not None
                    else (
                        datetime.now(inst.started_at.tzinfo)
                        if inst.started_at.tzinfo
                        else datetime.now()
                    )
                )
                start_eff = (
                    inst.started_at.replace(tzinfo=now.tzinfo)
                    if inst.started_at.tzinfo is None and getattr(now, "tzinfo", None)
                    else inst.started_at
                )
                secs = max(secs, (now - start_eff).total_seconds())
            if secs > 0:
                dur = self._format_duration(secs)
        except Exception:
            pass
        return dur

    def _token_display(self, inst: InstanceDisplay) -> str:
        try:
            tok = (
                f"{inst.total_tokens:,} (↓{inst.output_tokens:,} ↑{inst.input_tokens:,}"
            )
            if getattr(inst, "cached_input_tokens", 0):
                tok += f" +{inst.cached_input_tokens:,} cached"
            tok += ")"
            return tok
        except Exception:
            return f"{inst.total_tokens:,}"

    def _build_compact_table(self, rows, act_width: int) -> Table:
        STATUS_W = 12
        INST_W = 10
        MODEL_W = 16
        ACT_W = act_width
        DUR_W = 6
        TOK_W = 16
        COST_W = 10

        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Status", width=STATUS_W, no_wrap=True)
        table.add_column("Instance", style="cyan", width=INST_W, no_wrap=True)
        table.add_column("Model", width=MODEL_W, no_wrap=True)
        table.add_column("Activity", width=ACT_W, no_wrap=True)
        table.add_column("Dur", width=DUR_W, no_wrap=True)
        table.add_column("Tokens", width=TOK_W, no_wrap=True)
        table.add_column("Cost", width=COST_W, no_wrap=True)
        table.add_column("Branch", ratio=3, no_wrap=True)

        for stat_str, inst_label, act, dur, tok, cost, br in rows:
            stext = self._status_text(stat_str)
            act_disp = act if len(act) <= ACT_W else act[: max(0, ACT_W - 3)] + "..."
            inst_col, model_col = self._split_instance_model(inst_label)
            table.add_row(stext, inst_col, model_col, act_disp, dur, tok, cost, br)

        return table

    def _status_text(self, stat_str: str) -> Text:
        stext = Text(stat_str)
        try:
            parts = stat_str.split(" ", 1)
            if len(parts) == 2:
                stext = Text(parts[0] + " ")
                color = {
                    "queued": "dim",
                    "running": "yellow",
                    "interrupted": "magenta",
                    "completed": "green",
                    "failed": "red",
                }.get(parts[1], "white")
                stext.append(parts[1], style=color)
        except Exception:
            pass
        return stext

    def _split_instance_model(self, inst_label):
        if isinstance(inst_label, tuple):
            try:
                return inst_label
            except Exception:
                pass
        if "(" in inst_label and inst_label.endswith(")"):
            try:
                base, rest = inst_label.split(" ", 1)
                return base, rest.strip()[1:-1]
            except Exception:
                return inst_label, ""
        return inst_label, ""


__all__ = ["CompactRenderer"]
