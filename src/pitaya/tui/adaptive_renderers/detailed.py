"""Detailed dashboard renderer (≤5 instances)."""

from __future__ import annotations

from datetime import datetime

from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import InstanceDisplay, InstanceStatus, RunDisplay


class DetailedRenderer:
    """Render detailed cards for small instance counts."""

    def _render_detailed(self, run: RunDisplay, frame_now=None) -> RenderableType:
        panels = []
        try:
            strategies = list(run.strategies.values())
            instances_map = dict(run.instances)
        except Exception:
            return Panel(Align.center(Text("Rendering...", style="dim")), style="dim")

        for strategy in strategies:
            seen: set[str] = set()
            ordered_ids = []
            for iid in list(strategy.instance_ids):
                if iid in instances_map and iid not in seen:
                    seen.add(iid)
                    ordered_ids.append(iid)
            strategy_instances = [instances_map[iid] for iid in ordered_ids]
            if not strategy_instances:
                continue

            status_order = {
                InstanceStatus.RUNNING: 0,
                InstanceStatus.QUEUED: 1,
                InstanceStatus.COMPLETED: 2,
                InstanceStatus.FAILED: 2,
                InstanceStatus.INTERRUPTED: 2,
            }
            strategy_instances = sorted(
                strategy_instances,
                key=lambda i: (
                    status_order.get(i.status, 3),
                    self._aware(i.started_at),
                ),
            )

            branch_prefix = self._common_branch_prefix(strategy_instances)

            instance_panels = [
                self._create_detailed_instance_panel(
                    instance,
                    ui_started_at=getattr(run, "ui_started_at", None),
                    frame_now=frame_now,
                    branch_prefix=branch_prefix,
                )
                for instance in strategy_instances
            ]

            try:
                summary = strategy.status_summary()  # type: ignore[misc]
            except Exception:
                try:
                    summary = getattr(strategy, "status_summary", "")
                    summary = summary() if callable(summary) else str(summary)
                except Exception:
                    summary = ""
            strategy_panel = Panel(
                Group(*instance_panels),
                title=f"[bold]{strategy.strategy_name}[/bold] {summary}",
                border_style="blue",
                padding=(0, 1),
            )
            panels.append(strategy_panel)

        if not panels:
            debug_text = f"No instances found. Run has {len(run.instances)} instances, {len(run.strategies)} strategies"
            return Panel(Align.center(Text(debug_text, style="dim")), style="dim")
        return Group(*panels)

    def _common_branch_prefix(self, instances: list[InstanceDisplay]) -> str:
        try:
            branches = [i.branch_name for i in instances if i.branch_name]
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

    def _create_detailed_instance_panel(
        self,
        instance: InstanceDisplay,
        ui_started_at=None,
        frame_now=None,
        branch_prefix: str = "",
    ) -> Panel:
        table = Table(show_header=False, show_edge=False, padding=0, expand=True)
        table.add_column("label", style="bold", width=10)
        table.add_column("value")

        status_text = Text()
        status_text.append(instance.status.emoji + " ")
        status_text.append(instance.status.value.title(), style=instance.status.color)
        table.add_row("Status:", status_text)

        self._add_activity_row(instance, table)
        if instance.last_tool_use:
            table.add_row("Last Tool:", f"🔧 {instance.last_tool_use}")
        self._add_duration_row(instance, table, ui_started_at, frame_now)
        self._add_usage_rows(instance, table)
        self._add_error_row(instance, table)
        if instance.branch_name:
            table.add_row("Branch:", Text(instance.branch_name, style="bright_blue"))

        panel_style = (
            {
                InstanceStatus.RUNNING: "bright_yellow",
                InstanceStatus.COMPLETED: "bright_green",
                InstanceStatus.FAILED: "bright_red",
                InstanceStatus.QUEUED: "white",
                InstanceStatus.INTERRUPTED: "bright_magenta",
            }
            if self._color_scheme == "accessible"
            else {
                InstanceStatus.RUNNING: "yellow",
                InstanceStatus.COMPLETED: "green",
                InstanceStatus.FAILED: "red",
                InstanceStatus.QUEUED: "dim",
                InstanceStatus.INTERRUPTED: "magenta",
            }
        ).get(instance.status, "white")

        title = (
            f"[bold]{instance.display_name}[/bold] ({instance.model})"
            if instance.model
            else f"[bold]{instance.display_name}[/bold]"
        )
        return Panel(table, title=title, border_style=panel_style, padding=(0, 1))

    def _add_activity_row(
        self, instance: InstanceDisplay, table: Table, frame_now: datetime | None = None
    ) -> None:
        if not instance.current_activity:
            return
        stale_suffix = ""
        try:
            if instance.status == InstanceStatus.RUNNING and instance.last_updated:
                now = (
                    frame_now
                    if frame_now
                    else (
                        datetime.now(instance.last_updated.tzinfo)
                        if instance.last_updated.tzinfo
                        else datetime.now()
                    )
                )
                age = (now - instance.last_updated).total_seconds()
                if age >= 30:
                    stale_suffix = f"  (stalled {int(age)}s)"
        except Exception:
            pass
        if instance.status == InstanceStatus.RUNNING:
            activity = Text()
            activity.append("🔄 ", style="yellow")
            activity.append(instance.current_activity)
            if stale_suffix:
                activity.append(stale_suffix, style="red")
            table.add_row("Activity:", activity)
        else:
            value = instance.current_activity + stale_suffix
            table.add_row("Activity:", value)

    def _add_duration_row(
        self,
        instance: InstanceDisplay,
        table: Table,
        ui_started_at=None,
        frame_now=None,
    ) -> None:
        try:
            live_seconds = 0.0
            if instance.status == InstanceStatus.RUNNING and instance.started_at:
                now = (
                    frame_now
                    if frame_now is not None
                    else (
                        datetime.now(instance.started_at.tzinfo)
                        if instance.started_at.tzinfo
                        else datetime.now()
                    )
                )
                start_eff = instance.started_at
                if ui_started_at:
                    try:
                        if (
                            start_eff.tzinfo is None
                            and ui_started_at.tzinfo is not None
                        ):
                            start_eff = start_eff.replace(tzinfo=ui_started_at.tzinfo)
                    except Exception:
                        pass
                    if start_eff < ui_started_at:
                        start_eff = ui_started_at
                live_seconds = max(0.0, (now - start_eff).total_seconds())
            show_seconds = (
                instance.duration_seconds
                if instance.duration_seconds > 0
                else live_seconds
            )
            if show_seconds > 0:
                table.add_row("Duration:", self._format_duration(show_seconds))
        except Exception:
            pass

    def _add_usage_rows(self, instance: InstanceDisplay, table: Table) -> None:
        try:
            token_text = f"{instance.total_tokens:,} (↓{instance.output_tokens:,} ↑{instance.input_tokens:,}"
            if getattr(instance, "cached_input_tokens", 0):
                token_text += f" +{instance.cached_input_tokens:,} cached"
            token_text += ")"
            table.add_row("Tokens:", token_text)
        except Exception:
            pass
        if instance.cost:
            try:
                table.add_row("Cost:", f"${instance.cost:.4f}")
            except Exception:
                table.add_row("Cost:", f"${instance.cost}")

    def _add_error_row(self, instance: InstanceDisplay, table: Table) -> None:
        if not instance.error:
            return
        error_text = Text(
            instance.error[:50] + "..." if len(instance.error) > 50 else instance.error
        )
        error_text.stylize("red")
        table.add_row("Error:", error_text)


__all__ = ["DetailedRenderer"]
