"""
Adaptive display logic for the Pitaya TUI.

Implements three display modes based on instance count:
- Detailed (<=5 instances): Full instance details with progress
- Compact (6-30 instances): One line per instance
- Dense (>30 instances): Strategy-level summaries only
"""

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.console import Group

from .models import RunDisplay, InstanceDisplay, InstanceStatus
from datetime import datetime


class AdaptiveDisplay:
    """Handles adaptive display rendering based on instance count."""

    def __init__(self):
        """Initialize adaptive display."""
        self._mode_renderers = {
            "detailed": self._render_detailed,
            "compact": self._render_compact,
            "dense": self._render_dense,
        }
        self._color_scheme = "default"  # default|accessible

    def set_color_scheme(self, scheme: str) -> None:
        s = (scheme or "default").strip().lower()
        if s not in ("default", "accessible"):
            s = "default"
        self._color_scheme = s

    def render_dashboard(self, run: RunDisplay, display_mode: str, frame_now=None) -> RenderableType:
        """
        Render dashboard based on display mode.

        Args:
            run: Current run state
            display_mode: One of "detailed", "compact", "dense"

        Returns:
            Rich renderable for the dashboard
        """
        renderer = self._mode_renderers.get(display_mode, self._render_compact)
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"render_dashboard mode={display_mode} strategies={len(run.strategies)} instances={len(run.instances)}"
            )
        except Exception:
            pass
        return renderer(run, frame_now=frame_now)

    def _render_detailed(self, run: RunDisplay, frame_now=None) -> RenderableType:
        """
        Render detailed view for <=10 instances.

        Shows full instance cards with:
        - Instance ID and status
        - Current activity with animated spinner
        - Progress metrics (time, tokens, cost)
        - Last tool use
        """
        panels = []

        # Snapshot to avoid concurrent mutation during iteration
        try:
            strategies = list(run.strategies.values())
            instances_map = dict(run.instances)
        except Exception:
            # Fallback: empty view if snapshotting fails
            return Panel(Align.center(Text("Rendering...", style="dim")), style="dim")

        # Group instances by strategy
        for strategy in strategies:
            # Deduplicate instance IDs to avoid duplicate cards
            seen: set[str] = set()
            ordered_ids = []
            # Snapshot instance id list to avoid concurrent mutation during iteration
            for iid in list(strategy.instance_ids):
                if iid in instances_map and iid not in seen:
                    seen.add(iid)
                    ordered_ids.append(iid)
            strategy_instances = [instances_map[iid] for iid in ordered_ids]

            if not strategy_instances:
                continue

            # Order: running → queued → done; stable within groups
            status_order = {
                InstanceStatus.RUNNING: 0,
                InstanceStatus.QUEUED: 1,
                InstanceStatus.COMPLETED: 2,
                InstanceStatus.FAILED: 2,
                InstanceStatus.INTERRUPTED: 2,
            }
            strategy_instances = sorted(
                strategy_instances,
                key=lambda i: (status_order.get(i.status, 3), i.started_at or datetime.now())
            )
            # Create instance panels
            instance_panels = []
            for instance in strategy_instances:
                panel = self._create_detailed_instance_panel(
                    instance,
                    ui_started_at=getattr(run, "ui_started_at", None),
                    frame_now=frame_now,
                )
                instance_panels.append(panel)

            # Create strategy panel containing instances
            strategy_panel = Panel(
                Group(*instance_panels),
                title=f"[bold]{strategy.strategy_name}[/bold] {strategy.status_summary}",
                border_style="blue",
                padding=(0, 1),
            )
            panels.append(strategy_panel)

        if not panels:
            # Debug info
            debug_text = f"No instances found. Run has {len(run.instances)} instances, {len(run.strategies)} strategies"
            return Panel(Align.center(Text(debug_text, style="dim")), style="dim")

        # Return all strategy panels
        return Group(*panels)

    def _create_detailed_instance_panel(self, instance: InstanceDisplay, ui_started_at=None, frame_now=None) -> Panel:
        """Create a detailed panel for a single instance."""
        # Create content table
        table = Table(show_header=False, show_edge=False, padding=0, expand=True)
        table.add_column("label", style="bold", width=10)
        table.add_column("value")

        # Status with emoji + uppercase label
        label_map = {
            InstanceStatus.RUNNING: "RUN",
            InstanceStatus.COMPLETED: "DONE",
            InstanceStatus.FAILED: "FAIL",
            InstanceStatus.QUEUED: "QUE",
            InstanceStatus.INTERRUPTED: "INT",
        }
        status_text = Text()
        status_text.append(instance.status.emoji + " ")
        status_text.append(instance.status.value.title(), style=instance.status.color)
        status_text.append(f" [{label_map.get(instance.status, '').upper()}]", style="bold")
        table.add_row("Status:", status_text)

        # Current activity
        if instance.current_activity:
            # Detect staleness (> 30s since last update)
            stale_suffix = ""
            try:
                if instance.status == InstanceStatus.RUNNING and instance.last_updated:
                    now = datetime.now(instance.last_updated.tzinfo) if instance.last_updated.tzinfo else datetime.now()
                    age = (now - instance.last_updated).total_seconds()
                    if age >= 30:
                        stale_suffix = f"  (stalled {int(age)}s)"
            except Exception:
                pass
            if instance.status == InstanceStatus.RUNNING:
                # Add spinner for running instances
                activity = Text()
                activity.append("🔄 ", style="yellow")
                activity.append(instance.current_activity)
                if stale_suffix:
                    activity.append(stale_suffix, style="red")
                table.add_row("Activity:", activity)
            else:
                value = instance.current_activity + stale_suffix
                table.add_row("Activity:", value)

        # Last tool use
        if instance.last_tool_use:
            table.add_row("Last Tool:", f"🔧 {instance.last_tool_use}")

        # Progress metrics
        # Duration: prefer live duration for running, else recorded duration
        try:
            live_seconds = 0.0
            if instance.status == InstanceStatus.RUNNING and instance.started_at:
                if frame_now is not None:
                    now = frame_now if instance.started_at.tzinfo else frame_now.replace(tzinfo=None)
                else:
                    now = datetime.now(instance.started_at.tzinfo) if instance.started_at.tzinfo else datetime.now()
                start_eff = instance.started_at
                # Clamp to UI start if provided so per-instance never precedes global
                if ui_started_at:
                    try:
                        if start_eff.tzinfo is None and ui_started_at.tzinfo is not None:
                            start_eff = start_eff.replace(tzinfo=ui_started_at.tzinfo)
                    except Exception:
                        pass
                    if start_eff < ui_started_at:
                        start_eff = ui_started_at
                live_seconds = max(0.0, (now - start_eff).total_seconds())
            show_seconds = instance.duration_seconds if instance.duration_seconds > 0 else live_seconds
            if show_seconds > 0:
                table.add_row("Duration:", self._format_duration(show_seconds))
        except Exception:
            pass

        # Tokens and cost are omitted in cards by default (see details/footer)

        # Error info
        if instance.error:
            error_text = Text(
                instance.error[:50] + "..."
                if len(instance.error) > 50
                else instance.error
            )
            error_text.stylize("red")
            table.add_row("Error:", error_text)

        # Branch name
        if instance.branch_name:
            branch_text = Text(instance.branch_name, style="cyan")
            table.add_row("Branch:", branch_text)

        # Create panel with appropriate styling (supports accessible scheme)
        if self._color_scheme == "accessible":
            panel_style = {
                InstanceStatus.RUNNING: "bright_yellow",
                InstanceStatus.COMPLETED: "bright_green",
                InstanceStatus.FAILED: "bright_red",
                InstanceStatus.QUEUED: "white",
                InstanceStatus.INTERRUPTED: "bright_magenta",
            }.get(instance.status, "bright_white")
        else:
            panel_style = {
                InstanceStatus.RUNNING: "yellow",
                InstanceStatus.COMPLETED: "green",
                InstanceStatus.FAILED: "red",
                InstanceStatus.QUEUED: "dim",
                InstanceStatus.INTERRUPTED: "magenta",
            }.get(instance.status, "white")

        # Build title with optional model suffix
        if instance.model:
            title = f"[bold]{instance.display_name}[/bold] ({instance.model})"
        else:
            title = f"[bold]{instance.display_name}[/bold]"
        return Panel(
            table,
            title=title,
            border_style=panel_style,
            padding=(0, 1),
        )

    def _render_compact(self, run: RunDisplay, frame_now=None) -> RenderableType:
        """
        Render compact view for 6-30 instances.

        Shows one-line summaries per instance with:
        - Instance ID [status emoji]
        - Current activity or last tool
        - Key metrics (time, cost)
        """
        # Create main table
        table = Table(
            title="Instance Status", show_header=True, header_style="bold", expand=True
        )

        # Define columns
        table.add_column("Instance", style="cyan", width=12)
        table.add_column("Strategy", style="blue", width=15)
        table.add_column("Status", width=10)
        table.add_column("Activity", width=30)
        table.add_column("Duration", width=8)

        # Build a fallback mapping from instance_id -> strategy name to avoid 'unknown' labels
        fallback_strategy_by_instance: dict[str, str] = {}
        try:
            for strat in run.strategies.values():
                sname = strat.strategy_name
                if not sname or sname.lower() == "unknown":
                    continue
                for iid in strat.instance_ids:
                    fallback_strategy_by_instance[iid] = sname
        except Exception:
            pass

        # Snapshot and sort instances by strategy and status
        try:
            instances = list(run.instances.values())
        except Exception:
            instances = []
        status_order = {
            InstanceStatus.RUNNING: 0,
            InstanceStatus.QUEUED: 1,
            InstanceStatus.COMPLETED: 2,
            InstanceStatus.FAILED: 2,
            InstanceStatus.INTERRUPTED: 2,
        }
        instances = sorted(
            instances,
            key=lambda i: (i.strategy_name, status_order.get(i.status, 3), i.started_at or datetime.now()),
        )

        # Add rows
        for instance in instances:
            # Status with emoji
            label_map = {
                InstanceStatus.RUNNING: "RUN",
                InstanceStatus.COMPLETED: "DONE",
                InstanceStatus.FAILED: "FAIL",
                InstanceStatus.QUEUED: "QUE",
                InstanceStatus.INTERRUPTED: "INT",
            }
            status = Text()
            status.append(instance.status.emoji + " ")
            status.append(instance.status.value, style=instance.status.color)
            status.append(f" [{label_map.get(instance.status,'').upper()}]", style="bold")

            # Activity (+ staleness)
            activity = ""
            if instance.current_activity:
                # Add staleness if > 30s
                stale = ""
                try:
                    if instance.status == InstanceStatus.RUNNING and instance.last_updated:
                        now = datetime.now(instance.last_updated.tzinfo) if instance.last_updated.tzinfo else datetime.now()
                        age = (now - instance.last_updated).total_seconds()
                        if age >= 30:
                            stale = f" (stalled {int(age)}s)"
                except Exception:
                    pass
                base = instance.current_activity
                activity = (base[:30] + ("..." if len(base) > 30 else "")) + stale
            elif instance.last_tool_use:
                activity = f"🔧 {instance.last_tool_use}"

            # Metrics - live duration with clamping to UI start; synchronized to frame_now
            duration = "-"
            try:
                live_seconds = 0.0
                if instance.status == InstanceStatus.RUNNING and instance.started_at:
                    if frame_now is not None:
                        now = frame_now if instance.started_at.tzinfo else frame_now.replace(tzinfo=None)
                    else:
                        now = datetime.now(instance.started_at.tzinfo) if instance.started_at.tzinfo else datetime.now()
                    start_eff = instance.started_at
                    ui_start = getattr(run, "ui_started_at", None)
                    if ui_start:
                        try:
                            if start_eff.tzinfo is None and ui_start.tzinfo is not None:
                                start_eff = start_eff.replace(tzinfo=ui_start.tzinfo)
                        except Exception:
                            pass
                        if start_eff < ui_start:
                            start_eff = ui_start
                    live_seconds = max(0.0, (now - start_eff).total_seconds())
                seconds_to_show = instance.duration_seconds if instance.duration_seconds > 0 else live_seconds
                if seconds_to_show > 0:
                    duration = self._format_duration(seconds_to_show)
            except Exception:
                pass

            # Strategy label with robust fallback
            strategy_label = instance.strategy_name or ""
            if not strategy_label or strategy_label.lower() == "unknown":
                strategy_label = fallback_strategy_by_instance.get(instance.instance_id, "-")

            table.add_row(
                instance.display_name,
                strategy_label,
                status,
                activity,
                duration,
            )

        return Panel(table, border_style="blue", padding=(0, 1))

    def _render_dense(self, run: RunDisplay) -> RenderableType:
        """Render dense view for >30 instances with minimal per-instance lines (phase/runtime)."""
        panels = []
        try:
            strategies = list(run.strategies.values())
            instances_map = dict(run.instances)
        except Exception:
            strategies = []
            instances_map = {}

        for strategy in strategies:
            items = [instances_map[iid] for iid in strategy.instance_ids if iid in instances_map]
            if not items:
                continue
            status_order = {
                InstanceStatus.RUNNING: 0,
                InstanceStatus.QUEUED: 1,
                InstanceStatus.COMPLETED: 2,
                InstanceStatus.FAILED: 2,
                InstanceStatus.INTERRUPTED: 2,
            }
            items = sorted(items, key=lambda i: (status_order.get(i.status, 3), i.started_at or datetime.now()))
            table = Table(show_header=False, expand=True, padding=(0,1))
            table.add_column("Status", width=8)
            table.add_column("Instance", width=10)
            table.add_column("Phase", width=24)
            table.add_column("Runtime", width=8)
            table.add_column("Branch")
            for inst in items:
                label_map = {
                    InstanceStatus.RUNNING: "RUN",
                    InstanceStatus.COMPLETED: "DONE",
                    InstanceStatus.FAILED: "FAIL",
                    InstanceStatus.QUEUED: "QUE",
                    InstanceStatus.INTERRUPTED: "INT",
                }
                status = Text(inst.status.emoji + " " + inst.status.value, style=inst.status.color)
                status.append(f" [{label_map.get(inst.status,'').upper()}]", style="bold")
                inst_name = inst.instance_id[:8]
                phase = inst.current_activity or "-"
                dur = "-"
                try:
                    secs = inst.duration_seconds or 0
                    if inst.status == InstanceStatus.RUNNING and inst.started_at:
                        now = datetime.now(inst.started_at.tzinfo) if inst.started_at.tzinfo else datetime.now()
                        secs = max(secs, (now - inst.started_at).total_seconds())
                    if secs > 0:
                        dur = self._format_duration(secs)
                except Exception:
                    pass
                branch = (inst.branch_name or "")
                if branch and len(branch) > 48:
                    branch = branch[:45] + "..."
                table.add_row(status, inst_name, phase, dur, branch)
            panels.append(Panel(table, title=strategy.strategy_name, border_style="blue", padding=(0,1)))
        if not panels:
            return Panel(Align.center(Text("No instances", style="dim")), style="dim")
        return Group(*panels)

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
