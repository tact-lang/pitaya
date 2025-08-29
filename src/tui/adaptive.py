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
from rich.padding import Padding
from rich.progress_bar import ProgressBar

from .models import RunDisplay, InstanceDisplay, InstanceStatus
from datetime import datetime, timezone


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

    def _aware(self, dt: datetime | None) -> datetime:
        """Return a timezone-aware datetime (UTC) for safe comparisons/sorts."""
        if dt is None:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def set_color_scheme(self, scheme: str) -> None:
        s = (scheme or "default").strip().lower()
        if s not in ("default", "accessible"):
            s = "default"
        self._color_scheme = s

    def render_dashboard(
        self, run: RunDisplay, display_mode: str, frame_now=None
    ) -> RenderableType:
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
                key=lambda i: (
                    status_order.get(i.status, 3),
                    self._aware(i.started_at),
                ),
            )
            # Compute common branch prefix for this strategy to trim noisy leading segments
            try:
                branches = [i.branch_name for i in strategy_instances if i.branch_name]
                branch_prefix = ""
                if branches:
                    segs = [b.split("/") for b in branches]
                    min_len = min(len(s) for s in segs)
                    common = []
                    for idx in range(min_len):
                        token = segs[0][idx]
                        if all(s[idx] == token for s in segs):
                            common.append(token)
                        else:
                            break
                    branch_prefix = "/".join(common)
                    if branch_prefix and all(
                        b == branch_prefix or b.startswith(branch_prefix + "/")
                        for b in branches
                    ):
                        pass
                    else:
                        branch_prefix = ""
            except Exception:
                branch_prefix = ""

            # Create instance panels
            instance_panels = []
            for instance in strategy_instances:
                panel = self._create_detailed_instance_panel(
                    instance,
                    ui_started_at=getattr(run, "ui_started_at", None),
                    frame_now=frame_now,
                    branch_prefix=branch_prefix,
                )
                instance_panels.append(panel)

            # Create strategy panel containing instances
            # Safe status summary call
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
            # Debug info
            debug_text = f"No instances found. Run has {len(run.instances)} instances, {len(run.strategies)} strategies"
            return Panel(Align.center(Text(debug_text, style="dim")), style="dim")

        # Return all strategy panels
        return Group(*panels)

    def _create_detailed_instance_panel(
        self,
        instance: InstanceDisplay,
        ui_started_at=None,
        frame_now=None,
        branch_prefix: str = "",
    ) -> Panel:
        """Create a detailed panel for a single instance."""
        # Create content table
        table = Table(show_header=False, show_edge=False, padding=0, expand=True)
        table.add_column("label", style="bold", width=10)
        table.add_column("value")

        # Status with emoji + uppercase label
        status_text = Text()
        status_text.append(instance.status.emoji + " ")
        status_text.append(instance.status.value.title(), style=instance.status.color)
        table.add_row("Status:", status_text)

        # Current activity
        if instance.current_activity:
            # Detect staleness (> 30s since last update)
            stale_suffix = ""
            try:
                if instance.status == InstanceStatus.RUNNING and instance.last_updated:
                    now = (
                        datetime.now(instance.last_updated.tzinfo)
                        if instance.last_updated.tzinfo
                        else datetime.now()
                    )
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
                    now = (
                        frame_now
                        if instance.started_at.tzinfo
                        else frame_now.replace(tzinfo=None)
                    )
                else:
                    now = (
                        datetime.now(instance.started_at.tzinfo)
                        if instance.started_at.tzinfo
                        else datetime.now()
                    )
                start_eff = instance.started_at
                # Clamp to UI start if provided so per-instance never precedes global
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
            # Always show full branch in detailed mode for clarity
            branch_text = Text(instance.branch_name, style="bright_blue")
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
        """Render compact grouped by strategy with short, consistent columns."""
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
            status_order = {
                InstanceStatus.RUNNING: 0,
                InstanceStatus.QUEUED: 1,
                InstanceStatus.COMPLETED: 2,
                InstanceStatus.FAILED: 2,
                InstanceStatus.INTERRUPTED: 2,
            }
            items = sorted(
                items,
                key=lambda i: (
                    status_order.get(i.status, 3),
                    self._aware(i.started_at),
                ),
            )

            # Compute common branch prefix across items to emphasize unique suffix
            try:
                b_list = [i.branch_name for i in items if i.branch_name]
                branch_prefix = ""
                if b_list:
                    segs = [b.split("/") for b in b_list]
                    min_len = min(len(s) for s in segs)
                    common = []
                    for idx in range(min_len):
                        token = segs[0][idx]
                        if all(s[idx] == token for s in segs):
                            common.append(token)
                        else:
                            break
                    branch_prefix = "/".join(common)
                    if not branch_prefix or not all(
                        b == branch_prefix or b.startswith(branch_prefix + "/")
                        for b in b_list
                    ):
                        branch_prefix = ""
            except Exception:
                branch_prefix = ""

            rows = []
            max_status = 0
            max_activity = 0
            for inst in items:
                stat_str = f"{inst.status.emoji} {inst.status.value}"
                max_status = max(max_status, len(stat_str))
                inst_id_short = inst.instance_id[:8]
                inst_model = inst.model or ""
                # Keep instance id and model separate to avoid wrapping
                inst_label = (inst_id_short, inst_model)
                act = inst.current_activity or (
                    f"🔧 {inst.last_tool_use}" if inst.last_tool_use else ""
                )
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
                            act = f"{act} (stalled {int(age)}s)"
                except Exception:
                    pass
                act = act or ""
                max_activity = max(max_activity, len(act))
                dur = "-"
                try:
                    secs = inst.duration_seconds or 0
                    if inst.status == InstanceStatus.RUNNING and inst.started_at:
                        now = (
                            frame_now
                            if (frame_now is not None)
                            else (
                                datetime.now(inst.started_at.tzinfo)
                                if inst.started_at.tzinfo
                                else datetime.now()
                            )
                        )
                        if inst.started_at.tzinfo is None and getattr(
                            now, "tzinfo", None
                        ):
                            start_eff = inst.started_at.replace(tzinfo=now.tzinfo)
                        else:
                            start_eff = inst.started_at
                        secs = max(secs, (now - start_eff).total_seconds())
                    if secs > 0:
                        dur = self._format_duration(secs)
                except Exception:
                    pass
                br_full = inst.branch_name or ""
                br = br_full
                # Trim common prefix to show the unique tail; keep leading '/'
                # so the suffix visually starts at '/k...' instead of 'k...'.
                try:
                    trimmed = br_full
                    if branch_prefix and br_full.startswith(branch_prefix + "/"):
                        trimmed = br_full[len(branch_prefix) :]
                    # Prefer full branch if short; else use trimmed tail
                    BRANCH_FULL_MAX = 32
                    br = br_full if len(br_full) <= BRANCH_FULL_MAX else trimmed
                except Exception:
                    br = br_full
                rows.append((stat_str, inst_label, act, dur, br))

            # Fixed widths for consistency
            STATUS_W = 12
            INST_W = 10  # 8-char id + spacing
            MODEL_W = 16
            ACT_W = 24
            DUR_W = 6

            table = Table(show_header=True, header_style="bold", expand=True)
            table.add_column("Status", width=STATUS_W, no_wrap=True)
            table.add_column("Instance", style="cyan", width=INST_W, no_wrap=True)
            table.add_column("Model", width=MODEL_W, no_wrap=True)
            table.add_column("Activity", width=ACT_W, no_wrap=True)
            table.add_column("Dur", width=DUR_W, no_wrap=True)
            table.add_column("Branch", ratio=3, no_wrap=True)
            for stat_str, inst_label, act, dur, br in rows:
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
                act_disp = act
                if len(act_disp) > ACT_W:
                    act_disp = act_disp[: max(0, ACT_W - 3)] + "..."
                # Split instance/model into separate columns
                inst_col = inst_label
                model_col = ""
                # The instance label is 8-char id; model goes to model column
                if "(" in inst_label and inst_label.endswith(")"):
                    try:
                        base, rest = inst_label.split(" ", 1)
                        inst_col = base
                        model_col = rest.strip()[1:-1]
                    except Exception:
                        inst_col = inst_label
                        model_col = ""
                # If rows generator already separated model, detect via sentinel
                if isinstance(inst_label, tuple):
                    try:
                        inst_col, model_col = inst_label  # type: ignore[assignment]
                    except Exception:
                        inst_col = str(inst_label)
                        model_col = ""

                table.add_row(stext, inst_col, model_col, act_disp, dur, br)

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

    def _render_dense(self, run: RunDisplay, frame_now=None) -> RenderableType:
        """Render dense view for >30 instances with minimal per-instance lines (phase/runtime)."""
        panels = []
        try:
            strategies = list(run.strategies.values())
            instances_map = dict(run.instances)
        except Exception:
            strategies = []
            instances_map = {}

        for strategy in strategies:
            items = [
                instances_map[iid]
                for iid in strategy.instance_ids
                if iid in instances_map
            ]
            if not items:
                continue
            # Dense mode: per-strategy progress bar + stats; skip per-instance rows
            total = strategy.total_instances or len(strategy.instance_ids) or len(items)
            if total <= 0:
                total = len(items)

            running = sum(1 for i in items if i.status == InstanceStatus.RUNNING)
            queued = sum(1 for i in items if i.status == InstanceStatus.QUEUED)
            completed = sum(1 for i in items if i.status == InstanceStatus.COMPLETED)
            failed = sum(1 for i in items if i.status == InstanceStatus.FAILED)
            interrupted = sum(
                1 for i in items if i.status == InstanceStatus.INTERRUPTED
            )
            finished = min(completed + failed, total)

            try:
                cost = sum((i.cost or 0.0) for i in items)
            except Exception:
                cost = 0.0
            try:
                tokens = sum(int(i.total_tokens or 0) for i in items)
            except Exception:
                tokens = 0

            # Colorful progress bar (left-aligned, expands with panel)
            # Use neutral track with bright green completion to avoid clashing with border
            bar = ProgressBar(
                total=max(total, 1),
                completed=finished,
                style="grey35",
                complete_style="bright_green",
                finished_style="bright_green",
            )
            pct = int(100 * (finished / total)) if total else 0

            # Top row: bar | percent
            bar_row = Table.grid(expand=True)
            bar_row.add_column("bar", ratio=5)
            bar_row.add_column("pct", width=10, justify="right")
            bar_row.add_row(bar, Text(f"{finished}/{total} ({pct}%)", style="bold"))

            # Second row: colored counts across, tokens/cost right-aligned
            stats_row = Table.grid(expand=True)
            stats_row.add_column(justify="left")
            stats_row.add_column(justify="left")
            stats_row.add_column(justify="left")
            stats_row.add_column(justify="left")
            stats_row.add_column(justify="left")
            stats_row.add_column(justify="right")
            stats_row.add_row(
                Text(f"run:{running}", style="yellow"),
                Text(f"que:{queued}", style="white"),
                Text(f"done:{completed}", style="green"),
                Text(f"fail:{failed}", style="red"),
                Text(f"int:{interrupted}", style="magenta"),
                Text(f"tok:{tokens:,} • cost:${cost:.2f}", style="cyan"),
            )

            content = Group(bar_row, stats_row)

            # Safe status summary
            try:
                dsum = strategy.status_summary()  # type: ignore[misc]
            except Exception:
                try:
                    dsum = getattr(strategy, "status_summary", "")
                    dsum = dsum() if callable(dsum) else str(dsum)
                except Exception:
                    dsum = ""
            panel = Panel(
                content,
                title=f"{strategy.strategy_name} {dsum}",
                border_style="blue",
                padding=(0, 1),
            )
            # Add bottom spacing between groups
            panels.append(Padding(panel, (0, 0, 1, 0)))
            continue
            status_order = {
                InstanceStatus.RUNNING: 0,
                InstanceStatus.QUEUED: 1,
                InstanceStatus.COMPLETED: 2,
                InstanceStatus.FAILED: 2,
                InstanceStatus.INTERRUPTED: 2,
            }
            items = sorted(
                items,
                key=lambda i: (
                    status_order.get(i.status, 3),
                    self._aware(i.started_at),
                ),
            )
            # Precompute widths similarly to compact
            rows2 = []
            max_status = 0
            max_phase = 0
            for inst in items:
                stat_str = f"{inst.status.emoji} {inst.status.value}"
                max_status = max(max_status, len(stat_str))
                phase = inst.current_activity or "-"
                try:
                    if (
                        inst.status == InstanceStatus.RUNNING
                        and inst.last_updated is not None
                        and inst.current_activity
                    ):
                        now = (
                            datetime.now(inst.last_updated.tzinfo)
                            if inst.last_updated.tzinfo
                            else datetime.now()
                        )
                        age = (now - inst.last_updated).total_seconds()
                        if age >= 30:
                            phase = f"{phase} (stalled {int(age)}s)"
                except Exception:
                    pass
                max_phase = max(max_phase, len(phase))
                dur = "-"
                try:
                    secs = inst.duration_seconds or 0
                    if inst.status == InstanceStatus.RUNNING and inst.started_at:
                        now = (
                            frame_now
                            if (frame_now is not None)
                            else (
                                datetime.now(inst.started_at.tzinfo)
                                if inst.started_at.tzinfo
                                else datetime.now()
                            )
                        )
                        secs = max(secs, (now - inst.started_at).total_seconds())
                    if secs > 0:
                        dur = self._format_duration(secs)
                except Exception:
                    pass
                br = inst.branch_name or ""
                if br and len(br) > 48:
                    br = br[:45] + "..."
                rows2.append((stat_str, inst.instance_id[:8], phase, dur, br))

            status_width = max_status if max_status > 0 else None
            phase_width = min(max_phase, 24) if max_phase > 0 else 20

            table = Table(show_header=False, expand=True, padding=(0, 1))
            table.add_column("Status", width=status_width)
            table.add_column("Instance", width=10)
            table.add_column("Phase", width=phase_width)
            table.add_column("Run", width=5)
            table.add_column("Branch")
            for stat_str, inst_name, phase, dur, branch in rows2:
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
                phase_disp = phase
                if len(phase_disp) > phase_width:
                    phase_disp = phase_disp[: max(0, phase_width - 3)] + "..."
                table.add_row(stext, inst_name, phase_disp, dur, branch)
            # Safe status summary
            try:
                dsum = strategy.status_summary()  # type: ignore[misc]
            except Exception:
                try:
                    dsum = getattr(strategy, "status_summary", "")
                    dsum = dsum() if callable(dsum) else str(dsum)
                except Exception:
                    dsum = ""
            panels.append(
                Panel(
                    table,
                    title=f"{strategy.strategy_name} {dsum}",
                    border_style="blue",
                    padding=(0, 1),
                )
            )
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
