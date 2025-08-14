"""
Adaptive display logic for the orchestrator TUI.

Implements three display modes based on instance count:
- Detailed (<=10 instances): Full instance details with progress
- Compact (11-50 instances): One line per instance
- Dense (>50 instances): Strategy-level summaries only
"""

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.align import Align
from rich.console import Group

from .models import RunDisplay, InstanceDisplay, InstanceStatus


class AdaptiveDisplay:
    """Handles adaptive display rendering based on instance count."""

    def __init__(self):
        """Initialize adaptive display."""
        self._mode_renderers = {
            "detailed": self._render_detailed,
            "compact": self._render_compact,
            "dense": self._render_dense,
        }

    def render_dashboard(self, run: RunDisplay, display_mode: str) -> RenderableType:
        """
        Render dashboard based on display mode.

        Args:
            run: Current run state
            display_mode: One of "detailed", "compact", "dense"

        Returns:
            Rich renderable for the dashboard
        """
        renderer = self._mode_renderers.get(display_mode, self._render_compact)
        return renderer(run)

    def _render_detailed(self, run: RunDisplay) -> RenderableType:
        """
        Render detailed view for <=10 instances.

        Shows full instance cards with:
        - Instance ID and status
        - Current activity with animated spinner
        - Progress metrics (time, tokens, cost)
        - Last tool use
        """
        panels = []

        # Group instances by strategy
        for strategy in run.strategies.values():
            # Deduplicate instance IDs to avoid duplicate cards
            seen: set[str] = set()
            ordered_ids = []
            for iid in strategy.instance_ids:
                if iid in run.instances and iid not in seen:
                    seen.add(iid)
                    ordered_ids.append(iid)
            strategy_instances = [run.instances[iid] for iid in ordered_ids]

            if not strategy_instances:
                continue

            # Create instance panels
            instance_panels = []
            for instance in strategy_instances:
                panel = self._create_detailed_instance_panel(instance)
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

    def _create_detailed_instance_panel(self, instance: InstanceDisplay) -> Panel:
        """Create a detailed panel for a single instance."""
        # Create content table
        table = Table(show_header=False, show_edge=False, padding=0, expand=True)
        table.add_column("label", style="bold", width=10)
        table.add_column("value")

        # Status with emoji
        status_text = Text()
        status_text.append(instance.status.emoji + " ")
        status_text.append(instance.status.value.title(), style=instance.status.color)
        table.add_row("Status:", status_text)

        # Current activity
        if instance.current_activity:
            if instance.status == InstanceStatus.RUNNING:
                # Add spinner for running instances
                activity = Text()
                activity.append("🔄 ", style="yellow")
                activity.append(instance.current_activity)
                table.add_row("Activity:", activity)
            else:
                table.add_row("Activity:", instance.current_activity)

        # Last tool use
        if instance.last_tool_use:
            table.add_row("Last Tool:", f"🔧 {instance.last_tool_use}")

        # Progress metrics
        if instance.duration_seconds > 0:
            table.add_row("Duration:", self._format_duration(instance.duration_seconds))

        if instance.total_tokens > 0:
            table.add_row(
                "Tokens:",
                f"{instance.total_tokens:,} (↓{instance.input_tokens:,} ↑{instance.output_tokens:,})",
            )

        if instance.cost > 0:
            table.add_row("Cost:", f"${instance.cost:.4f}")

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

        # Create panel with appropriate styling
        panel_style = {
            InstanceStatus.RUNNING: "yellow",
            InstanceStatus.COMPLETED: "green",
            InstanceStatus.FAILED: "red",
            InstanceStatus.QUEUED: "dim",
        }.get(instance.status, "white")

        return Panel(
            table,
            title=f"[bold]{instance.display_name}[/bold] ({instance.model})",
            border_style=panel_style,
            padding=(0, 1),
        )

    def _render_compact(self, run: RunDisplay) -> RenderableType:
        """
        Render compact view for 11-50 instances.

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
        table.add_column("Cost", width=10)
        table.add_column("Tokens", width=10)

        # Sort instances by strategy and status
        instances = sorted(
            run.instances.values(),
            key=lambda i: (
                i.strategy_name,
                0 if i.status == InstanceStatus.RUNNING else 1,
                i.instance_id,
            ),
        )

        # Add rows
        for instance in instances:
            # Status with emoji
            status = Text()
            status.append(instance.status.emoji + " ")
            status.append(instance.status.value, style=instance.status.color)

            # Activity
            activity = ""
            if instance.current_activity:
                activity = instance.current_activity[:30]
                if len(instance.current_activity) > 30:
                    activity += "..."
            elif instance.last_tool_use:
                activity = f"🔧 {instance.last_tool_use}"

            # Metrics
            duration = (
                self._format_duration(instance.duration_seconds)
                if instance.duration_seconds > 0
                else "-"
            )
            cost = f"${instance.cost:.3f}" if instance.cost > 0 else "-"
            tokens = f"{instance.total_tokens:,}" if instance.total_tokens > 0 else "-"

            table.add_row(
                instance.display_name,
                instance.strategy_name,
                status,
                activity,
                duration,
                cost,
                tokens,
            )

        return Panel(table, border_style="blue", padding=(0, 1))

    def _render_dense(self, run: RunDisplay) -> RenderableType:
        """
        Render dense view for >50 instances.

        Shows strategy-level summaries only:
        - Strategy name and configuration
        - Aggregate progress bar
        - Success/failure counts
        - Total cost and duration
        """
        # Create strategy summary table
        table = Table(
            title="Strategy Summary", show_header=True, header_style="bold", expand=True
        )

        # Define columns
        table.add_column("Strategy", style="blue", width=20)
        table.add_column("Progress", width=30)
        table.add_column("Success", style="green", width=8)
        table.add_column("Failed", style="red", width=8)
        table.add_column("Active", style="yellow", width=8)
        table.add_column("Cost", width=10)
        table.add_column("Avg Time", width=10)

        # Add strategy rows
        for strategy in run.strategies.values():
            # Calculate strategy metrics
            strategy_instances = [
                run.instances[iid]
                for iid in strategy.instance_ids
                if iid in run.instances
            ]

            if not strategy_instances:
                continue

            # Count by status
            active = sum(1 for i in strategy_instances if i.is_active)
            completed = sum(
                1 for i in strategy_instances if i.status == InstanceStatus.COMPLETED
            )
            failed = sum(
                1 for i in strategy_instances if i.status == InstanceStatus.FAILED
            )

            # Calculate aggregates
            total_cost = sum(i.cost for i in strategy_instances)
            total_duration = sum(
                i.duration_seconds for i in strategy_instances if i.duration_seconds > 0
            )
            avg_duration = (
                total_duration / len(strategy_instances) if strategy_instances else 0
            )

            # Create progress bar
            progress = Progress(
                SpinnerColumn(),
                BarColumn(bar_width=20),
                TextColumn("{task.percentage:>3.0f}%"),
                console=None,
                expand=False,
            )

            progress.add_task(
                "", total=len(strategy_instances), completed=completed + failed
            )

            # Strategy name with config hint
            strategy_name = strategy.strategy_name
            if strategy.config:
                # Add key config values
                if "n" in strategy.config:
                    strategy_name += f" (n={strategy.config['n']})"
                elif "temperature" in strategy.config:
                    strategy_name += f" (t={strategy.config['temperature']})"

            table.add_row(
                strategy_name,
                progress,
                str(completed),
                str(failed),
                str(active),
                f"${total_cost:.2f}",
                self._format_duration(avg_duration),
            )

        # Add summary panel below
        summary_items = [
            f"Total Instances: {run.total_instances}",
            f"Active: {run.active_instances}",
            f"Success Rate: {run.success_rate:.1f}%",
            f"Total Cost: ${run.total_cost:.2f}",
            f"Cost/Hour: ${run.cost_per_hour:.2f}",
        ]

        summary_text = " | ".join(summary_items)

        return Group(
            Panel(table, border_style="blue", padding=(0, 1)),
            Panel(
                Align.center(Text(summary_text, style="bold")),
                border_style="blue",
                padding=(0, 1),
            ),
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
