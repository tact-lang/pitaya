"""Dense dashboard renderer (>30 instances)."""

from __future__ import annotations

from rich.align import Align
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..models import InstanceStatus, RunDisplay


class DenseRenderer:
    """Render dense per-strategy summaries."""

    def _render_dense(self, run: RunDisplay, frame_now=None) -> RenderableType:
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

            total = strategy.total_instances or len(strategy.instance_ids) or len(items)
            total = total if total > 0 else len(items)

            running = sum(1 for i in items if i.status == InstanceStatus.RUNNING)
            queued = sum(1 for i in items if i.status == InstanceStatus.QUEUED)
            completed = sum(1 for i in items if i.status == InstanceStatus.COMPLETED)
            failed = sum(1 for i in items if i.status == InstanceStatus.FAILED)
            interrupted = sum(
                1 for i in items if i.status == InstanceStatus.INTERRUPTED
            )
            finished = min(completed + failed, total)

            cost = sum((i.cost or 0.0) for i in items)
            tokens = sum(int(i.total_tokens or 0) for i in items)

            bar = ProgressBar(
                total=max(total, 1),
                completed=finished,
                style="grey35",
                complete_style="bright_green",
                finished_style="bright_green",
            )
            pct = int(100 * (finished / total)) if total else 0

            bar_row = Table.grid(expand=True)
            bar_row.add_column("bar", ratio=5)
            bar_row.add_column("pct", width=10, justify="right")
            bar_row.add_row(bar, Text(f"{finished}/{total} ({pct}%)", style="bold"))

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

            try:
                dsum = strategy.status_summary()  # type: ignore[misc]
            except Exception:
                try:
                    dsum = getattr(strategy, "status_summary", "")
                    dsum = dsum() if callable(dsum) else str(dsum)
                except Exception:
                    dsum = ""

            panel = Panel(
                Group(bar_row, stats_row),
                title=f"{strategy.strategy_name} {dsum}",
                border_style="blue",
                padding=(0, 1),
            )
            panels.append(Padding(panel, (0, 0, 1, 0)))

        if not panels:
            return Panel(Align.center(Text("No instances", style="dim")), style="dim")
        return Group(*panels)


__all__ = ["DenseRenderer"]
