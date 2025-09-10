"""List and show runs from the state directory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import argparse

from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
from rich.panel import Panel
from rich.tree import Tree

__all__ = ["run_list_runs", "run_show_run"]


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


async def run_list_runs(console: Console, args: argparse.Namespace) -> int:
    state_dir: Path = args.state_dir
    if not state_dir.exists():
        console.print("[yellow]No runs found[/yellow]")
        return 0

    run_dirs = [
        d for d in state_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        console.print("[yellow]No runs found[/yellow]")
        return 0
    run_dirs.sort(reverse=True)

    table = Table(title="Previous Runs", box=ROUNDED)
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Started", style="blue")
    table.add_column("Duration")
    table.add_column("Instances")
    table.add_column("Cost", style="yellow")
    table.add_column("Prompt", style="white", max_width=40)

    for run_dir in run_dirs:
        run_id = run_dir.name
        data = _load_json(run_dir / "state.json")
        if data:
            started = datetime.fromisoformat(data["started_at"])  # type: ignore[arg-type]
            completed = datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None  # type: ignore[arg-type]
            if completed:
                duration = completed - started
                duration_str = f"{duration.total_seconds():.0f}s"
                status = "✓ Completed"
            else:
                duration_str = "-"
                status = "○ Interrupted"
            table.add_row(
                run_id,
                status,
                started.strftime("%Y-%m-%d %H:%M:%S"),
                duration_str,
                f"{data.get('completed_instances', 0)}/{data.get('total_instances', 0)}",
                f"${data.get('total_cost', 0):.2f}",
                data.get("prompt", "")[:40]
                + ("..." if len(data.get("prompt", "")) > 40 else ""),
            )
        else:
            table.add_row(run_id, "? Unknown", "-", "-", "-", "-", "-")

    console.print(table)
    console.print(f"\nTotal runs: {len(run_dirs)}")
    return 0


def _summary_panel(run_id: str, data: dict) -> Panel:
    started = datetime.fromisoformat(data["started_at"])  # type: ignore[arg-type]
    completed = datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None  # type: ignore[arg-type]
    summary = f"""[bold]Run ID:[/bold] {run_id}
[bold]Status:[/bold] {"Completed" if completed else "Interrupted"}
[bold]Started:[/bold] {started.strftime("%Y-%m-%d %H:%M:%S")}
[bold]Completed:[/bold] {completed.strftime("%Y-%m-%d %H:%M:%S") if completed else "-"}
[bold]Duration:[/bold] {(completed - started).total_seconds():.0f}s if completed else "-"
[bold]Prompt:[/bold] {data.get("prompt", "")}
[bold]Repository:[/bold] {data.get("repo_path", "")}
[bold]Base Branch:[/bold] {data.get("base_branch", "")}

[bold]Metrics:[/bold]
  Total Instances: {data.get("total_instances", 0)}
  Completed: {data.get("completed_instances", 0)}
  Failed: {data.get("failed_instances", 0)}
  Total Cost: ${data.get("total_cost", 0):.2f}
  Total Tokens: {data.get("total_tokens", 0):,}"""
    return Panel(summary, title=f"Run Details: {run_id}")


def _render_strategies(console: Console, data: dict) -> None:
    if not data.get("strategies"):
        return
    tree = Tree("[bold]Strategies[/bold]")
    for strat_id, strat_data in data["strategies"].items():
        strat_node = tree.add(
            f"{strat_data.get('strategy_name','unknown')} ({strat_id})"
        )
        strat_node.add(f"State: {strat_data['state']}")
        strat_node.add(f"Started: {strat_data['started_at']}")
        if strat_data.get("results"):
            results_node = strat_node.add("Results")
            for i, result in enumerate(strat_data["results"]):
                results_node.add(
                    f"Instance {i+1}: Branch {result.get('branch_name', 'unknown')}"
                )
    console.print(tree)


def _render_instances(console: Console, data: dict) -> None:
    if not data.get("instances"):
        return
    inst_tree = Tree("[bold]Instances[/bold]")
    for inst_id, inst_data in data["instances"].items():
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "running": "🔄",
            "pending": "⏳",
        }.get(inst_data["state"], "❓")
        inst_node = inst_tree.add(
            f"{status_emoji} {inst_id[:8]} - {inst_data['state']}"
        )
        inst_node.add(f"Branch: {inst_data.get('branch_name', '-')}")
        inst_node.add(f"Container: {inst_data.get('container_name', '-')}")
        if inst_data.get("cost"):
            inst_node.add(f"Cost: ${inst_data['cost']:.2f}")
        if inst_data.get("duration_seconds"):
            inst_node.add(f"Duration: {inst_data['duration_seconds']:.0f}s")
    console.print(inst_tree)


async def run_show_run(console: Console, args: argparse.Namespace) -> int:
    run_id: str = args.show_run
    state_dir: Path = args.state_dir / run_id
    if not state_dir.exists():
        console.print(f"[red]Run {run_id} not found[/red]")
        return 1

    data = _load_json(state_dir / "state.json")
    if not data:
        console.print(f"[red]No snapshot found for run {run_id}[/red]")
        return 1

    console.print(_summary_panel(run_id, data))
    _render_strategies(console, data)
    _render_instances(console, data)

    console.print("\n[dim]Available actions:[/dim]")
    completed = data.get("completed_at") is not None
    if not completed:
        console.print(f"  Resume: pitaya --resume {run_id}")
    console.print(f"  View logs: {args.logs_dir}/{run_id}/")
    return 0
