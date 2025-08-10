#!/usr/bin/env python3
"""
Main CLI for the orchestrator - runs AI coding agents with TUI display.

This is the primary entry point that combines:
- Orchestrator execution
- Strategy selection
- TUI display
- All configuration options
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import urllib3
import yaml
from rich.console import Console

from .config import (
    load_env_config,
    load_dotenv_config,
    merge_config,
    get_default_config,
)
from .orchestration import Orchestrator
from .orchestration.strategies import AVAILABLE_STRATEGIES
from .shared import AuthConfig, ContainerLimits, RetryConfig, InstanceResult
from .exceptions import OrchestratorError, ValidationError, DockerError
from .tui import TUIDisplay

# Suppress urllib3 warnings about closed connections from docker-py
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning)
urllib3.disable_warnings()

logger = logging.getLogger(__name__)


class OrchestratorCLI:
    """Main orchestrator CLI application."""

    def __init__(self):
        """Initialize CLI."""
        self.console = Console()
        self.orchestrator = None
        self.tui_display = None
        self.shutdown_event = None

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser for orchestrator."""
        parser = argparse.ArgumentParser(
            description="Orchestrator - Run multiple AI coding agents in parallel",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run with simple strategy
  orchestrator "implement auth" --strategy simple
  
  # Run with best-of-n strategy
  orchestrator "fix the bug in user.py" --strategy best-of-n -S n=5
  
  # Run with custom model
  orchestrator "add tests" --model opus
  
  # Use -S for strategy-specific parameters
  orchestrator "implement auth" --strategy best-of-n -S n=10 -S scorer_prompt="evaluate correctness"
  
  # Run multiple strategy executions in parallel
  orchestrator "fix bug" --strategy simple --runs 5
  
  # Resume interrupted run
  orchestrator --resume run_20250114_123456
  
  # Run without TUI (headless)
  orchestrator "implement feature" --no-tui
""",
        )

        # Main argument - the prompt
        parser.add_argument("prompt", nargs="?", help="Prompt for the AI coding agent")

        # Strategy options
        parser.add_argument(
            "--strategy",
            choices=list(AVAILABLE_STRATEGIES.keys()),
            default="simple",
            help="Execution strategy (default: simple)",
        )
        parser.add_argument(
            "-S",
            action="append",
            dest="strategy_params",
            metavar="KEY=VALUE",
            help="Set strategy-specific parameters (can be used multiple times)",
        )
        parser.add_argument(
            "--runs",
            type=int,
            default=1,
            help="Number of parallel strategy executions (default: 1)",
        )

        # Model options
        parser.add_argument(
            "--model",
            choices=["sonnet", "opus"],
            default="sonnet",
            help="Claude model to use (default: sonnet)",
        )

        # Repository options
        parser.add_argument(
            "--repo",
            type=Path,
            default=Path.cwd(),
            help="Repository path (default: current directory)",
        )
        parser.add_argument(
            "--base-branch",
            default="main",
            help="Base branch to work from (default: main)",
        )
        parser.add_argument(
            "--force-import",
            action="store_true",
            help="Force import of branches even if they already exist",
        )

        # Display options
        parser.add_argument(
            "--no-tui",
            action="store_true",
            help="Disable TUI and stream events to console",
        )
        parser.add_argument(
            "--output",
            choices=["streaming", "json", "quiet"],
            default="streaming",
            help="Output format when using --no-tui (default: streaming)",
        )

        # Authentication mode
        parser.add_argument(
            "--mode",
            choices=["subscription", "api"],
            help="Authentication mode to use (default: auto-detect based on available credentials)",
        )

        # Resource limits
        parser.add_argument(
            "--max-parallel",
            type=int,
            default=20,
            help="Max parallel instances (default: 20)",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=3600,
            help="Timeout per instance in seconds (default: 3600)",
        )

        # Authentication
        parser.add_argument(
            "--oauth-token",
            help="OAuth token for API access (or use CLAUDE_CODE_OAUTH_TOKEN env var)",
        )
        parser.add_argument(
            "--api-key", help="API key for access (or use ANTHROPIC_API_KEY env var)"
        )

        # Other options
        parser.add_argument(
            "--resume", metavar="RUN_ID", help="Resume an interrupted run"
        )
        parser.add_argument(
            "--resume-fresh",
            metavar="RUN_ID",
            help="Resume a run with fresh containers (re-run incomplete instances)",
        )
        parser.add_argument(
            "--list-runs", action="store_true", help="List all previous runs"
        )
        parser.add_argument(
            "--show-run", metavar="RUN_ID", help="Show details of a specific run"
        )
        parser.add_argument(
            "--clean-containers",
            metavar="RUN_ID",
            help="Remove containers and state for a specific run",
        )
        # Alias per spec examples
        parser.add_argument(
            "--cleanup-run",
            metavar="RUN_ID",
            dest="clean_containers",
            help="Alias for --clean-containers",
        )
        parser.add_argument(
            "--config",
            type=Path,
            help="Configuration file (default: orchestrator.yaml if exists)",
        )
        parser.add_argument(
            "--state-dir",
            type=Path,
            default=Path("./orchestrator_state"),
            help="State directory (default: ./orchestrator_state)",
        )
        parser.add_argument(
            "--logs-dir",
            type=Path,
            default=Path("./logs"),
            help="Logs directory (default: ./logs)",
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument(
            "--http-port",
            type=int,
            help="Enable HTTP server on specified port for multi-UI support",
        )

        return parser

    def _get_auth_config(
        self, args: argparse.Namespace, config: Optional[Dict[str, Any]] = None
    ) -> AuthConfig:
        """Get authentication configuration with precedence: CLI > env > .env > config file."""
        # Load configuration sources
        env_config = load_env_config()
        dotenv_config = load_dotenv_config()
        defaults = get_default_config()

        # Build full config with proper precedence
        cli_config = {}
        if hasattr(args, "oauth_token") and args.oauth_token:
            cli_config.setdefault("runner", {})["oauth_token"] = args.oauth_token
        if hasattr(args, "api_key") and args.api_key:
            cli_config.setdefault("runner", {})["api_key"] = args.api_key

        # Merge all sources
        full_config = merge_config(
            cli_config, env_config, dotenv_config, config or {}, defaults
        )

        # Extract auth values from merged config
        oauth_token = full_config.get("runner", {}).get("oauth_token")
        api_key = full_config.get("runner", {}).get("api_key")
        base_url = full_config.get("runner", {}).get("base_url")

        # Apply mode selection logic per spec section 6.1
        # 1. If --mode api specified, use API key
        if hasattr(args, "mode") and args.mode == "api":
            if not api_key:
                self.console.print(
                    "[red]Error: --mode api specified but no API key provided.[/red]\n"
                    "Set ANTHROPIC_API_KEY in:\n"
                    "  - .env file\n"
                    "  - Environment variables\n"
                    "  - Command line: --api-key"
                )
                sys.exit(1)
            # Clear oauth token to ensure API mode
            oauth_token = None
        # 2. If --mode subscription specified, use OAuth token
        elif hasattr(args, "mode") and args.mode == "subscription":
            if not oauth_token:
                self.console.print(
                    "[red]Error: --mode subscription specified but no OAuth token provided.[/red]\n"
                    "Set CLAUDE_CODE_OAUTH_TOKEN in:\n"
                    "  - .env file\n"
                    "  - Environment variables\n"
                    "  - Command line: --oauth-token"
                )
                sys.exit(1)
            # Clear API key to ensure subscription mode
            api_key = None
        # 3. If OAuth token present (and no mode specified), use subscription mode
        elif oauth_token:
            # Subscription mode is default when OAuth token is available
            api_key = None
        # 4. If only API key present, use API mode
        elif api_key:
            # API mode when only API key is available
            pass
        # 5. Otherwise, error with clear message
        else:
            self.console.print(
                "[red]Error: No authentication provided.[/red]\n"
                "Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in:\n"
                "  - .env file\n"
                "  - Environment variables\n"
                "  - Command line: --oauth-token or --api-key"
            )
            sys.exit(1)

        return AuthConfig(oauth_token=oauth_token, api_key=api_key, base_url=base_url)

    def _get_strategy_config(
        self, args: argparse.Namespace, full_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get strategy configuration from args and config file."""
        if full_config is None:
            full_config = {}

        # Start with strategy defaults from config file
        strategy_name = args.strategy or full_config.get("strategy", "simple")
        config = {}

        # Load strategy-specific defaults from config file (support hyphen/underscore aliases)
        if "strategies" in full_config:
            strat_section = full_config["strategies"]
            if strategy_name in strat_section:
                config.update(strat_section[strategy_name])
            else:
                alias1 = strategy_name.replace("-", "_")
                alias2 = strategy_name.replace("_", "-")
                if alias1 in strat_section:
                    config.update(strat_section[alias1])
                elif alias2 in strat_section:
                    config.update(strat_section[alias2])

        # Add model (CLI overrides config)
        config["model"] = args.model or full_config.get(
            "model", "sonnet"
        )

        # Add force_import if specified
        if hasattr(args, "force_import") and args.force_import:
            config["force_import"] = True

        # Parse -S key=value parameters
        if hasattr(args, "strategy_params") and args.strategy_params:
            for param in args.strategy_params:
                if "=" not in param:
                    self.console.print(
                        f"[red]Invalid -S parameter format: {param}[/red]"
                    )
                    self.console.print("Expected format: -S key=value")
                    sys.exit(1)

                key, value = param.split("=", 1)

                # Try to parse value as different types
                # First try int
                try:
                    config[key] = int(value)
                except ValueError:
                    # Then try float
                    try:
                        config[key] = float(value)
                    except ValueError:
                        # Then try boolean
                        if value.lower() in ("true", "yes", "1"):
                            config[key] = True
                        elif value.lower() in ("false", "no", "0"):
                            config[key] = False
                        else:
                            # Otherwise keep as string
                            config[key] = value

        return config

    async def _perform_preflight_checks(self, args: argparse.Namespace) -> bool:
        """
        Perform comprehensive pre-flight checks before starting a run.

        Per spec section 7.1, checks:
        - Docker daemon is running and accessible (already done)
        - Repository exists and has requested base branch
        - Authentication is configured (done elsewhere)
        - Required tools are available in containers
        - Sufficient disk space for logs and results

        Returns:
            True if all checks pass, False otherwise
        """
        from pathlib import Path
        import shutil

        # 1. Check repository exists
        repo_path = Path(args.repo)
        if not repo_path.exists():
            self.console.print(
                f"[red]Error: Repository path does not exist: {repo_path}[/red]"
            )
            return False

        if not repo_path.is_dir():
            self.console.print(
                f"[red]Error: Repository path is not a directory: {repo_path}[/red]"
            )
            return False

        # 2. Check it's a git repository
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            self.console.print(f"[red]Error: Not a git repository: {repo_path}[/red]")
            self.console.print("Initialize with: git init")
            return False

        # 3. Check base branch exists
        try:
            import subprocess

            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "rev-parse",
                    "--verify",
                    args.base_branch,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.console.print(
                    f"[red]Error: Base branch '{args.base_branch}' does not exist[/red]"
                )
                self.console.print("Available branches:")
                branches_result = subprocess.run(
                    ["git", "-C", str(repo_path), "branch", "-a"],
                    capture_output=True,
                    text=True,
                )
                if branches_result.returncode == 0:
                    self.console.print(branches_result.stdout)
                return False
        except (subprocess.SubprocessError, OSError) as e:
            self.console.print(f"[red]Error checking git repository: {e}[/red]")
            return False

        # 4. Check disk space (20GB minimum as per spec)
        try:
            stat = shutil.disk_usage(str(repo_path))
            free_gb = stat.free / (1024**3)
            if free_gb < 20:
                self.console.print(
                    f"[red]Error: Insufficient disk space: {free_gb:.1f}GB free (20GB required)[/red]"
                )
                return False
        except OSError as e:
            self.console.print(
                f"[yellow]Warning: Could not check disk space: {e}[/yellow]"
            )

        # 5. Quick test of Docker image availability (non-blocking)
        # This is a quick check - actual image pull happens later if needed
        try:
            import docker

            client = docker.from_env()
            # Just check if we can connect and list images
            client.images.list()
            client.close()
        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Could not verify Docker images: {e}[/yellow]"
            )
            self.console.print("Images will be pulled on first use if needed.")

        return True

    def _load_config_file(self, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
        """
        Load configuration from orchestrator.yaml file.

        Precedence order:
        1. If config file is specified via args, use that
        2. If orchestrator.yaml exists in current directory, use that
        3. Otherwise return None
        """

        config_path = None

        # Check if config file was specified
        if hasattr(args, "config") and args.config:
            config_path = Path(args.config)
        else:
            # Check for default orchestrator.yaml
            default_path = Path("orchestrator.yaml")
            if default_path.exists():
                config_path = default_path

        if not config_path:
            return None

        if not config_path.exists():
            self.console.print(f"[red]Config file not found: {config_path}[/red]")
            sys.exit(1)

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            logger.debug(f"Loaded configuration from {config_path}")
            return config

        except (FileNotFoundError, yaml.YAMLError, OSError) as e:
            self.console.print(f"[red]Error loading config file: {e}[/red]")
            sys.exit(1)

    async def run_cleanup(self, args: argparse.Namespace) -> int:
        """Clean up containers and state for a specific run."""
        run_id = args.clean_containers
        self.console.print(f"[yellow]Cleaning up run {run_id}...[/yellow]")

        # Create orchestrator just for cleanup
        orchestrator = Orchestrator(state_dir=args.state_dir, logs_dir=args.logs_dir)

        try:
            # Initialize orchestrator
            await orchestrator.initialize()

            # Clean up containers for this specific run
            from ..instance_runner.docker_manager import DockerManager

            docker_manager = DockerManager()
            try:
                await docker_manager.initialize()

                # Find and remove containers with this run_id
                containers = await docker_manager._list_containers(all=True)
                cleaned = 0

                for container in containers:
                    labels = container.labels
                    if labels.get("orchestrator.run_id") == run_id:
                        try:
                            if container.status == "running":
                                await asyncio.to_thread(container.stop)
                            await asyncio.to_thread(container.remove)
                            cleaned += 1
                            self.console.print(f"  Removed container: {container.name}")
                        except Exception as e:
                            self.console.print(
                                f"  [red]Failed to remove {container.name}: {e}[/red]"
                            )
            finally:
                docker_manager.close()

            # Remove state directory for this run
            state_file = args.state_dir / f"{run_id}.json"
            if state_file.exists():
                state_file.unlink()
                self.console.print(f"  Removed state file: {state_file.name}")

            # Remove logs directory for this run
            logs_dir = args.logs_dir / run_id
            if logs_dir.exists():
                import shutil

                shutil.rmtree(logs_dir)
                self.console.print(f"  Removed logs directory: {logs_dir.name}")

            # Remove results directory for this run
            results_dir = Path("./results") / run_id
            if results_dir.exists():
                import shutil

                shutil.rmtree(results_dir)
                self.console.print(f"  Removed results directory: {results_dir.name}")

            self.console.print(
                f"[green]Cleaned up run {run_id} ({cleaned} containers)[/green]"
            )
            return 0

        except (DockerError, OrchestratorError) as e:
            self.console.print(f"[red]Cleanup failed: {e}[/red]")
            return 1
        finally:
            await orchestrator.shutdown()

    async def run_list_runs(self, args: argparse.Namespace) -> int:
        """List all previous runs."""
        from rich.table import Table
        from rich.box import ROUNDED

        try:
            state_dir = args.state_dir
            if not state_dir.exists():
                self.console.print("[yellow]No runs found[/yellow]")
                return 0

            # Find all run directories
            run_dirs = [
                d
                for d in state_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]

            if not run_dirs:
                self.console.print("[yellow]No runs found[/yellow]")
                return 0

            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)

            # Create table
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
                snapshot_file = run_dir / "state.json"

                if snapshot_file.exists():
                    # Load snapshot data
                    import json

                    with open(snapshot_file) as f:
                        data = json.load(f)

                    # Calculate duration
                    started = datetime.fromisoformat(data["started_at"])
                    completed = (
                        datetime.fromisoformat(data["completed_at"])
                        if data["completed_at"]
                        else None
                    )

                    if completed:
                        duration = completed - started
                        duration_str = f"{duration.total_seconds():.0f}s"
                        status = "✓ Completed"
                    else:
                        duration_str = "-"
                        status = "○ Interrupted"

                    # Format values
                    started_str = started.strftime("%Y-%m-%d %H:%M:%S")
                    instances_str = f"{data.get('completed_instances', 0)}/{data.get('total_instances', 0)}"
                    cost_str = f"${data.get('total_cost', 0):.2f}"
                    prompt = data.get("prompt", "")[:40] + (
                        "..." if len(data.get("prompt", "")) > 40 else ""
                    )

                    table.add_row(
                        run_id,
                        status,
                        started_str,
                        duration_str,
                        instances_str,
                        cost_str,
                        prompt,
                    )
                else:
                    # No snapshot, minimal info
                    table.add_row(run_id, "? Unknown", "-", "-", "-", "-", "-")

            self.console.print(table)
            self.console.print(f"\nTotal runs: {len(run_dirs)}")

            return 0

        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error listing runs: {e}[/red]")
            if args.debug:
                self.console.print_exception()
            return 1

    async def run_show_run(self, args: argparse.Namespace) -> int:
        """Show details of a specific run."""
        from rich.panel import Panel
        from rich.tree import Tree

        try:
            run_id = args.show_run
            state_dir = args.state_dir / run_id

            if not state_dir.exists():
                self.console.print(f"[red]Run {run_id} not found[/red]")
                return 1

            # Load snapshot
            snapshot_file = state_dir / "state.json"
            if not snapshot_file.exists():
                self.console.print(f"[red]No snapshot found for run {run_id}[/red]")
                return 1

            import json

            with open(snapshot_file) as f:
                data = json.load(f)

            # Display run summary
            started = datetime.fromisoformat(data["started_at"])
            completed = (
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            )

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

            self.console.print(Panel(summary, title=f"Run Details: {run_id}"))

            # Display strategies
            if data.get("strategies"):
                tree = Tree("[bold]Strategies[/bold]")

                for strat_id, strat_data in data["strategies"].items():
                    strat_node = tree.add(f"{strat_data['name']} ({strat_id})")
                    strat_node.add(f"State: {strat_data['state']}")
                    strat_node.add(f"Started: {strat_data['started_at']}")

                    if strat_data.get("results"):
                        results_node = strat_node.add("Results")
                        for i, result in enumerate(strat_data["results"]):
                            results_node.add(
                                f"Instance {i+1}: Branch {result.get('branch_name', 'unknown')}"
                            )

                self.console.print(tree)

            # Display instances
            if data.get("instances"):
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

                self.console.print(inst_tree)

            # Show available actions
            self.console.print("\n[dim]Available actions:[/dim]")
            if not completed:
                self.console.print(f"  Resume: orchestrator --resume {run_id}")
            self.console.print(f"  View logs: {args.logs_dir}/{run_id}/")

            return 0

        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error showing run: {e}[/red]")
            if args.debug:
                self.console.print_exception()
            return 1

    def _display_detailed_results(
        self, results: List["InstanceResult"], run_id: str, state: Optional[Any] = None
    ) -> None:
        """
        Display detailed results per spec section 7.5.

        Shows:
        - Run header with strategy configuration
        - Results grouped by strategy execution
        - Instance details with metadata
        - Summary statistics
        - Final branches and next steps
        """

        # Header
        self.console.print(f"\n═══ Run Complete: {run_id} ═══\n")

        # Strategy info from state if available
        if state and hasattr(state, "strategies"):
            # Group results by strategy
            strategy_results = {}
            for result in results:
                strat_id = result.metadata.get("strategy_execution_id", "unknown")
                if strat_id not in strategy_results:
                    strategy_results[strat_id] = []
                strategy_results[strat_id].append(result)

            # Display strategy configuration
            for strat_id, strat_info in state.strategies.items():
                self.console.print(f"[bold]Strategy: {strat_info.strategy_name}[/bold]")
                if strat_info.config:
                    for key, value in strat_info.config.items():
                        self.console.print(f"  {key}: {value}")
                self.console.print()

            self.console.print(f"[bold]Runs:[/bold] {len(state.strategies)}\n")

            # Results by strategy
            self.console.print("[bold]Results by strategy:[/bold]\n")

            for idx, (strat_id, strat_info) in enumerate(state.strategies.items(), 1):
                strat_results = strategy_results.get(strat_id, [])
                self.console.print(
                    f"[bold]Strategy #{idx} ({strat_info.strategy_name}):[/bold]"
                )

                # Find the selected result for best-of-n strategies
                selected_branch = None
                if strat_info.strategy_name == "best-of-n" and strat_info.results:
                    # The first result is typically the selected one
                    selected_branch = (
                        strat_info.results[0].get("branch_name")
                        if strat_info.results
                        else None
                    )

                # Display each instance
                for result in strat_results:
                    # Status emoji
                    status = "✓" if result.success else "✗"

                    # Format duration
                    duration = (
                        f"{result.duration_seconds:.0f}s"
                        if result.duration_seconds
                        else "-"
                    )
                    if result.duration_seconds and result.duration_seconds >= 60:
                        minutes = int(result.duration_seconds // 60)
                        seconds = int(result.duration_seconds % 60)
                        duration = f"{minutes}m {seconds}s"

                    # Format cost and tokens
                    cost = (
                        f"${result.metrics.get('total_cost', 0):.2f}"
                        if result.metrics
                        else "$0.00"
                    )
                    tokens = (
                        result.metrics.get("total_tokens", 0) if result.metrics else 0
                    )
                    token_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)

                    # Build instance line
                    line = f"  {status} {result.branch_name or 'no-branch'}  {duration} • {cost} • {token_str} tokens"

                    if result.success:
                        self.console.print(
                            line, style="green" if status == "✓" else "red"
                        )
                    else:
                        error_msg = result.error or "Unknown error"
                        self.console.print(f"{line}  [red]Failed: {error_msg}[/red]")

                    # Show metadata if present
                    if result.metadata:
                        metadata_items = []
                        if "score" in result.metadata:
                            metadata_items.append(f"score={result.metadata['score']}")
                        if "complexity" in result.metadata:
                            metadata_items.append(
                                f"complexity={result.metadata['complexity']}"
                            )
                        if "test_coverage" in result.metadata:
                            metadata_items.append(
                                f"test_coverage={result.metadata['test_coverage']}%"
                            )

                        if metadata_items:
                            self.console.print(
                                f"    metadata: {', '.join(metadata_items)}"
                            )

                    # Show final message if present
                    if result.final_message:
                        # Truncate very long messages but show enough to be useful
                        message = result.final_message
                        if len(message) > 500:
                            message = message[:497] + "..."
                        self.console.print(f"    [dim]final_message:[/dim] {message}")

                # Show selected result for best-of-n
                if selected_branch:
                    self.console.print(f"  → Selected: {selected_branch}")

                self.console.print()  # Empty line between strategies

        else:
            # Fallback display without state info
            for result in results:
                status = "✓" if result.success else "✗"
                duration = (
                    f"{result.duration_seconds:.1f}s"
                    if result.duration_seconds
                    else "-"
                )
                cost = (
                    f"${result.metrics.get('total_cost', 0):.2f}"
                    if result.metrics
                    else "$0.00"
                )

                self.console.print(
                    f"{status} {result.branch_name or 'no-branch'}  {duration} • {cost}"
                )

                # Show final message if present
                if result.final_message:
                    # Truncate very long messages but show enough to be useful
                    message = result.final_message
                    if len(message) > 500:
                        message = message[:497] + "..."
                    self.console.print(f"  [dim]final_message:[/dim] {message}")

        # Summary section
        self.console.print("[bold]Summary:[/bold]")

        # Calculate totals
        total_duration = sum(r.duration_seconds or 0 for r in results)
        total_cost = sum(r.metrics.get("total_cost", 0) for r in results if r.metrics)
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)

        # Format duration
        if total_duration >= 60:
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{int(total_duration)}s"

        self.console.print(f"  Total Duration: {duration_str}")
        self.console.print(f"  Total Cost: ${total_cost:.2f}")
        self.console.print(
            f"  Success Rate: {success_count}/{total_count} instances ({success_count/total_count*100:.0f}%)"
        )

        # Final branches
        final_branches = [r.branch_name for r in results if r.branch_name and r.success]
        if final_branches:
            self.console.print(
                f"\n[bold]Final branches ({len(final_branches)}):[/bold]"
            )
            for branch in final_branches:
                self.console.print(f"  {branch}")

        # Show results directory
        self.console.print(f"\n[dim]Full results: ./results/{run_id}/[/dim]")

        # Show next steps
        if final_branches:
            self.console.print("\n[dim]To compare implementations:[/dim]")
            self.console.print(f"  git diff main..{final_branches[0]}")
            if len(final_branches) > 1:
                self.console.print(f"  git checkout {final_branches[1]}")

            self.console.print("\n[dim]To merge selected solution:[/dim]")
            self.console.print(f"  git merge {final_branches[0]}")

    async def run_orchestrator(self, args: argparse.Namespace) -> int:
        """
        Run the orchestrator with or without TUI.

        Args:
            args: Parsed command line arguments

        Returns:
            Exit code
        """
        # Determine run_id for logging
        if args.resume:
            run_id = args.resume
        elif args.resume_fresh:
            run_id = args.resume_fresh
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"

        # Setup structured logging
        from .utils.structured_logging import setup_structured_logging

        setup_structured_logging(
            logs_dir=args.logs_dir,
            run_id=run_id,
            debug=args.debug,
            quiet=args.no_tui and args.output == "quiet",
            no_tui=args.no_tui,
        )

        # Setup log rotation and cleanup old logs
        from .utils.log_rotation import cleanup_old_logs, setup_log_rotation_task

        try:
            cleanup_count = cleanup_old_logs(args.logs_dir)
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old log directories")

            # Setup periodic cleanup and size-based rotation for logs
            # Convert configured byte limit to MB for rotation helper
            try:
                max_bytes = full_config.get("logging", {}).get("max_file_size", 10485760)
                max_mb = int(max_bytes / (1024 * 1024)) if isinstance(max_bytes, (int, float)) else 100
            except Exception:
                max_mb = 100
            asyncio.create_task(setup_log_rotation_task(args.logs_dir, max_size_mb=max_mb))
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to setup log rotation: {e}")

        # Check platform and show recommendations if needed
        from .utils.platform_utils import (
            validate_docker_setup,
            get_platform_recommendations,
        )

        # Validate Docker setup
        docker_valid, docker_error = validate_docker_setup()
        if not docker_valid:
            self.console.print(f"[red]Docker Setup Error:[/red] {docker_error}")
            return 1

        # Show platform recommendations if any
        recommendations = get_platform_recommendations()
        if recommendations and not args.no_tui:
            for rec in recommendations:
                self.console.print(f"[yellow]Platform Note:[/yellow] {rec}")

        # Handle cleanup mode
        if args.clean_containers:
            return await self.run_cleanup(args)

        # Handle list-runs mode
        if args.list_runs:
            return await self.run_list_runs(args)

        # Handle show-run mode
        if args.show_run:
            return await self.run_show_run(args)

        # Check for prompt or resume
        if not args.prompt and not args.resume and not args.resume_fresh:
            self.console.print(
                "[red]Error: Either provide a prompt or use --resume/--resume-fresh[/red]"
            )
            return 1

        # Perform pre-flight checks for new runs
        if not args.resume and not args.resume_fresh:
            if not await self._perform_preflight_checks(args):
                return 1

        # Load configuration from file if specified
        config = self._load_config_file(args)

        # Load all configuration sources
        env_config = load_env_config()
        dotenv_config = load_dotenv_config()  # Load .env file separately
        defaults = get_default_config()

        # Build CLI config dict from args
        cli_config = {}
        if args.max_parallel:
            cli_config.setdefault("orchestration", {})[
                "max_parallel_instances"
            ] = args.max_parallel
        if hasattr(args, "timeout") and args.timeout:
            cli_config.setdefault("runner", {})["timeout"] = args.timeout
        if args.model:
            cli_config["model"] = args.model
        if args.strategy:
            cli_config["strategy"] = args.strategy
        if args.state_dir:
            cli_config.setdefault("orchestration", {})["state_dir"] = args.state_dir
        if args.logs_dir:
            cli_config.setdefault("orchestration", {})["logs_dir"] = args.logs_dir
        if args.output:
            cli_config["output"] = args.output
        if args.http_port:
            cli_config["http_port"] = args.http_port
        if args.debug:
            cli_config.setdefault("logging", {})["level"] = "DEBUG"
        # Add CLI auth args
        if hasattr(args, "oauth_token") and args.oauth_token:
            cli_config.setdefault("runner", {})["oauth_token"] = args.oauth_token
        if hasattr(args, "api_key") and args.api_key:
            cli_config.setdefault("runner", {})["api_key"] = args.api_key

        # Merge configurations with proper precedence: CLI > env > .env > file > defaults
        full_config = merge_config(
            cli_config, env_config, dotenv_config, config or {}, defaults
        )

        # Configure container limits from merged config
        memory_limit_str = full_config["runner"]["memory_limit"]
        if isinstance(memory_limit_str, str):
            if memory_limit_str.endswith("g") or memory_limit_str.endswith("G"):
                memory_gb = int(memory_limit_str[:-1])
            else:
                memory_gb = int(memory_limit_str)
        else:
            memory_gb = int(memory_limit_str)

        container_limits = ContainerLimits(
            cpu_count=int(full_config["runner"]["cpu_limit"]),
            memory_gb=memory_gb,
            memory_swap_gb=memory_gb,  # Same as memory limit
        )

        retry_config = RetryConfig(max_attempts=3)

        # Create auth config from merged configuration
        oauth_token = full_config.get("runner", {}).get("oauth_token")
        api_key = full_config.get("runner", {}).get("api_key")
        base_url = full_config.get("runner", {}).get("base_url")

        if not oauth_token and not api_key:
            self.console.print(
                "[red]Error: No authentication provided.[/red]\n"
                "Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in:\n"
                "  - .env file\n"
                "  - Environment variables\n"
                "  - Command line: --oauth-token or --api-key"
            )
            return 1

        auth_config = AuthConfig(
            oauth_token=oauth_token, api_key=api_key, base_url=base_url
        )

        # Get orchestration settings from merged config
        max_parallel = full_config["orchestration"]["max_parallel_instances"]
        state_dir = full_config.get("orchestration", {}).get(
            "state_dir"
        ) or full_config.get("state_dir", Path("./orchestrator_state"))
        logs_dir = full_config.get("orchestration", {}).get(
            "logs_dir"
        ) or full_config.get("logs_dir", Path("./logs"))

        # Log configuration sources (only in debug mode)
        if full_config.get("debug") or args.debug:
            self.console.print("[dim]Configuration loaded from:[/dim]")
            if cli_config:
                self.console.print("  - Command line arguments")
            if env_config:
                self.console.print("  - ORCHESTRATOR_* environment variables")
            if dotenv_config:
                self.console.print("  - .env file")
            if config:
                self.console.print(
                    f"  - Config file: {args.config or 'orchestrator.yaml'}"
                )
            self.console.print("  - Built-in defaults")

        # Create orchestrator
        self.orchestrator = Orchestrator(
            max_parallel_instances=int(max_parallel),
            state_dir=Path(state_dir),
            logs_dir=Path(logs_dir),
            container_limits=container_limits,
            retry_config=retry_config,
            auth_config=auth_config,
            snapshot_interval=int(full_config.get("orchestration", {}).get("snapshot_interval", 30)),
            event_buffer_size=int(full_config.get("orchestration", {}).get("event_buffer_size", 10000)),
            container_retention_failed_hours=int(full_config.get("orchestration", {}).get("container_retention_failed", 86400) // 3600 if isinstance(full_config.get("orchestration", {}).get("container_retention_failed", 86400), int) else 24),
            container_retention_success_hours=int(full_config.get("orchestration", {}).get("container_retention_success", 7200) // 3600 if isinstance(full_config.get("orchestration", {}).get("container_retention_success", 7200), int) else 2),
        )

        # Initialize orchestrator
        await self.orchestrator.initialize()

        # Start HTTP server if requested
        http_port = args.http_port or full_config.get("http_port")
        if http_port:
            await self.orchestrator.start_http_server(http_port)
            self.console.print(
                f"[blue]HTTP server started on port {http_port}[/blue]"
            )

        # Determine run mode
        if args.no_tui:
            # Headless mode
            return await self._run_headless(args, full_config)
        else:
            # TUI mode
            return await self._run_with_tui(args, run_id, full_config)

    async def _run_headless(
        self, args: argparse.Namespace, full_config: Dict[str, Any]
    ) -> int:
        """Run orchestrator in headless mode."""
        output_mode = args.output or "streaming"

        # Set up event subscriptions for output
        if output_mode == "streaming":
            # Subscribe to key events for console output
            def print_event(event):
                event_type = event.get("type", "unknown")
                data = event.get("data", {})

                # Format event based on type
                if event_type == "instance.started":
                    self.console.print(
                        f"[blue]Starting instance:[/blue] {data.get('prompt', '')[:80]}..."
                    )
                elif event_type == "instance.completed":
                    success = data.get("success", False)
                    branch = data.get("branch_name", "unknown")
                    duration = data.get("duration_seconds", 0)
                    if success:
                        self.console.print(
                            f"[green]✓ Instance completed:[/green] {branch} ({duration:.1f}s)"
                        )
                    else:
                        self.console.print("[red]✗ Instance failed[/red]")
                elif event_type == "instance.failed":
                    error = data.get("error", "Unknown error")
                    self.console.print(f"[red]Instance failed:[/red] {error}")
                elif event_type == "strategy.completed":
                    self.console.print("[green]Strategy completed[/green]")

            # Subscribe to events
            self.orchestrator.subscribe("instance.started", print_event)
            self.orchestrator.subscribe("instance.completed", print_event)
            self.orchestrator.subscribe("instance.failed", print_event)
            self.orchestrator.subscribe("strategy.completed", print_event)

        try:
            if args.resume:
                # Resume existing run
                run_id = args.resume
                self.console.print(f"[blue]Resuming run {run_id}...[/blue]")

                # Resume the run
                result = await self.orchestrator.resume_run(run_id)
            elif args.resume_fresh:
                # Resume with fresh containers
                run_id = args.resume_fresh
                self.console.print(
                    f"[blue]Resuming run {run_id} with fresh containers...[/blue]"
                )

                # Resume with fresh containers (force_fresh=True)
                result = await self.orchestrator.resume_run(run_id, force_fresh=True)
            else:
                # Start new run
                self.console.print(f"[blue]Starting strategy: {args.strategy}[/blue]")
                strategy_config = self._get_strategy_config(args, full_config)

                result = await self.orchestrator.run_strategy(
                    strategy_name=args.strategy,
                    prompt=args.prompt,
                    repo_path=args.repo,
                    base_branch=args.base_branch,
                    runs=args.runs,
                    strategy_config=strategy_config,
                )

            # Output results based on mode
            if output_mode == "json":
                # Convert results to JSON-serializable format
                if isinstance(result, list):
                    # List of InstanceResult objects
                    result_data = [
                        {
                            "session_id": r.session_id,
                            "branch_name": r.branch_name,
                            "success": r.success,
                            "error": r.error,
                            "status": r.status,
                            "container_name": r.container_name,
                            "has_changes": r.has_changes,
                            "duration_seconds": r.duration_seconds,
                            "metrics": r.metrics,
                            "final_message": r.final_message,
                        }
                        for r in result
                    ]
                else:
                    result_data = result
                print(json.dumps(result_data, indent=2, default=str))
            elif output_mode == "quiet":
                # Just return code
                pass
            else:
                # Streaming mode - events are already printed
                # Use detailed display
                if isinstance(result, list) and result:
                    # Try to get state for more detailed display
                    state = None
                    actual_run_id = None
                    if (
                        hasattr(self.orchestrator, "state_manager")
                        and self.orchestrator.state_manager
                    ):
                        state = self.orchestrator.state_manager.get_current_state()
                        if state:
                            actual_run_id = state.run_id

                    # Use the run_id from state if available
                    display_run_id = actual_run_id if actual_run_id else "unknown"
                    self._display_detailed_results(result, display_run_id, state)
                else:
                    self.console.print("\n[green]Run completed[/green]")

            # Shutdown orchestrator to stop background tasks
            await self.orchestrator.shutdown()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            # Show resume command if not in resume mode
            if not args.resume and not args.resume_fresh:
                # Use the run_id from orchestrator's current state
                if self.orchestrator and self.orchestrator.state_manager:
                    state = self.orchestrator.state_manager.current_state
                    if state and state.run_id:
                        self.console.print("\n[blue]To resume this run:[/blue]")
                        self.console.print(f"  orchestrator --resume {state.run_id}")
                        self.console.print(
                            f"  orchestrator --resume-fresh {state.run_id}  # With fresh containers\n"
                        )
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            return 130
        except (OrchestratorError, DockerError, ValidationError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            if args.debug:
                self.console.print_exception()
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            return 1

    async def _run_with_tui(
        self, args: argparse.Namespace, run_id: str, full_config: Dict[str, Any]
    ) -> int:
        """Run orchestrator with TUI display."""
        # Create TUI display
        self.tui_display = TUIDisplay(
            console=self.console, refresh_rate=0.1, state_poll_interval=3.0
        )

        # Start orchestrator and TUI together
        try:
            # Use the passed run_id (already determined in run() method)

            events_file = args.logs_dir / run_id / "events.jsonl"

            # Ensure the logs directory exists
            events_file.parent.mkdir(parents=True, exist_ok=True)

            # Create tasks for both orchestrator and TUI
            orchestrator_task = None
            tui_task = None

            if args.resume:
                # Resume existing run
                orchestrator_task = asyncio.create_task(
                    self.orchestrator.resume_run(run_id)
                )
            elif args.resume_fresh:
                # Resume with fresh containers
                orchestrator_task = asyncio.create_task(
                    self.orchestrator.resume_run(run_id, force_fresh=True)
                )
            else:
                # Start new run
                strategy_config = self._get_strategy_config(args, full_config)

                orchestrator_task = asyncio.create_task(
                    self.orchestrator.run_strategy(
                        strategy_name=args.strategy,
                        prompt=args.prompt,
                        repo_path=args.repo,
                        base_branch=args.base_branch,
                        runs=args.runs,
                        strategy_config=strategy_config,
                        run_id=run_id,  # Pass the run_id we generated
                    )
                )

            # Start TUI with proper event file waiting
            async def start_tui():
                # Wait for events file to be created by orchestrator
                max_wait = 10  # seconds
                start_wait = asyncio.get_event_loop().time()
                while not events_file.exists():
                    if asyncio.get_event_loop().time() - start_wait > max_wait:
                        raise RuntimeError(f"Events file not created after {max_wait}s")
                    await asyncio.sleep(0.01)  # Very short check interval

                # Don't print to console here - it interferes with the TUI
                logger.info(f"Starting TUI with events file: {events_file}")
                await self.tui_display.run(
                    orchestrator=self.orchestrator,
                    events_file=events_file,
                    from_offset=0,
                )

            tui_task = asyncio.create_task(start_tui())

            # Create a task to monitor the shutdown event
            async def monitor_shutdown():
                if self.shutdown_event:
                    await self.shutdown_event.wait()
                    # Shutdown was requested
                    return "shutdown"

            shutdown_task = (
                asyncio.create_task(monitor_shutdown()) if self.shutdown_event else None
            )

            # Wait for any task to complete
            tasks = [orchestrator_task, tui_task]
            if shutdown_task:
                tasks.append(shutdown_task)

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check if shutdown was requested
            shutdown_requested = False
            for task in done:
                if task == shutdown_task and task.result() == "shutdown":
                    shutdown_requested = True
                    break

            # Get the result from orchestrator
            result = None
            if not shutdown_requested:
                for task in done:
                    if task == orchestrator_task:
                        result = task.result()
                        break

            # Shutdown orchestrator to stop background tasks
            await self.orchestrator.shutdown()

            # Stop TUI gracefully
            if self.tui_display:
                await self.tui_display.stop()

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception) as e:
                    # Log but don't fail - we're shutting down anyway
                    if not isinstance(e, asyncio.CancelledError):
                        logger.warning(f"Error while cancelling task: {e}")
                    pass

            # If shutdown was requested, show resume command
            if shutdown_requested and not args.resume and not args.resume_fresh:
                self.console.print("\n[blue]To resume this run:[/blue]")
                self.console.print(f"  orchestrator --resume {run_id}")
                self.console.print(
                    f"  orchestrator --resume-fresh {run_id}  # With fresh containers\n"
                )
                return 130

            # Get actual run statistics from orchestrator state
            state = self.orchestrator.get_current_state()

            # Print final result
            if state:
                # Use actual instance counts from state
                total_instances = state.total_instances
                completed_instances = state.completed_instances
                failed_instances = state.failed_instances
                total_cost = state.total_cost

                self.console.print(
                    f"\n[green]Run completed: {completed_instances}/{total_instances} instances succeeded[/green]"
                )
                if failed_instances > 0:
                    self.console.print(
                        f"[red]Failed instances: {failed_instances}[/red]"
                    )
                self.console.print(f"Total cost: ${total_cost:.4f}")
                self.console.print(f"Total instances: {total_instances}")

                # Show strategy results if available
                if isinstance(result, list) and result:
                    self.console.print(
                        f"\nStrategy returned {len(result)} final result(s):"
                    )
                    for i, r in enumerate(result):
                        if r.branch_name:
                            self.console.print(f"  [{i+1}] Branch: {r.branch_name}")
                            # Display ALL metrics fields
                            if r.metrics:
                                for key, value in sorted(r.metrics.items()):
                                    # Format the value nicely
                                    if isinstance(value, float):
                                        formatted_value = (
                                            f"{value:.4f}"
                                            if value < 100
                                            else f"{value:.2f}"
                                        )
                                    elif isinstance(value, bool):
                                        formatted_value = "Yes" if value else "No"
                                    elif value is None:
                                        formatted_value = "N/A"
                                    else:
                                        formatted_value = str(value)
                                    self.console.print(
                                        f"      {key}: {formatted_value}"
                                    )
                            # Display final message if present
                            if r.final_message:
                                # Truncate very long messages
                                message = r.final_message
                                if len(message) > 500:
                                    message = message[:497] + "..."
                                self.console.print(f"      [dim]final_message:[/dim] {message}")
                            
                            # Display metadata if present
                            if r.metadata:
                                # Show specific metadata fields if they exist
                                metadata_items = []
                                if "score" in r.metadata:
                                    metadata_items.append(f"score={r.metadata['score']}")
                                if "complexity" in r.metadata:
                                    metadata_items.append(f"complexity={r.metadata['complexity']}")
                                if "test_coverage" in r.metadata:
                                    metadata_items.append(f"test_coverage={r.metadata['test_coverage']}%")
                                if "bug_confirmed" in r.metadata:
                                    metadata_items.append(f"bug_confirmed={r.metadata['bug_confirmed']}")
                                if "bug_report_branch" in r.metadata:
                                    metadata_items.append(f"bug_report_branch={r.metadata['bug_report_branch']}")
                                    
                                # Show any other metadata keys not already displayed
                                displayed_keys = {"score", "complexity", "test_coverage", "bug_confirmed", "bug_report_branch", "strategy_execution_id", "model"}
                                for key, value in r.metadata.items():
                                    if key not in displayed_keys:
                                        metadata_items.append(f"{key}={value}")
                                
                                if metadata_items:
                                    self.console.print(f"      [dim]metadata:[/dim] {', '.join(metadata_items)}")
            else:
                # Fallback to result-based counting
                if isinstance(result, list) and result:
                    successful = sum(1 for r in result if r.success)
                    total = len(result)
                    total_cost = sum(r.metrics.get("total_cost", 0) for r in result)
                    self.console.print(
                        f"\n[green]Run completed: {successful}/{total} instances succeeded[/green]"
                    )
                    self.console.print(f"Total cost: ${total_cost:.4f}")
                    self.console.print(f"Total instances: {total}")
                else:
                    self.console.print("\n[green]Run completed[/green]")

            return 0

        except KeyboardInterrupt:
            # This should rarely be reached now due to signal handler
            # But keep it for safety
            return 130
        except (OrchestratorError, DockerError, ValidationError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            if args.debug:
                self.console.print_exception()
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            return 1

    async def run(self, args: argparse.Namespace) -> int:
        """
        Main entry point for CLI.

        Args:
            args: Parsed arguments

        Returns:
            Exit code
        """
        return await self.run_orchestrator(args)


def main():
    """Main entry point."""
    # Note: .env file is now loaded as a separate configuration layer
    # in the merge_config hierarchy, not directly into environment

    # Create parser
    parser = OrchestratorCLI.create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Create and run CLI
    cli = OrchestratorCLI()

    # Set up a shared shutdown event
    shutdown_event = asyncio.Event()

    # Set up signal handler that works with asyncio
    import signal

    def signal_handler(signum, frame):
        # On first Ctrl+C, set the shutdown event
        if not shutdown_event.is_set():
            print("\n\nShutting down gracefully... (Press Ctrl+C again to force exit)")
            shutdown_event.set()
        else:
            # On second Ctrl+C, force exit
            print("\nForce exit!")
            os._exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    # Pass the shutdown event to CLI
    cli.shutdown_event = shutdown_event

    # Run async
    try:
        exit_code = asyncio.run(cli.run(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
