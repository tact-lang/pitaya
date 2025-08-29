#!/usr/bin/env python3
"""
Pitaya CLI - runs AI coding agents with TUI display.

This is the primary entry point that combines:
- Strategy execution
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

from . import __version__

from .config import (
    load_env_config,
    load_dotenv_config,
    load_global_config,
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
    """Main Pitaya CLI application."""

    def __init__(self):
        """Initialize CLI."""
        self.console = Console()
        self.orchestrator = None
        self.tui_display = None
        self.shutdown_event = None

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser for Pitaya with clear grouping and examples."""
        parser = argparse.ArgumentParser(
            prog="pitaya",
            description=(
                "Orchestrate AI coding agents with pluggable strategies and a clean TUI.\n"
                "Quote your prompt; pass strategy params with -S key=value."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Quick examples:\n"
                "  # One-shot\n"
                '  pitaya "implement auth" --strategy simple\n\n'
                "  # Best-of-N with scoring\n"
                '  pitaya "fix bug in user.py" --strategy best-of-n -S n=5 -S scorer_model=opus\n\n'
                "  # Iterative refine (3 rounds)\n"
                '  pitaya "refactor module" --strategy iterative -S iterations=3\n\n'
                "  # Bug finding (target area)\n"
                '  pitaya "find a bug" --strategy bug-finding -S target_area=src/parser\n\n'
                "  # Custom strategy (module or file)\n"
                '  pitaya "task" --strategy examples.fanout_two:FanOutTwoStrategy\n'
                '  pitaya "task" --strategy ./examples/propose_refine.py\n\n'
                "  # Headless modes\n"
                '  pitaya "task" --no-tui --output streaming --verbose\n'
                '  pitaya "task" --json\n\n'
                "  # Run management\n"
                "  pitaya --list-runs\n"
                "  pitaya --show-run run_20250114_123456\n"
                "  pitaya --resume run_20250114_123456\n\n"
                "Tips:\n"
                "  • Built-ins: "
                + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys())))
                + "\n"
                "  • Strategies accept -S key=value (numbers/bools auto-parsed).\n"
                "  • --model picks the agent model; --plugin chooses the runner.\n"
            ),
        )

        # Global flags
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )

        # Main argument - the task prompt or subcommand (doctor | config)
        parser.add_argument(
            "prompt",
            nargs="?",
            help="Task prompt, or 'doctor'/'config' for utility modes",
        )
        # Optional secondary token for 'config print'
        parser.add_argument(
            "subcommand",
            nargs="?",
            help="Subcommand for special modes (e.g., 'print' for 'config print')",
        )

        # Strategy group
        g_strategy = parser.add_argument_group("Strategy")
        g_strategy.add_argument(
            "--strategy",
            default="simple",
            help=(
                "Execution strategy. Built-ins: "
                + ", ".join(sorted(list(AVAILABLE_STRATEGIES.keys())))
                + ". Or pass a file/module: path/to/strategy.py[:Class] or package.module[:Class]"
            ),
        )
        g_strategy.add_argument(
            "-S",
            action="append",
            dest="strategy_params",
            metavar="KEY=VALUE",
            help="Strategy parameter (repeatable)",
        )
        g_strategy.add_argument(
            "--set",
            action="append",
            dest="strategy_params",
            metavar="KEY=VALUE",
            help="Alias for -S",
        )
        g_strategy.add_argument(
            "--runs",
            type=int,
            default=1,
            help="Parallel strategy executions (default: 1)",
        )

        # Model & plugin group
        try:
            from .instance_runner.plugins import AVAILABLE_PLUGINS as _APLUG

            _plugin_choices = sorted(list(_APLUG.keys()))
        except Exception:
            _plugin_choices = ["claude-code"]
        g_model = parser.add_argument_group("Model & Plugin")
        g_model.add_argument(
            "--model",
            type=str,
            default="sonnet",
            help="Model name (plugin-agnostic)",
        )
        g_model.add_argument(
            "--plugin",
            choices=_plugin_choices,
            default="claude-code",
            help="Runner plugin (e.g., claude-code, codex)",
        )
        g_model.add_argument(
            "--docker-image",
            type=str,
            help="Override Docker image (e.g., myrepo/custom:tag)",
        )
        # Agent CLI passthrough
        g_model.add_argument(
            "--cli-arg",
            action="append",
            dest="agent_cli_arg",
            metavar="ARG",
            help="Pass-through single argument to the agent CLI (repeatable)",
        )
        g_model.add_argument(
            "--cli-args",
            dest="agent_cli_args_str",
            metavar="'ARGS'",
            help="Quoted list of arguments to pass to the agent CLI (e.g., --cli-args '-c key=\"v\" --flag')",
        )

        # Repository group
        g_repo = parser.add_argument_group("Repository")
        g_repo.add_argument(
            "--repo", type=Path, default=Path.cwd(), help="Path to git repository"
        )
        g_repo.add_argument(
            "--base-branch", default="main", help="Base branch to work from"
        )
        g_repo.add_argument(
            "--require-clean-wt",
            action="store_true",
            help="Fail if working tree is dirty",
        )
        g_repo.add_argument(
            "--allow-overwrite-protected-refs",
            action="store_true",
            help="Allow overwriting protected refs (use with extreme caution)",
        )

        # Display & output group
        g_display = parser.add_argument_group("Display & Output")
        g_display.add_argument(
            "--no-tui",
            action="store_true",
            help="Disable TUI; stream output to console",
        )
        g_display.add_argument(
            "--display",
            choices=["auto", "detailed", "compact", "dense"],
            default="auto",
            help="TUI density (default: auto)",
        )
        g_display.add_argument(
            "--output",
            choices=["streaming", "json", "quiet"],
            default="streaming",
            help="Output format when --no-tui (default: streaming)",
        )
        g_display.add_argument(
            "--json",
            action="store_true",
            help="Convenience: JSON output (implies --no-tui --output json)",
        )
        g_display.add_argument(
            "--no-emoji", action="store_true", help="Disable emoji in console output"
        )
        g_display.add_argument(
            "--show-ids",
            choices=["short", "full"],
            default="short",
            help="Identifier verbosity in streaming output",
        )
        g_display.add_argument(
            "--verbose",
            action="store_true",
            help="More verbose streaming logs (container phases, imports)",
        )

        # Auth group
        g_auth = parser.add_argument_group("Auth")
        g_auth.add_argument(
            "--mode",
            choices=["subscription", "api"],
            help="Auth mode (default: auto-detect)",
        )
        g_auth.add_argument(
            "--oauth-token",
            help="OAuth token (or set CLAUDE_CODE_OAUTH_TOKEN)",
        )
        g_auth.add_argument(
            "--api-key",
            help="API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY)",
        )
        g_auth.add_argument(
            "--base-url",
            help="Custom API base URL (e.g., OpenAI/Anthropic proxy)",
        )

        # Execution & limits group
        g_limits = parser.add_argument_group("Execution & Limits")
        g_limits.add_argument(
            "--max-parallel",
            type=int,
            default=None,
            help="Max parallel instances (auto if omitted)",
        )
        g_limits.add_argument(
            "--max-startup-parallel",
            type=int,
            default=None,
            help="Max parallel startup initializations (default: min(10,max-parallel))",
        )
        g_limits.add_argument(
            "--timeout", type=int, default=3600, help="Timeout per instance (seconds)"
        )
        # Legacy parallel presets removed; use --max-parallel or auto
        g_limits.add_argument(
            "--force-commit",
            action="store_true",
            help="Force a git commit in the workspace after agent finishes (if there are changes)",
        )
        g_limits.add_argument(
            "--randomize-queue",
            dest="randomize_queue",
            action="store_true",
            help="Execute queued instances in random order instead of FIFO",
        )

        # Config, state & logs group
        g_state = parser.add_argument_group("Config, State & Logs")
        g_state.add_argument(
            "--config",
            type=Path,
            help="Config file (default: pitaya.yaml if present)",
        )
        g_state.add_argument(
            "--state-dir",
            type=Path,
            default=Path("./pitaya_state"),
            help="State directory",
        )
        g_state.add_argument(
            "--logs-dir", type=Path, default=Path("./logs"), help="Logs directory"
        )
        g_state.add_argument(
            "--redact",
            choices=["true", "false"],
            default="true",
            help="For 'config print': redact secrets (default: true)",
        )

        # Maintenance group
        g_maint = parser.add_argument_group("Maintenance")
        g_maint.add_argument(
            "--resume", metavar="RUN_ID", help="Resume an interrupted run"
        )
        g_maint.add_argument(
            "--list-runs", action="store_true", help="List previous runs"
        )
        g_maint.add_argument("--show-run", metavar="RUN_ID", help="Show run details")
        # Prune commands removed; cleanup is automatic
        # Cleanup commands removed for simpler UX

        # Diagnostics group
        g_diag = parser.add_argument_group("Diagnostics")
        g_diag.add_argument(
            "--yes", action="store_true", help="Assume 'yes' for prompts"
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
        if hasattr(args, "base_url") and args.base_url:
            cli_config.setdefault("runner", {})["base_url"] = args.base_url
        # Include plugin selection at top-level for strategy-agnostic wiring
        if hasattr(args, "plugin") and args.plugin:
            cli_config["plugin_name"] = args.plugin

        # Merge all sources
        full_config = merge_config(
            cli_config, env_config, dotenv_config, config or {}, defaults
        )

        # Extract auth values from merged config; select by plugin for consistency
        runner_cfg = full_config.get("runner", {})
        plugin_name = full_config.get("plugin_name") or getattr(
            args, "plugin", "claude-code"
        )

        if str(plugin_name) == "codex":
            # Prefer OpenAI keys; allow CLI overrides via generic api_key/base_url
            api_key = runner_cfg.get("api_key") or runner_cfg.get("openai_api_key")
            base_url = runner_cfg.get("base_url") or runner_cfg.get("openai_base_url")
            oauth_token = None
        else:
            # Claude Code: prefer OAuth, then Anthropic API key
            oauth_token = runner_cfg.get("oauth_token")
            api_key = runner_cfg.get("api_key") or runner_cfg.get("anthropic_api_key")
            base_url = runner_cfg.get("base_url") or runner_cfg.get(
                "anthropic_base_url"
            )

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
        # 5. Otherwise, error with clear message (fail fast for both providers)
        else:
            if str(plugin_name) == "codex":
                self.console.print(
                    "[red]Error: Missing credentials for OpenAI‑compatible provider.[/red]\n"
                    "Set OPENAI_API_KEY in:\n"
                    "  - .env file (OPENAI_API_KEY=...)\n"
                    "  - Environment variables (export OPENAI_API_KEY=...)\n"
                    "  - Command line: --api-key\n"
                    "Optionally set --base-url (or OPENAI_BASE_URL) for non-default endpoints."
                )
            else:
                self.console.print(
                    "[red]Error: Missing credentials for Anthropic.[/red]\n"
                    "Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in:\n"
                    "  - .env file\n"
                    "  - Environment variables\n"
                    "  - Command line: --oauth-token or --api-key"
                )
            return AuthConfig(oauth_token=None, api_key=None, base_url=None)

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

        # Add model (CLI overrides config); strategies remain plugin-agnostic
        config["model"] = args.model or full_config.get("model", "sonnet")

        # --force-import removed; rely on import_conflict_policy in config/strategy

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

        def _try(message: str, bullets: list[str]) -> None:
            self.console.print(f"[red]{message}[/red]")
            if bullets:
                self.console.print("Try:")
                for b in bullets[:3]:
                    self.console.print(f"  • {b}")

        # 1. Check repository exists
        repo_path = Path(args.repo)
        if not repo_path.exists():
            _try(
                f"repository not found: {repo_path}",
                ["create repo: git init", f"verify path: {repo_path}"],
            )
            return False

        if not repo_path.is_dir():
            _try(
                f"path is not a directory: {repo_path}",
                ["choose a git repo directory", f"ls -la {repo_path}"],
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
                _try(
                    f"base branch not found: '{args.base_branch}'",
                    [
                        "fetch origin: git fetch origin --prune",
                        "list branches: git branch --all",
                    ],
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
            _try(
                "git error: failed to check repository",
                [str(e), "ensure git is installed"],
            )
            return False

        # 3b. Check working tree cleanliness (warn only)
        try:
            import subprocess

            dirty_cmd = ["git", "-C", str(repo_path), "status", "--porcelain"]
            result = subprocess.run(dirty_cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                self.console.print(
                    "[yellow]Warning: Working tree has uncommitted changes. Proceeding is safe (imports touch refs only).[/yellow]"
                )
        except (subprocess.SubprocessError, OSError) as e:
            self.console.print(f"[red]Error checking working tree: {e}[/red]")
            return False

        # 4. Disk space check removed: do not block runs based on free space.

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
        Load configuration from pitaya.yaml file.

        Precedence order:
        1. If config file is specified via args, use that
        2. If pitaya.yaml exists in current directory, use that
        3. Otherwise return None
        """

        config_path = None

        # Check if config file was specified
        if hasattr(args, "config") and args.config:
            config_path = Path(args.config)
        else:
            # Check for default pitaya.yaml
            default_path = Path("pitaya.yaml")
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

    def _validate_full_config(
        self, full_config: Dict[str, Any], args: argparse.Namespace
    ) -> bool:
        """Validate merged config. Print compact error table when invalid.

        Returns True when valid; False otherwise.
        """
        errors: list[tuple[str, str, str]] = []  # field, reason, example

        def _add(field: str, reason: str, example: str = ""):
            errors.append((field, reason, example))

        try:
            rt = full_config.get("runner", {}).get("timeout")
            if not isinstance(rt, (int, float)) or int(rt) <= 0:
                _add("runner.timeout", "must be a positive integer", "3600")
        except Exception:
            _add("runner.timeout", "invalid", "3600")
        try:
            cpu = int(full_config.get("runner", {}).get("cpu_limit", 2))
            if cpu <= 0:
                _add("runner.cpu_limit", "must be > 0", "2")
        except Exception:
            _add("runner.cpu_limit", "must be an integer", "2")
        try:
            mem = full_config.get("runner", {}).get("memory_limit", "4g")
            if isinstance(mem, str):
                val = int(mem[:-1]) if mem.lower().endswith("g") else int(mem)
            else:
                val = int(mem)
            if val <= 0:
                _add("runner.memory_limit", "must be > 0", "4g")
        except Exception:
            _add("runner.memory_limit", "must be number or '<n>g'", "4g")
        try:
            egress = str(
                full_config.get("runner", {}).get("network_egress", "online")
            ).lower()
            if egress not in {"online", "offline", "proxy"}:
                _add("runner.network_egress", "must be online|offline|proxy", "online")
        except Exception:
            pass
        try:
            ip = str(full_config.get("import_policy", "auto")).lower()
            if ip not in {"auto", "never", "always"}:
                _add("import_policy", "must be auto|never|always", "auto")
        except Exception:
            pass
        try:
            icp = str(full_config.get("import_conflict_policy", "fail")).lower()
            if icp not in {"fail", "overwrite", "suffix"}:
                _add("import_conflict_policy", "must be fail|overwrite|suffix", "fail")
        except Exception:
            pass
        # Allow 'auto' for both
        try:
            mpi = full_config.get("orchestration", {}).get(
                "max_parallel_instances", "auto"
            )
            if isinstance(mpi, str) and mpi.lower() == "auto":
                pass
            else:
                v = int(mpi)
                if v <= 0:
                    _add(
                        "orchestration.max_parallel_instances",
                        "must be > 0 or 'auto'",
                        "auto",
                    )
        except Exception:
            _add(
                "orchestration.max_parallel_instances",
                "must be integer or 'auto'",
                "auto",
            )
        try:
            mps = full_config.get("orchestration", {}).get(
                "max_parallel_startup", "auto"
            )
            if isinstance(mps, str) and mps.lower() == "auto":
                pass
            else:
                v2 = int(mps)
                if v2 <= 0:
                    _add(
                        "orchestration.max_parallel_startup",
                        "must be > 0 or 'auto'",
                        "auto",
                    )
        except Exception:
            _add(
                "orchestration.max_parallel_startup",
                "must be integer or 'auto'",
                "auto",
            )
        # Strategy exists (built-in name or file.py[:Class] or module.path[:Class])
        try:
            strategy = full_config.get("strategy", args.strategy)
            ok = False
            if isinstance(strategy, str):
                if strategy in AVAILABLE_STRATEGIES:
                    ok = True
                else:
                    spec = str(strategy)
                    if ":" in spec:
                        spec_path = spec.split(":", 1)[0]
                    else:
                        spec_path = spec
                    from pathlib import Path as _P

                    if spec_path.endswith(".py") and _P(spec_path).exists():
                        ok = True
                    else:
                        # Accept dotted module paths without importing (avoid executing user code during validation)
                        try:
                            import re as _re

                            mod_re = _re.compile(
                                r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
                            )
                            if mod_re.match(spec_path):
                                ok = True
                        except Exception:
                            pass
            if not ok:
                _add(
                    "strategy",
                    "unknown strategy (use built-in, file.py[:Class], or module.path[:Class])",
                    ",".join(AVAILABLE_STRATEGIES.keys()),
                )
        except Exception:
            pass
        if errors:
            self.console.print("[red]Invalid configuration:[/red]")
            self.console.print("field | reason | example")
            for f, r, ex in errors:
                self.console.print(f"{f} | {r} | {ex}")
            return False
        return True

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
                self.console.print(f"  Resume: pitaya --resume {run_id}")
            self.console.print(f"  View logs: {args.logs_dir}/{run_id}/")

            return 0

        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error showing run: {e}[/red]")
            self.console.print_exception()
            return 1

    # Explicit prune command removed — cleanup is automatic

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

        # Summary section (prefer strategy-level status; fall back to instances)
        self.console.print("[bold]Summary:[/bold]")

        # Strategy-level rollup when state is available
        if state and hasattr(state, "strategies") and state.strategies:
            try:
                strat_states = [
                    getattr(s, "state", "") for s in state.strategies.values()
                ]
                strat_success = sum(1 for s in strat_states if s == "completed")
                strat_canceled = sum(1 for s in strat_states if s == "canceled")
                strat_failed = sum(1 for s in strat_states if s == "failed")
                self.console.print(
                    f"  Strategies: {strat_success}/{len(strat_states)} completed; {strat_canceled} canceled; {strat_failed} failed"
                )
                # Show resume tip when any strategy canceled
                if strat_canceled > 0:
                    self.console.print(
                        "\n[blue]Run interrupted. To resume this run:[/blue]"
                    )
                    self.console.print(f"  pitaya --resume {run_id}")
            except Exception:
                pass

        # Instance-level totals for context
        total_duration = sum(r.duration_seconds or 0 for r in results)
        total_cost = sum(r.metrics.get("total_cost", 0) for r in results if r.metrics)
        success_count = sum(1 for r in results if r.success)
        int_count = sum(1 for r in results if getattr(r, "status", "") == "canceled")
        failed_count = sum(
            1
            for r in results
            if (not r.success and getattr(r, "status", "") != "canceled")
        )
        total_count = len(results)

        if total_duration >= 60:
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{int(total_duration)}s"

        self.console.print(f"  Total Duration: {duration_str}")
        self.console.print(f"  Total Cost: ${total_cost:.2f}")
        self.console.print(
            f"  Instances: {success_count} succeeded, {int_count} canceled, {failed_count} failed (total {total_count})"
        )

        # Final branches (show generated branches even if scoring canceled)
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

            self.console.print("\n[dim]To merge a solution:[/dim]")
            self.console.print(f"  git merge {final_branches[0]}")

    async def run_orchestrator(self, args: argparse.Namespace) -> int:
        """
        Run Pitaya with or without TUI.

        Args:
            args: Parsed command line arguments

        Returns:
            Exit code
        """
        # Determine run_id for logging (spec: run_YYYYMMDD_HHMMSS_<short8>)
        if args.resume:
            run_id = args.resume
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            import uuid as _uuid

            short8 = _uuid.uuid4().hex[:8]
            run_id = f"run_{timestamp}_{short8}"

        # Setup structured logging
        from .utils.structured_logging import setup_structured_logging

        setup_structured_logging(
            logs_dir=args.logs_dir,
            run_id=run_id,
            debug=True,
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
                cfg = locals().get("full_config") or {}
                max_bytes = (cfg.get("logging", {}) or {}).get(
                    "max_file_size", 10485760
                )
                max_mb = (
                    int(max_bytes / (1024 * 1024))
                    if isinstance(max_bytes, (int, float))
                    else 100
                )
            except Exception:
                max_mb = 100
            asyncio.create_task(
                setup_log_rotation_task(args.logs_dir, max_size_mb=max_mb)
            )
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to setup log rotation: {e}")

        # Check platform and show recommendations if needed
        from .utils.platform_utils import (
            validate_docker_setup,
            get_platform_recommendations,
        )

        # Apply --json convenience alias
        if getattr(args, "json", False):
            args.no_tui = True
            args.output = "json"
        # Default to headless in non-TTY/CI unless explicitly overridden
        try:
            if not sys.stdout.isatty() and not getattr(args, "json", False):
                args.no_tui = True
        except Exception:
            pass

        # Validate Docker setup
        docker_valid, docker_error = validate_docker_setup()
        if not docker_valid:
            # Microcopy with Try bullets
            self.console.print("[red]cannot connect to docker daemon[/red]")
            self.console.print("Try:")
            self.console.print("  • start Docker Desktop / system service")
            self.console.print("  • check $DOCKER_HOST")
            self.console.print("  • run: docker info")
            return 1

        # Fast path: diagnostics smoke test
        # Removed: docker smoke test path

        # Show platform recommendations if any
        recommendations = get_platform_recommendations()
        if recommendations and not args.no_tui:
            for rec in recommendations:
                self.console.print(f"[yellow]Platform Note:[/yellow] {rec}")

        # Cleanup modes removed

        # Handle list-runs mode
        if args.list_runs:
            return await self.run_list_runs(args)

        # No explicit prune mode; handled automatically

        # Handle show-run mode
        if args.show_run:
            return await self.run_show_run(args)

        # Normalize missing prompt to empty string and let strategies validate if needed
        if not args.resume:
            args.prompt = getattr(args, "prompt", "") or ""

        # Perform pre-flight checks for new runs
        if not args.resume:
            if getattr(args, "require_clean_wt", False):
                # enforce clean working tree before preflight
                try:
                    import subprocess as _sp

                    status = _sp.run(
                        ["git", "-C", str(args.repo), "status", "--porcelain"],
                        capture_output=True,
                        text=True,
                    )
                    if status.returncode == 0 and status.stdout.strip():
                        self.console.print(
                            "[red]Working tree has uncommitted changes. Use --require-clean-wt=false to bypass.[/red]"
                        )
                        return 1
                except Exception as e:
                    self.console.print(f"[red]Failed checking working tree: {e}[/red]")
                    return 1
            if not await self._perform_preflight_checks(args):
                return 1

        # Load configuration from file if specified
        config = self._load_config_file(args)

        # Load all configuration sources
        env_config = load_env_config()
        dotenv_config = load_dotenv_config()  # Load .env file separately
        if dotenv_config:
            # Headless: show a small hint; TUI: log only
            if getattr(args, "no_tui", False):
                try:
                    self.console.print(
                        "[yellow]Loaded secrets from .env (development convenience). Consider using env vars in CI.[/yellow]"
                    )
                except Exception:
                    pass
            try:
                logger.info("dotenv: loaded .env values (dev convenience)")
            except Exception:
                pass
        defaults = get_default_config()

        # Build CLI config dict from args
        cli_config = {}
        if args.max_parallel:
            cli_config.setdefault("orchestration", {})[
                "max_parallel_instances"
            ] = args.max_parallel
        if getattr(args, "max_startup_parallel", None):
            cli_config.setdefault("orchestration", {})[
                "max_parallel_startup"
            ] = args.max_startup_parallel
        if getattr(args, "randomize_queue", False):
            cli_config.setdefault("orchestration", {})["randomize_queue_order"] = True
        if hasattr(args, "timeout") and args.timeout:
            cli_config.setdefault("runner", {})["timeout"] = args.timeout
        if hasattr(args, "force_commit") and args.force_commit:
            cli_config.setdefault("runner", {})["force_commit"] = True
        if args.model:
            cli_config["model"] = args.model
        if hasattr(args, "plugin") and args.plugin:
            cli_config["plugin_name"] = args.plugin
        if hasattr(args, "docker_image") and args.docker_image:
            cli_config.setdefault("runner", {})["docker_image"] = args.docker_image
        if args.strategy:
            cli_config["strategy"] = args.strategy
        if args.state_dir:
            cli_config.setdefault("orchestration", {})["state_dir"] = args.state_dir
        if args.logs_dir:
            cli_config.setdefault("orchestration", {})["logs_dir"] = args.logs_dir
        if args.output:
            cli_config["output"] = args.output
        # Server extension support removed
        # Always use full logging; no debug toggle
        # Add CLI auth args
        if hasattr(args, "oauth_token") and args.oauth_token:
            cli_config.setdefault("runner", {})["oauth_token"] = args.oauth_token
        if hasattr(args, "api_key") and args.api_key:
            cli_config.setdefault("runner", {})["api_key"] = args.api_key
        if hasattr(args, "base_url") and args.base_url:
            cli_config.setdefault("runner", {})["base_url"] = args.base_url

        # Apply proxy flags to environment early so downstream components observe them
        # Proxy environment mapping removed

        # Merge configurations with proper precedence: CLI > env > .env > project file > global user config > defaults
        global_config = load_global_config()
        full_config = merge_config(
            cli_config,
            env_config,
            dotenv_config,
            config or {},
            merge_config({}, {}, {}, global_config or {}, defaults),
        )
        # If both global locations exist, note which one is used
        try:
            home = Path.home()
            preferred = home / ".pitaya" / "config.yaml"
            fallback = home / ".config" / "pitaya" / "config.yaml"
            if preferred.exists() and fallback.exists():
                if getattr(args, "no_tui", False):
                    self.console.print(
                        "[dim]Using ~/.pitaya/config.yaml (XDG fallback ignored).[/dim]"
                    )
                try:
                    logger.info(
                        "config: preferred ~/.pitaya/config.yaml; XDG fallback ignored"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Removed: --safe-fsync propagation (use default interval batching)

        # Print Effective Config header (redacted, truncated)
        try:

            def _flatten(d: dict, prefix=""):
                out = {}
                for k, v in (d or {}).items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    if isinstance(v, dict):
                        out.update(_flatten(v, key))
                    else:
                        out[key] = v
                return out

            # sources
            srcs = {
                "cli": _flatten(cli_config),
                "env": _flatten(env_config),
                "dotenv": _flatten(dotenv_config),
                "project": _flatten(config or {}),
                "global": _flatten(global_config or {}),
                "defaults": _flatten(defaults),
            }
            eff = _flatten(full_config)

            # redaction
            def _red(k, v):
                kl = k.lower()
                if any(
                    s in kl
                    for s in (
                        "token",
                        "key",
                        "secret",
                        "password",
                        "authorization",
                        "cookie",
                    )
                ):
                    return "[REDACTED]"
                return v

            lines = []
            count = 0
            for k in sorted(eff.keys()):
                v = _red(k, eff[k])
                # find winning source
                src = next(
                    (
                        name
                        for name in (
                            "cli",
                            "env",
                            "dotenv",
                            "project",
                            "global",
                            "defaults",
                        )
                        if k in srcs[name]
                    ),
                    "defaults",
                )
                # unify naming for display
                disp = src
                lines.append(f"  {k} = {v}  ({disp})")
                count += 1
                if count >= 20:
                    lines.append("  … see `pitaya config print` for full config")
                    break
            # Insert a concise secrets redaction note for visibility
            lines.insert(0, "  secrets=REDACTED")
            hdr = "Effective Config:\n" + "\n".join(lines)
            if args.no_tui:
                # Console: plain text, no color for CI friendliness
                print(hdr)
            else:
                # TUI: do not print to console; log instead
                try:
                    logger.info(hdr)
                except Exception:
                    pass
        except Exception:
            pass

        # models.yaml alias mapping removed; no drift checks

        # Validate merged configuration; render compact error table on invalid
        if not self._validate_full_config(full_config, args):
            return 1

        # Configure container limits from merged config; apply --parallel preset if provided
        # Store allow flags for constructor
        allow_overwrite = bool(getattr(args, "allow_overwrite_protected_refs", False))

        # Parallel presets removed. Concurrency is either auto (default) or explicitly set via --max-parallel.

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

        # Create auth config from merged configuration (plugin-aware)
        plugin_name = full_config.get("plugin_name") or getattr(
            args, "plugin", "claude-code"
        )
        rcfg = full_config.get("runner", {})
        if str(plugin_name) == "codex":
            api_key = rcfg.get("api_key") or rcfg.get("openai_api_key")
            base_url = rcfg.get("base_url") or rcfg.get("openai_base_url")
            oauth_token = None
            if not api_key:
                self.console.print(
                    "[red]Error: Missing credentials for OpenAI‑compatible provider.[/red]\n"
                    "Set OPENAI_API_KEY in:\n"
                    "  - .env file (OPENAI_API_KEY=...)\n"
                    "  - Environment variables (export OPENAI_API_KEY=...)\n"
                    "  - Command line: --api-key\n"
                    "Optionally set --base-url (or OPENAI_BASE_URL) for non-default endpoints."
                )
                return 1
        else:
            oauth_token = rcfg.get("oauth_token")
            api_key = rcfg.get("api_key") or rcfg.get("anthropic_api_key")
            base_url = rcfg.get("base_url") or rcfg.get("anthropic_base_url")
            if not oauth_token and not api_key:
                self.console.print(
                    "[red]Error: Missing credentials for Anthropic.[/red]\n"
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
        max_startup_parallel = full_config["orchestration"].get(
            "max_parallel_startup", "auto"
        )
        state_dir = full_config.get("orchestration", {}).get(
            "state_dir"
        ) or full_config.get("state_dir", Path("./pitaya_state"))
        logs_dir = full_config.get("orchestration", {}).get(
            "logs_dir"
        ) or full_config.get("logs_dir", Path("./logs"))

        # Configuration sources: print only in headless; log in all modes
        src_parts = []
        if cli_config:
            src_parts.append("cli")
        if dotenv_config:
            src_parts.append("dotenv")
        if config:
            src_parts.append(f"file={args.config or 'pitaya.yaml'}")
        src_parts.append("defaults")
        try:
            logger.info("config sources: %s", ", ".join(src_parts))
        except Exception:
            pass
        if getattr(args, "no_tui", False):
            self.console.print("[dim]Configuration loaded from:[/dim]")
            if cli_config:
                self.console.print("  - Command line arguments")
            if dotenv_config:
                self.console.print("  - .env file")
            if config:
                self.console.print(f"  - Config file: {args.config or 'pitaya.yaml'}")
            self.console.print("  - Built-in defaults")

        # Global session volume support removed

        # Resolve parallelism
        import os as _os

        def _cpu_default() -> int:
            try:
                return max(2, int(_os.cpu_count() or 2))
            except Exception:
                return 2

        # Total parallel: auto -> cpu-based
        if isinstance(max_parallel, str):
            if max_parallel.lower() == "auto":
                max_parallel_val = _cpu_default()
            else:
                max_parallel_val = int(max_parallel)
        elif isinstance(max_parallel, int):
            max_parallel_val = max(1, max_parallel)
        else:
            max_parallel_val = _cpu_default()

        # Startup parallel: auto -> min(10, total)
        if isinstance(max_startup_parallel, str):
            if max_startup_parallel.lower() == "auto":
                max_startup_parallel_val = min(10, max_parallel_val)
            else:
                max_startup_parallel_val = int(max_startup_parallel)
        elif isinstance(max_startup_parallel, int):
            max_startup_parallel_val = max(1, max_startup_parallel)
        else:
            max_startup_parallel_val = min(10, max_parallel_val)
        # Clamp startup to not exceed total
        max_startup_parallel_val = min(max_startup_parallel_val, max_parallel_val)

        # Proxy automatic egress defaults removed

        # Combine agent CLI passthrough args and create orchestrator
        try:
            import shlex as _shlex

            _agent_cli_args: list[str] = []
            if getattr(args, "agent_cli_arg", None):
                _agent_cli_args.extend(
                    [str(a) for a in args.agent_cli_arg if a is not None]
                )
            if getattr(args, "agent_cli_args_str", None):
                _agent_cli_args.extend(_shlex.split(str(args.agent_cli_args_str)))
        except Exception:
            _agent_cli_args = []

        # Create orchestrator
        self.orchestrator = Orchestrator(
            max_parallel_instances=max_parallel_val,
            max_parallel_startup=max_startup_parallel_val,
            state_dir=Path(state_dir),
            logs_dir=Path(logs_dir),
            container_limits=container_limits,
            retry_config=retry_config,
            auth_config=auth_config,
            snapshot_interval=int(
                full_config.get("orchestration", {}).get("snapshot_interval", 30)
            ),
            event_buffer_size=int(
                full_config.get("orchestration", {}).get("event_buffer_size", 10000)
            ),
            runner_timeout_seconds=int(
                full_config.get("runner", {}).get("timeout", 3600)
            ),
            default_network_egress=str(
                full_config.get("runner", {}).get("network_egress", "online")
            ),
            branch_namespace=str(
                full_config.get("orchestration", {}).get(
                    "branch_namespace", "hierarchical"
                )
            ),
            allow_overwrite_protected_refs=allow_overwrite,
            default_plugin_name=str(
                full_config.get("plugin_name", getattr(args, "plugin", "claude-code"))
            ),
            default_model_alias=str(
                full_config.get("model", getattr(args, "model", "sonnet"))
            ),
            default_docker_image=full_config.get("runner", {}).get("docker_image"),
            default_agent_cli_args=_agent_cli_args or None,
            force_commit=bool(full_config.get("runner", {}).get("force_commit", False)),
            randomize_queue_order=bool(
                full_config.get("orchestration", {}).get("randomize_queue_order", False)
            ),
        )

        # Initialize orchestrator
        await self.orchestrator.initialize()
        # Apply custom redaction patterns to event bus, if any
        try:
            red = (
                full_config.get("logging", {})
                .get("redaction", {})
                .get("custom_patterns", [])
            )
            if getattr(self.orchestrator, "event_bus", None):
                self.orchestrator.event_bus.set_custom_redaction_patterns(
                    red if isinstance(red, list) else []
                )
        except Exception:
            pass

        # Server extension support removed

        # Determine run mode
        if args.no_tui:
            # Headless mode
            return await self._run_headless(args, run_id, full_config)
        else:
            # TUI mode
            return await self._run_with_tui(args, run_id, full_config)

    # Docker smoke test removed (minimal CLI)

    async def _run_headless(
        self, args: argparse.Namespace, run_id: str, full_config: Dict[str, Any]
    ) -> int:
        """Run Pitaya in headless mode."""
        output_mode = args.output or "streaming"

        # Set up event subscriptions for output
        if output_mode == "streaming":
            # Canonical public events only; compact hybrid lines
            import hashlib as _hashlib
            from typing import Set as _Set

            def _short8(s: str) -> str:
                return _hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[
                    :8
                ]

            no_emoji = bool(getattr(args, "no_emoji", False))
            show_full = getattr(args, "show_ids", "short") == "full"
            verbose = bool(getattr(args, "verbose", False))
            # once-per-task offline hint
            _hint_emitted: _Set[str] = set()

            def _prefix(ev: dict) -> str:
                sid = ev.get("strategy_execution_id") or ev.get("data", {}).get(
                    "strategy_execution_id"
                )
                key = ev.get("key") or ev.get("data", {}).get("key")
                iid = ev.get("instance_id") or ev.get("data", {}).get("instance_id", "")
                k8 = "????????"
                if sid and key:
                    k8 = _short8(f"{sid}|{key}")
                inst = iid if show_full else (iid[:5] if iid else "?????")
                return f"[k{k8}][inst-{inst}]"

            def _glyph(name: str) -> str:
                if no_emoji:
                    return ""
                return {
                    "started": " ▶",
                    "completed": " ✅",
                    "failed": " ❌",
                    "interrupted": " ⏸",
                }.get(name, "")

            def _fmt_time(seconds: float) -> str:
                try:
                    s = float(seconds)
                except Exception:
                    return "-"
                if s < 60:
                    return f"{s:.0f}s"
                m, sec = divmod(int(s), 60)
                h, m = divmod(m, 60)
                return f"{h}h{m}m{sec}s" if h else f"{m}m{sec}s"

            def print_event(ev: dict):
                et = ev.get("type", "")
                data = ev.get("data", {})
                pre = _prefix(ev)
                # Lifecycle only by default
                if et == "task.scheduled":
                    # keep minimal – often too chatty in multi-run
                    return
                if et == "task.started":
                    model = data.get("model")
                    base = data.get("base_branch")
                    line = f"{pre}{_glyph('started')} started"
                    if model:
                        line += f" model={model}"
                    if base:
                        line += f" base={base}"
                    print(line)
                elif et == "task.completed":
                    art = data.get("artifact", {}) or {}
                    dur = data.get("metrics", {}).get("duration_seconds") or data.get(
                        "duration_seconds"
                    )
                    cost = data.get("metrics", {}).get("total_cost")
                    toks = data.get("metrics", {}).get("total_tokens")
                    branch = art.get("branch_final") or art.get("branch_planned")
                    parts = [f"{pre}{_glyph('completed')} completed"]
                    if dur is not None:
                        parts.append(f"time={_fmt_time(dur)}")
                    if cost is not None:
                        try:
                            parts.append(f"cost=${float(cost):.2f}")
                        except Exception:
                            pass
                    if toks is not None:
                        parts.append(f"tok={toks}")
                    if branch:
                        parts.append(f"branch={branch}")
                    # Truncation path hint
                    if data.get("final_message_truncated") and data.get(
                        "final_message_path"
                    ):
                        parts.append(f"final_message={data.get('final_message_path')}")
                    print(" ".join(parts))
                elif et == "task.failed":
                    etype = data.get("error_type") or "unknown"
                    msg = (data.get("message") or "").strip().replace("\n", " ")
                    line = f"{pre}{_glyph('failed')} failed type={etype}"
                    if msg:
                        line += f' message="{msg[:200]}"'
                    # offline egress hint (once per task)
                    if str(data.get("network_egress")) == "offline":
                        iid = data.get("instance_id") or ""
                        if iid and iid not in _hint_emitted:
                            _hint_emitted.add(iid)
                            line += " hint=egress=offline:set runner.network_egress=online/proxy"
                    print(line)
                elif et == "task.interrupted":
                    print(f"{pre}{_glyph('interrupted')} interrupted")
                elif verbose and et == "task.progress":
                    phase = data.get("phase") or data.get("activity")
                    if phase in ("container_created", "branch_imported", "no_changes"):
                        print(f"{pre} phase={phase}")

            for t in (
                "task.started",
                "task.completed",
                "task.failed",
                "task.interrupted",
                "task.progress",
            ):
                self.orchestrator.subscribe(t, print_event)
        elif output_mode == "json":
            # Stream canonical public events as NDJSON (subscribe BEFORE run/resume)
            def emit_json(ev: dict):
                print(json.dumps(ev, separators=(",", ":")))

            for t in (
                "task.scheduled",
                "task.started",
                "task.progress",
                "task.completed",
                "task.failed",
                "task.interrupted",
                "strategy.started",
                "strategy.completed",
            ):
                self.orchestrator.subscribe(t, emit_json)

        try:
            if args.resume:
                # Resume existing run
                run_id = args.resume
                self.console.print(f"[blue]Resuming run {run_id}...[/blue]")

                # Resume the run
                result = await self.orchestrator.resume_run(run_id)
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
                    run_id=run_id,
                )

            # Output results based on mode
            if output_mode == "quiet":
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

            # CI artifacts bundle removed

            # Decide exit code based on failures (spec: 3 = completed with failures)
            exit_code = 0
            try:
                st = None
                if self.orchestrator and getattr(
                    self.orchestrator, "state_manager", None
                ):
                    st = self.orchestrator.state_manager.get_current_state()
                if st:
                    try:
                        from .shared import InstanceStatus as _IS

                        insts = list(st.instances.values())
                        failed_count = sum(1 for i in insts if i.state == _IS.FAILED)
                        if failed_count > 0:
                            exit_code = 3
                    except Exception:
                        pass
            except Exception:
                pass

            # Shutdown orchestrator to stop background tasks
            await self.orchestrator.shutdown()

            return exit_code

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            # Show resume command if not in resume mode
            if not args.resume:
                # Use the run_id from orchestrator's current state
                if self.orchestrator and self.orchestrator.state_manager:
                    state = self.orchestrator.state_manager.current_state
                    if state and state.run_id:
                        self.console.print("\n[blue]To resume this run:[/blue]")
                        self.console.print(f"  pitaya --resume {state.run_id}")
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            return 2
        except (OrchestratorError, DockerError, ValidationError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
            self.console.print_exception()
            # Shutdown orchestrator
            await self.orchestrator.shutdown()
            return 1

    async def _run_with_tui(
        self, args: argparse.Namespace, run_id: str, full_config: Dict[str, Any]
    ) -> int:
        """Run Pitaya with TUI display."""
        # Create TUI display with configured refresh rate
        try:
            rr_ms = int(full_config.get("tui", {}).get("refresh_rate_ms", 100))
            rr_sec = max(0.01, rr_ms / 1000.0)
        except Exception:
            rr_sec = 0.1
        self.tui_display = TUIDisplay(
            console=self.console, refresh_rate=rr_sec, state_poll_interval=3.0
        )
        # Apply display density override
        if getattr(args, "display", None) and args.display != "auto":
            try:
                self.tui_display.set_forced_display_mode(args.display)
            except Exception:
                pass
        # Details pane mode removed from CLI; TUI defaults apply
        # IDs verbosity note in TUI header
        try:
            self.tui_display.set_ids_full(getattr(args, "show_ids", "short") == "full")
        except Exception:
            pass
        # Configure details message count
        try:
            details_n = int(full_config.get("tui", {}).get("details_messages", 10))
            if hasattr(self.tui_display, "set_details_messages"):
                self.tui_display.set_details_messages(max(1, details_n))
        except Exception:
            pass
        # Apply color scheme
        try:
            scheme = str(full_config.get("tui", {}).get("color_scheme", "default"))
            if hasattr(self.tui_display, "adaptive_display") and hasattr(
                self.tui_display.adaptive_display, "set_color_scheme"
            ):
                self.tui_display.adaptive_display.set_color_scheme(scheme)
        except Exception:
            pass

        # Start Pitaya and TUI together
        try:
            # Use the passed run_id (already determined in run() method)

            events_file = args.logs_dir / run_id / "events.jsonl"

            # Ensure the logs directory exists
            events_file.parent.mkdir(parents=True, exist_ok=True)

            # Create tasks for both Pitaya and TUI
            orchestrator_task = None
            tui_task = None

            if args.resume:
                # Resume existing run
                orchestrator_task = asyncio.create_task(
                    self.orchestrator.resume_run(run_id)
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
                # Wait for events file to be created by Pitaya
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

            # Detect task errors early so users see failures in TUI mode
            tui_error = None
            orch_error = None
            try:
                for task in done:
                    if task == tui_task:
                        try:
                            _ = task.result()
                        except Exception as e:  # capture but continue graceful shutdown
                            tui_error = e
                    elif task == orchestrator_task:
                        # Retrieve result below; capture exception here if present
                        try:
                            _ = task.result()
                        except Exception as e:
                            orch_error = e
            except Exception:
                pass

            # Check if shutdown was requested
            shutdown_requested = False
            for task in done:
                if task == shutdown_task and task.result() == "shutdown":
                    shutdown_requested = True
                    break

            # Get the result from Pitaya when available (ignore if it errored)
            result = None
            if (
                not shutdown_requested
                and orchestrator_task in done
                and orch_error is None
            ):
                try:
                    result = orchestrator_task.result()
                except Exception:
                    result = None

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

            # If shutdown was requested, always show resume command (even during a resume)
            if shutdown_requested:
                self.console.print("\n[blue]To resume this run:[/blue]")
                self.console.print(f"  pitaya --resume {run_id}")
                # Standardized interrupted exit code per spec
                return 2

            # Surface task errors (TUI or orchestrator) to the user
            if tui_error or orch_error:
                if tui_error:
                    self.console.print(
                        f"[red]TUI crashed:[/red] {getattr(tui_error, 'args', tui_error)}"
                    )
                if orch_error:
                    self.console.print(
                        f"[red]Run failed:[/red] {getattr(orch_error, 'args', orch_error)}"
                    )
                try:
                    self.console.print(
                        f"[dim]See logs for details: {args.logs_dir}/{run_id}/[/dim]"
                    )
                except Exception:
                    pass
                return 1

            # Get actual run statistics from orchestrator state
            state = self.orchestrator.get_current_state()

            # Print final result
            if state:
                # Derive counts directly from instance states to avoid double-counting across resumes
                try:
                    from .shared import InstanceStatus as _IS
                except Exception:
                    _IS = None
                try:
                    instance_list = list(state.instances.values())
                    total_instances = len(instance_list)
                    if _IS:
                        completed_instances = sum(
                            1 for i in instance_list if i.state == _IS.COMPLETED
                        )
                        failed_instances = sum(
                            1 for i in instance_list if i.state == _IS.FAILED
                        )
                    else:
                        # Fallback to aggregate counters if type import fails
                        completed_instances = state.completed_instances
                        failed_instances = state.failed_instances
                    total_cost = state.total_cost
                except Exception:
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
                        name = r.branch_name or "no-branch"
                        self.console.print(f"  [{i+1}] Branch: {name}")
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
                                self.console.print(f"      {key}: {formatted_value}")
                        # Display final message if present
                        if r.final_message:
                            # Truncate very long messages
                            message = r.final_message
                            if len(message) > 500:
                                message = message[:497] + "..."
                            self.console.print(
                                f"      [dim]final_message:[/dim] {message}"
                            )

                        # Display metadata if present
                        if r.metadata:
                            # Show specific metadata fields if they exist
                            metadata_items = []
                            if "score" in r.metadata:
                                metadata_items.append(f"score={r.metadata['score']}")
                            if "complexity" in r.metadata:
                                metadata_items.append(
                                    f"complexity={r.metadata['complexity']}"
                                )
                            if "test_coverage" in r.metadata:
                                metadata_items.append(
                                    f"test_coverage={r.metadata['test_coverage']}%"
                                )
                            if "bug_confirmed" in r.metadata:
                                metadata_items.append(
                                    f"bug_confirmed={r.metadata['bug_confirmed']}"
                                )
                            if "bug_report_branch" in r.metadata:
                                metadata_items.append(
                                    f"bug_report_branch={r.metadata['bug_report_branch']}"
                                )

                            # Show any other metadata keys not already displayed
                            displayed_keys = {
                                "score",
                                "complexity",
                                "test_coverage",
                                "bug_confirmed",
                                "bug_report_branch",
                                "strategy_execution_id",
                                "model",
                            }
                            for key, value in r.metadata.items():
                                if key not in displayed_keys:
                                    metadata_items.append(f"{key}={value}")

                            if metadata_items:
                                self.console.print(
                                    f"      [dim]metadata:[/dim] {', '.join(metadata_items)}"
                                )
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

            # Decide exit code based on failures (spec: 3 = completed with failures)
            try:
                exit_code = 0
                if state:
                    from .shared import InstanceStatus as _IS

                    instance_list = list(state.instances.values())
                    failed_instances = sum(
                        1 for i in instance_list if i.state == _IS.FAILED
                    )
                    if failed_instances > 0:
                        exit_code = 3
                else:
                    # Fallback: infer from results list
                    if isinstance(result, list) and any(
                        (not getattr(r, "success", False)) for r in result
                    ):
                        exit_code = 3
            except Exception:
                exit_code = 0
            return exit_code

        except KeyboardInterrupt:
            # Ensure graceful shutdown on Ctrl+C in TUI path
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            if self.orchestrator:
                try:
                    await self.orchestrator.shutdown()
                except Exception:
                    pass
            if not args.resume:
                # Show resume command if we have a run id
                rid = run_id
                if rid:
                    self.console.print("\n[blue]To resume this run:[/blue]")
                    self.console.print(f"  pitaya --resume {rid}")
            return 2
        except (OrchestratorError, DockerError, ValidationError) as e:
            self.console.print(f"[red]Error: {e}[/red]")
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
        # Subcommands: doctor, config print
        if (args.prompt or "").strip().lower() == "doctor":
            return await self.run_doctor(args)
        if (args.prompt or "").strip().lower() == "config" and (
            args.subcommand or ""
        ).strip().lower() == "print":
            return await self.run_config_print(args)
        return await self.run_orchestrator(args)

    async def run_doctor(self, args: argparse.Namespace) -> int:
        """System checks for environment and config."""
        ok = True
        rows = []  # (status, title, message, tries)

        def pass_line(title: str, msg: str = ""):
            rows.append(("✓", title, msg, []))

        def fail_line(title: str, msg: str, tries: list[str]):
            nonlocal ok
            ok = False
            rows.append(("✗", title, msg, tries))

        def info_line(title: str, msg: str):
            rows.append(("i", title, msg, []))

        # Docker
        try:
            from .utils.platform_utils import validate_docker_setup

            valid, err = validate_docker_setup()
            if valid:
                pass_line("docker", "ok")
            else:
                fail_line(
                    "docker",
                    "cannot connect to docker daemon",
                    ["start Docker", "check $DOCKER_HOST", "run: docker info"],
                )
        except Exception as e:
            fail_line("docker", str(e), ["ensure Docker installed", "run: docker info"])
        # Disk
        try:
            import shutil as _sh

            stat = _sh.disk_usage(str(Path.cwd()))
            free_gb = stat.free / (1024**3)
            if free_gb >= 20:
                pass_line("disk", f"{free_gb:.1f}GB free")
            else:
                fail_line(
                    "disk",
                    f"insufficient disk space: {free_gb:.1f}GB free (<20GB)",
                    ["free space on this volume", "move repo to larger disk"],
                )
        except Exception as e:
            info_line("disk", f"could not check: {e}")
        # Repo and base branch
        try:
            repo = args.repo or Path.cwd()
            if not (Path(repo) / ".git").exists():
                fail_line(
                    "repo", f"not a git repo: {repo}", ["git init", "verify path"]
                )
            else:
                import subprocess as _sp

                base = args.base_branch or "main"
                rc = _sp.run(
                    ["git", "-C", str(repo), "rev-parse", "--verify", base],
                    capture_output=True,
                )
                if rc.returncode == 0:
                    pass_line("base_branch", base)
                else:
                    fail_line(
                        "base_branch",
                        f"not found: '{base}'",
                        [
                            "git fetch origin --prune",
                            "verify branch name",
                            "git branch --all",
                        ],
                    )
        except Exception as e:
            info_line("repo", f"check failed: {e}")
        # Temp dir writable
        try:
            from .utils.platform_utils import get_temp_dir

            td = get_temp_dir()
            td.mkdir(parents=True, exist_ok=True)
            test = td / "_pitaya_doctor.tmp"
            test.write_text("ok")
            test.unlink(missing_ok=True)
            pass_line("temp", str(td))
        except Exception as e:
            fail_line(
                "temp", f"not writable: {e}", ["adjust permissions", "set TMPDIR"]
            )
        # Auth
        try:
            env = load_env_config()
            dotenv = load_dotenv_config()
            if (
                env.get("runner", {}).get("oauth_token")
                or dotenv.get("runner", {}).get("oauth_token")
                or env.get("runner", {}).get("api_key")
                or dotenv.get("runner", {}).get("api_key")
            ):
                pass_line("auth", "credentials found")
            else:
                fail_line(
                    "auth",
                    "no credentials",
                    ["set CLAUDE_CODE_OAUTH_TOKEN", "or set ANTHROPIC_API_KEY"],
                )
        except Exception:
            pass
        # models.yaml checks removed
        # SELinux / WSL2 hints
        try:
            import platform

            if platform.system() == "Linux":
                info_line("selinux", "if enabled, :z labels will be applied")
        except Exception:
            pass
        try:
            from .utils.platform_utils import is_wsl
            import platform

            if platform.system() == "Windows" or is_wsl():
                info_line("wsl2", "place repo in WSL filesystem for performance")
        except Exception:
            pass
        # Print table-like output
        for status, title, message, tries in rows:
            if status in ("✓", "✗"):
                print(f"{status} {title}: {message}")
            else:
                print(f"i {title}: {message}")
            if status == "✗" and tries:
                print("Try:")
                for t in tries[:3]:
                    print(f"  • {t}")
        return 0 if ok else 1

    async def run_config_print(self, args: argparse.Namespace) -> int:
        """Print effective config with source and redaction."""
        env = load_env_config()
        dotenv = load_dotenv_config()
        defaults = get_default_config()
        global_cfg = load_global_config()
        project_cfg = self._load_config_file(args) or {}
        merged = merge_config(
            {},
            env,
            dotenv,
            project_cfg,
            merge_config({}, {}, {}, global_cfg or {}, defaults),
        )
        allow_unred = (
            os.environ.get("PITAYA_ALLOW_UNREDACTED") == "1"
            and str(getattr(args, "redact", "true")).lower() == "false"
        )

        def _red(k, v):
            if allow_unred:
                return v
            kl = k.lower()
            if any(
                s in kl
                for s in (
                    "token",
                    "key",
                    "secret",
                    "password",
                    "authorization",
                    "cookie",
                )
            ):
                return "[REDACTED]"
            return v

        def _flat(d, p=""):
            out = {}
            for k, v in (d or {}).items():
                kk = f"{p}.{k}" if p else str(k)
                if isinstance(v, dict):
                    out.update(_flat(v, kk))
                else:
                    out[kk] = v
            return out

        # Determine source per key
        sources = {
            "env": _flat(env),
            "dotenv": _flat(dotenv),
            "project": _flat(project_cfg),
            "global": _flat(global_cfg or {}),
            "defaults": _flat(defaults),
        }
        flat = _flat(merged)

        def _src_for(k: str) -> str:
            for name in ("env", "dotenv", "project", "global", "defaults"):
                if k in sources[name]:
                    return name
            return "defaults"

        if getattr(args, "json", False):
            out = {
                k: {"value": _red(k, v), "source": _src_for(k)} for k, v in flat.items()
            }
            print(json.dumps(out, indent=2, default=str))
            return 0
        for k in sorted(flat.keys()):
            print(f"{k}: {_red(k, flat[k])}  ({_src_for(k)})")
        return 0


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
            os._exit(2)

    signal.signal(signal.SIGINT, signal_handler)

    # Pass the shutdown event to CLI
    cli.shutdown_event = shutdown_event

    # Run async
    try:
        exit_code = asyncio.run(cli.run(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Standardized interrupted exit code per spec
        sys.exit(2)


if __name__ == "__main__":
    main()
