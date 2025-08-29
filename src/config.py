"""
Configuration management for Pitaya.

Configuration precedence: CLI > .env (auth only) > pitaya.yaml > defaults.
No Pitaya-specific environment variables are used.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def merge_config(
    cli_args: Dict[str, Any],
    env_config: Dict[str, Any],
    dotenv_config: Dict[str, Any],
    file_config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge configuration from multiple sources with proper precedence.

    Precedence order (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. .env file
    4. pitaya.yaml config file
    5. Defaults
    """
    # Start with defaults
    result = defaults.copy()

    # Overlay file config (pitaya.yaml)
    deep_merge(result, file_config)

    # Overlay .env config
    deep_merge(result, dotenv_config)

    # Overlay environment config
    deep_merge(result, env_config)

    # Overlay CLI args (only non-None values)
    cli_filtered = {k: v for k, v in cli_args.items() if v is not None}
    deep_merge(result, cli_filtered)

    return result


def _apply_yaml_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize YAML keys to spec-compatible names.

    Aliases:
    - container_cpu -> runner.cpu_limit
    - container_memory -> runner.memory_limit
    Both at top-level or under a known section are handled.
    """

    def set_runner_key(key: str, value: Any):
        config.setdefault("runner", {})[key] = value

    # Top-level aliases
    if "container_cpu" in config:
        set_runner_key("cpu_limit", config.pop("container_cpu"))
    if "container_memory" in config:
        set_runner_key("memory_limit", config.pop("container_memory"))

    # Nested aliases under 'runner'
    runner = config.get("runner", {})
    if "container_cpu" in runner:
        set_runner_key("cpu_limit", runner.pop("container_cpu"))
    if "container_memory" in runner:
        set_runner_key("memory_limit", runner.pop("container_memory"))

    return config


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Recursively merge overlay dict into base dict."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def load_yaml_config(yaml_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from pitaya.yaml file.

    Args:
        yaml_path: Path to YAML config file (default: pitaya.yaml in current directory)

    Returns:
        Configuration dictionary from YAML file
    """
    if yaml_path is None:
        yaml_path = Path("pitaya.yaml")

    if not yaml_path.exists():
        return {}

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Ensure we return a dict even if YAML is empty
        if not isinstance(config, dict):
            return {}

        return config
    except (yaml.YAMLError, OSError) as e:
        # Log error but don't fail - just return empty config
        import logging

        logging.getLogger(__name__).warning(f"Failed to load {yaml_path}: {e}")
        return {}


def load_global_config() -> Dict[str, Any]:
    """Load global user config.

    Preferred path: ~/.pitaya/config.yaml
    Fallback path: ~/.config/pitaya/config.yaml
    If both exist, the preferred path wins.
    """
    try:
        home = Path.home()
    except Exception:
        return {}
    preferred = home / ".pitaya" / "config.yaml"
    fallback = home / ".config" / "pitaya" / "config.yaml"
    for p in (preferred, fallback):
        try:
            if p.exists():
                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def load_dotenv_config(dotenv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from .env file.

    Supports standard environment variables:
    - CLAUDE_CODE_OAUTH_TOKEN
    - ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
    - OPENAI_API_KEY / OPENAI_BASE_URL

    Args:
        dotenv_path: Path to .env file (default: .env in current directory)

    Returns:
        Configuration dictionary from .env file
    """
    from dotenv import dotenv_values

    if dotenv_path is None:
        dotenv_path = Path(".env")

    if not dotenv_path.exists():
        return {}

    # Load .env file values (without modifying environment)
    env_values = dotenv_values(dotenv_path)

    config: Dict[str, Any] = {}

    # Handle authentication variables with standard names
    if "CLAUDE_CODE_OAUTH_TOKEN" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["oauth_token"] = env_values["CLAUDE_CODE_OAUTH_TOKEN"]

    if "ANTHROPIC_API_KEY" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["anthropic_api_key"] = env_values["ANTHROPIC_API_KEY"]

    if "ANTHROPIC_BASE_URL" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["anthropic_base_url"] = env_values["ANTHROPIC_BASE_URL"]

    if "OPENAI_API_KEY" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["openai_api_key"] = env_values["OPENAI_API_KEY"]

    if "OPENAI_BASE_URL" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["openai_base_url"] = env_values["OPENAI_BASE_URL"]

    return config


def load_env_config() -> Dict[str, Any]:
    """
    Load environment variables into a config dict.

    Supports standard names (CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_*, OPENAI_*).
    """
    config: Dict[str, Any] = {}

    # First, handle standard authentication environment variables
    if "CLAUDE_CODE_OAUTH_TOKEN" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["oauth_token"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]

    if "ANTHROPIC_API_KEY" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]

    if "ANTHROPIC_BASE_URL" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["anthropic_base_url"] = os.environ["ANTHROPIC_BASE_URL"]

    if "OPENAI_API_KEY" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["openai_api_key"] = os.environ["OPENAI_API_KEY"]

    if "OPENAI_BASE_URL" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["openai_base_url"] = os.environ["OPENAI_BASE_URL"]

    # No Pitaya-specific env variables; use CLI or config file

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "model": "sonnet",
        "strategy": "simple",
        "output": "tui",
        "state_dir": Path("./pitaya_state"),
        "logs_dir": Path("./logs"),
        # Debug mode removed; logs are verbose by default
        # Normative defaults per spec §6.1.1 (subset applied where supported)
        "import_policy": "auto",  # auto|never|always
        "import_conflict_policy": "fail",  # fail|overwrite|suffix
        "skip_empty_import": True,
        "runner": {
            "timeout": 3600,  # 1 hour
            "cpu_limit": 2.0,
            "memory_limit": "4g",
            # Session volume scope: run|global
            "session_volume_scope": "run",
            # review workspace mode: rw|ro (ro enforces RO when import_policy=never)
            "review_workspace_mode": "ro",
            # tmpfs size for /tmp mount in MB
            "tmpfs_size_mb": 512,
        },
        "orchestration": {
            # Default concurrency: auto-calculated from CPU count
            "max_parallel_instances": "auto",
            # Startup cap: auto -> min(10, max_parallel_instances)
            "max_parallel_startup": "auto",
            # Branch namespace is hierarchical by default
            # Format: orc/<strategy>/<run_id>/k<short8>
            "branch_namespace": "hierarchical",
            "snapshot_interval": 30,  # seconds
            "event_buffer_size": 10000,
            # When true, execute queued instances in random order (not FIFO)
            "randomize_queue_order": False,
        },
        "strategy_config": {
            "n": 3,
            "threshold": 0.8,
            "max_iterations": 3,
        },
        "tui": {
            # preferred configuration is ms; keep Hz fallback for compatibility
            "refresh_rate": 10,  # Hz (fallback)
            "refresh_rate_ms": 100,  # 10Hz
            "show_timestamps": False,
            "color_scheme": "accessible",
            "details_messages": 10,
        },
        "logging": {
            "level": "INFO",
            "max_file_size": 10485760,  # 10MB
            "retention_days": 7,  # component logs only
            "redaction": {
                "custom_patterns": [],
            },
        },
    }


def select_auth_mode(config: Dict[str, Any]) -> str:
    """
    Select authentication mode based on configuration and available credentials.

    Selection logic (per specification):
    1. If --mode api specified, use API key
    2. If OAuth token present, use subscription mode
    3. If only API key present, use API mode
    4. Otherwise, error

    Returns:
        "subscription" or "api"

    Raises:
        ValueError: If no valid authentication is configured
    """
    runner_config = config.get("runner", {})

    # Check if mode was explicitly set (from CLI)
    explicit_mode = config.get("auth_mode") or runner_config.get("auth_mode")

    oauth_token = runner_config.get("oauth_token")
    api_key = runner_config.get("api_key")

    # Step 1: If mode explicitly set to api, use API key
    if explicit_mode == "api":
        if not api_key:
            raise ValueError("API mode specified but no ANTHROPIC_API_KEY found")
        return "api"

    # Step 2: If OAuth token present, use subscription mode
    if oauth_token:
        return "subscription"

    # Step 3: If only API key present, use API mode
    if api_key:
        return "api"

    # Step 4: No auth configured
    raise ValueError(
        "No authentication configured. Please set either:\n"
        "  - CLAUDE_CODE_OAUTH_TOKEN for subscription mode (recommended)\n"
        "  - ANTHROPIC_API_KEY for API mode\n"
        "See docs for authentication setup."
    )


def validate_auth_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate authentication configuration.

    Ensures required credentials are present for the selected mode.

    Returns:
        Validated auth configuration with mode set

    Raises:
        ValueError: If authentication is invalid
    """
    runner_config = config.get("runner", {})

    # Determine auth mode
    auth_mode = select_auth_mode(config)

    # Store selected mode in config
    if "runner" not in config:
        config["runner"] = {}
    config["runner"]["auth_mode"] = auth_mode

    # Validate based on mode
    if auth_mode == "subscription":
        if not runner_config.get("oauth_token"):
            raise ValueError("Subscription mode requires CLAUDE_CODE_OAUTH_TOKEN")
    elif auth_mode == "api":
        if not runner_config.get("api_key"):
            raise ValueError("API mode requires ANTHROPIC_API_KEY")

    return config


def load_config(
    cli_args: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load configuration from all sources with proper precedence.

    Precedence order (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. .env file
    4. pitaya.yaml
    5. Built-in defaults

    Args:
        cli_args: Command line arguments (highest precedence)
        yaml_path: Path to pitaya.yaml (default: ./pitaya.yaml)
        dotenv_path: Path to .env file (default: ./.env)

    Returns:
        Merged configuration dictionary

    Raises:
        ValueError: If authentication configuration is invalid
    """
    # Load from each source
    defaults = get_default_config()
    yaml_config = load_yaml_config(yaml_path)
    dotenv_config = load_dotenv_config(dotenv_path)
    env_config = load_env_config()
    cli_args = cli_args or {}

    # Merge with proper precedence
    config = merge_config(
        cli_args=cli_args,
        env_config=env_config,
        dotenv_config=dotenv_config,
        file_config=yaml_config,
        defaults=defaults,
    )

    # Normalize YAML alias keys to match spec naming
    config = _apply_yaml_aliases(config)

    # Validate authentication
    config = validate_auth_config(config)

    return config
