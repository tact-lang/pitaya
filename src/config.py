"""
Configuration management for the orchestrator.

Handles configuration precedence: CLI > ENV > .env > orchestrator.yaml > defaults
Supports ORCHESTRATOR_* environment variables for all settings.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def get_env_value(key: str, default: Any = None) -> Any:
    """
    Get value from ORCHESTRATOR_* environment variable with type conversion.

    Environment variable naming convention:
    - Nested keys use double underscores: orchestration__max_parallel_instances
    - Boolean values: 'true', 'yes', '1' = True; 'false', 'no', '0' = False
    - Numbers are automatically converted to int or float
    """
    env_key = f"ORCHESTRATOR_{key.upper()}"
    value = os.environ.get(env_key)

    if value is None:
        return default

    # Type conversion
    # Try boolean first
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


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
    4. orchestrator.yaml config file
    5. Defaults
    """
    # Start with defaults
    result = defaults.copy()

    # Overlay file config (orchestrator.yaml)
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
    Load configuration from orchestrator.yaml file.

    Args:
        yaml_path: Path to YAML config file (default: orchestrator.yaml in current directory)

    Returns:
        Configuration dictionary from YAML file
    """
    if yaml_path is None:
        yaml_path = Path("orchestrator.yaml")

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


def load_dotenv_config(dotenv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from .env file.

    Supports standard environment variables:
    - CLAUDE_CODE_OAUTH_TOKEN
    - ANTHROPIC_API_KEY
    - ANTHROPIC_BASE_URL

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
        config["runner"]["api_key"] = env_values["ANTHROPIC_API_KEY"]

    if "ANTHROPIC_BASE_URL" in env_values:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["base_url"] = env_values["ANTHROPIC_BASE_URL"]

    return config


def load_env_config() -> Dict[str, Any]:
    """
    Load environment variables into a config dict.

    Supports both standard names (CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY)
    and ORCHESTRATOR_* prefixed variables for other settings.
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
        config["runner"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]

    if "ANTHROPIC_BASE_URL" in os.environ:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["base_url"] = os.environ["ANTHROPIC_BASE_URL"]

    # Then handle ORCHESTRATOR_* variables for other settings
    env_mappings = {
        # Top-level settings
        "MODEL": "model",
        # Also support default model name per spec
        "DEFAULT_MODEL": "model",
        "STRATEGY": "strategy",
        "REPO": "repo",
        "BASE_BRANCH": "base_branch",
        "RUNS": "runs",
        "OUTPUT": "output",
        "STATE_DIR": "state_dir",
        "LOGS_DIR": "logs_dir",
        "HTTP_PORT": "http_port",
        "DEBUG": "debug",
        # Runner settings
        "RUNNER__OAUTH_TOKEN": "runner.oauth_token",
        "RUNNER__API_KEY": "runner.api_key",
        "RUNNER__BASE_URL": "runner.base_url",
        "RUNNER__TIMEOUT": "runner.timeout",
        "RUNNER__CPU_LIMIT": "runner.cpu_limit",
        "RUNNER__MEMORY_LIMIT": "runner.memory_limit",
        # Spec-friendly resource names
        "CONTAINER_CPU": "runner.cpu_limit",
        "CONTAINER_MEMORY": "runner.memory_limit",
        # Orchestration settings
        "ORCHESTRATION__MAX_PARALLEL_INSTANCES": "orchestration.max_parallel_instances",
        "ORCHESTRATION__SNAPSHOT_INTERVAL": "orchestration.snapshot_interval",
        "ORCHESTRATION__EVENT_BUFFER_SIZE": "orchestration.event_buffer_size",
        "ORCHESTRATION__CONTAINER_RETENTION_FAILED": "orchestration.container_retention_failed",
        "ORCHESTRATION__CONTAINER_RETENTION_SUCCESS": "orchestration.container_retention_success",
        # Strategy parameters
        "STRATEGY__N": "strategy.n",
        "STRATEGY__THRESHOLD": "strategy.threshold",
        "STRATEGY__MAX_ITERATIONS": "strategy.max_iterations",
        "STRATEGY__SCORING_MODEL": "strategy.scoring_model",
        "STRATEGY__FORCE_IMPORT": "strategy.force_import",
        # Logging settings
        "LOGGING__LEVEL": "logging.level",
        "LOGGING__MAX_FILE_SIZE": "logging.max_file_size",
        "LOGGING__RETENTION_DAYS": "logging.retention_days",
        # TUI settings
        "TUI__REFRESH_RATE": "tui.refresh_rate",
        "TUI__SHOW_TIMESTAMPS": "tui.show_timestamps",
        "TUI__COLOR_SCHEME": "tui.color_scheme",
        "TUI__FORCE_DISPLAY_MODE": "tui.force_display_mode",
    }

    for env_suffix, config_path in env_mappings.items():
        value = get_env_value(env_suffix)
        if value is not None:
            # Convert path like "runner.timeout" to nested dict
            parts = config_path.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    # Dynamic strategy parameters: ORCHESTRATOR_STRATEGY__<NAME>__KEY=VALUE
    def _auto_convert(val: str):
        low = val.lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    for key, value in os.environ.items():
        if not key.startswith("ORCHESTRATOR_STRATEGY__"):
            continue
        try:
            _, _, rest = key.partition("ORCHESTRATOR_STRATEGY__")
            strat_name, _, param = rest.partition("__")
            if not strat_name or not param:
                continue
            # Normalize
            strategy = strat_name.lower().replace(" ", "_")
            param_key = param.lower()
            # Convert value types
            converted = _auto_convert(value)
            # Place into config under strategies
            strategies = config.setdefault("strategies", {})
            strategies.setdefault(strategy, {})[param_key] = converted
        except Exception:
            continue

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "model": "sonnet",
        "strategy": "simple",
        "output": "tui",
        "state_dir": Path("./orchestrator_state"),
        "logs_dir": Path("./logs"),
        "debug": False,
        "runner": {
            "timeout": 3600,  # 1 hour
            "cpu_limit": 2.0,
            "memory_limit": "4g",
        },
        "orchestration": {
            "max_parallel_instances": 20,
            "snapshot_interval": 30,  # seconds
            "event_buffer_size": 10000,
            "container_retention_failed": 86400,  # 24 hours
            "container_retention_success": 7200,  # 2 hours
        },
        "strategy_config": {
            "n": 3,
            "threshold": 0.8,
            "max_iterations": 3,
        },
        "logging": {
            "level": "INFO",
            "max_file_size": 10485760,  # 10MB
            "retention_days": 7,
        },
        "tui": {
            "refresh_rate": 10,  # Hz
            "show_timestamps": False,
            "color_scheme": "default",
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
    4. orchestrator.yaml
    5. Built-in defaults

    Args:
        cli_args: Command line arguments (highest precedence)
        yaml_path: Path to orchestrator.yaml (default: ./orchestrator.yaml)
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
