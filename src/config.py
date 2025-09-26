"""Configuration management utilities for Pitaya."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = [
    "deep_merge",
    "get_default_config",
    "load_config",
    "load_dotenv_config",
    "load_env_config",
    "load_global_config",
    "load_yaml_config",
    "merge_config",
    "select_auth_mode",
    "validate_auth_config",
]

logger = logging.getLogger(__name__)

_RUNNER_SECTION = "runner"
_ENV_TO_CONFIG_KEY = {
    "CLAUDE_CODE_OAUTH_TOKEN": "oauth_token",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "ANTHROPIC_BASE_URL": "anthropic_base_url",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    # Allow OpenAI-compatible providers via OpenRouter to flow through config
    # so CLI auth detection doesn't rely solely on runtime env probing.
    "OPENROUTER_API_KEY": "api_key",
    "OPENROUTER_BASE_URL": "base_url",
}


def merge_config(
    cli_args: Dict[str, Any],
    env_config: Dict[str, Any],
    dotenv_config: Dict[str, Any],
    file_config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configuration dictionaries honoring precedence order."""
    merged = defaults.copy()
    deep_merge(merged, file_config)
    deep_merge(merged, dotenv_config)
    deep_merge(merged, env_config)
    cli_filtered = {key: value for key, value in cli_args.items() if value is not None}
    deep_merge(merged, cli_filtered)
    return merged


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Recursively merge ``overlay`` into ``base`` in place."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def load_yaml_config(yaml_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from ``pitaya.yaml`` or return an empty dict."""
    path = yaml_path or Path("pitaya.yaml")
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}

    return data if isinstance(data, dict) else {}


def load_global_config() -> Dict[str, Any]:
    """Load user-level configuration from standard locations."""
    try:
        home = Path.home()
    except OSError:
        return {}

    for candidate in (
        home / ".pitaya" / "config.yaml",
        home / ".config" / "pitaya" / "config.yaml",
    ):
        try:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                if isinstance(data, dict):
                    return data
        except (yaml.YAMLError, OSError):
            continue
    return {}


def load_dotenv_config(dotenv_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load authentication data from a ``.env`` file."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - optional dependency
        return {}

    path = dotenv_path or Path(".env")
    if not path.exists():
        return {}

    values = dotenv_values(path)
    config: Dict[str, Any] = {}
    for env_key, config_key in _ENV_TO_CONFIG_KEY.items():
        if env_key in values:
            _set_runner_value(config, config_key, values[env_key])
    return config


def load_env_config() -> Dict[str, Any]:
    """Load supported authentication variables from the current environment."""
    config: Dict[str, Any] = {}
    for env_key, config_key in _ENV_TO_CONFIG_KEY.items():
        if env_key in os.environ:
            _set_runner_value(config, config_key, os.environ[env_key])
    return config


def get_default_config() -> Dict[str, Any]:
    """Return a fresh copy of Pitaya's default configuration."""
    return {
        "model": "sonnet",
        "strategy": "simple",
        "output": "tui",
        "state_dir": Path("./pitaya_state"),
        "logs_dir": Path("./logs"),
        "import_policy": "auto",
        "import_conflict_policy": "fail",
        "skip_empty_import": True,
        "runner": {
            "timeout": 3600,
            "cpu_limit": 2.0,
            "memory_limit": "4g",
            "session_volume_scope": "run",
            "review_workspace_mode": "ro",
            "tmpfs_size_mb": 512,
        },
        "orchestration": {
            "max_parallel_instances": "auto",
            "max_parallel_startup": "auto",
            "branch_namespace": "hierarchical",
            "snapshot_interval": 30,
            "event_buffer_size": 10000,
            "randomize_queue_order": False,
        },
        "strategy_config": {
            "n": 3,
            "threshold": 0.8,
            "max_iterations": 3,
        },
        "tui": {
            "refresh_rate": 10,
            "refresh_rate_ms": 100,
            "show_timestamps": False,
            "color_scheme": "accessible",
            "details_messages": 10,
        },
        "logging": {
            "level": "INFO",
            "max_file_size": 10_485_760,
            "retention_days": 7,
            "redaction": {"custom_patterns": []},
        },
    }


def select_auth_mode(config: Dict[str, Any]) -> str:
    """Determine whether to use ``subscription`` or ``api`` mode."""
    runner_config = config.get(_RUNNER_SECTION, {})
    explicit_mode = config.get("auth_mode") or runner_config.get("auth_mode")
    oauth_token = runner_config.get("oauth_token")
    api_key = runner_config.get("api_key")

    if explicit_mode == "api":
        if not api_key:
            raise ValueError("API mode specified but no ANTHROPIC_API_KEY found")
        return "api"

    if oauth_token:
        return "subscription"

    if api_key:
        return "api"

    raise ValueError(
        "No authentication configured. Please set either:\n"
        "  - CLAUDE_CODE_OAUTH_TOKEN for subscription mode (recommended)\n"
        "  - ANTHROPIC_API_KEY for API mode\n"
        "See docs for authentication setup."
    )


def validate_auth_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate authentication settings and persist the resolved mode."""
    runner_config = config.setdefault(_RUNNER_SECTION, {})
    auth_mode = select_auth_mode(config)
    runner_config["auth_mode"] = auth_mode

    if auth_mode == "subscription" and not runner_config.get("oauth_token"):
        raise ValueError("Subscription mode requires CLAUDE_CODE_OAUTH_TOKEN")
    if auth_mode == "api" and not runner_config.get("api_key"):
        raise ValueError("API mode requires ANTHROPIC_API_KEY")

    return config


def load_config(
    cli_args: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load configuration from defaults, files, environment, and CLI."""
    config = merge_config(
        cli_args=cli_args or {},
        env_config=load_env_config(),
        dotenv_config=load_dotenv_config(dotenv_path),
        file_config=load_yaml_config(yaml_path),
        defaults=get_default_config(),
    )

    _apply_yaml_aliases(config)
    return validate_auth_config(config)


def _apply_yaml_aliases(config: Dict[str, Any]) -> None:
    """Normalize legacy YAML keys to their canonical names in place."""

    def _set_runner(key: str, value: Any) -> None:
        config.setdefault(_RUNNER_SECTION, {})[key] = value

    if "container_cpu" in config:
        _set_runner("cpu_limit", config.pop("container_cpu"))
    if "container_memory" in config:
        _set_runner("memory_limit", config.pop("container_memory"))

    runner = config.get(_RUNNER_SECTION, {})
    if "container_cpu" in runner:
        _set_runner("cpu_limit", runner.pop("container_cpu"))
    if "container_memory" in runner:
        _set_runner("memory_limit", runner.pop("container_memory"))


def _set_runner_value(config: Dict[str, Any], key: str, value: Any) -> None:
    config.setdefault(_RUNNER_SECTION, {})[key] = value
