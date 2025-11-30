"""Configuration models, loading, validation, and defaults."""

from pitaya.config.models import AuthConfig, ContainerLimits, RetryConfig
from pitaya.config.loader import load_project_config, build_cli_config
from pitaya.config.strategy import get_strategy_config
from pitaya.config.defaults import (
    merge_config,
    deep_merge,
    load_yaml_config,
    load_global_config,
    load_dotenv_config,
    load_env_config,
    get_default_config,
    select_auth_mode,
    validate_auth_config,
    load_config,
)

__all__ = [
    "AuthConfig",
    "ContainerLimits",
    "RetryConfig",
    "load_project_config",
    "build_cli_config",
    "get_strategy_config",
    "merge_config",
    "deep_merge",
    "load_yaml_config",
    "load_global_config",
    "load_dotenv_config",
    "load_env_config",
    "get_default_config",
    "select_auth_mode",
    "validate_auth_config",
    "load_config",
    "validate_full_config",
]


def validate_full_config(*args, **kwargs):
    """Lazily import validation to avoid circular imports at package import."""
    from pitaya.config.validation import validate_full_config as _vf

    return _vf(*args, **kwargs)
