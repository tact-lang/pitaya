"""Auth configuration helpers for CLI."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...config import (
    get_default_config,
    load_dotenv_config,
    load_env_config,
    merge_config,
)
from ...shared import AuthConfig

__all__ = ["get_auth_config"]


def _merge_sources(args, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    env_config = load_env_config()
    dotenv_config = load_dotenv_config()
    defaults = get_default_config()
    cli_config: Dict[str, Any] = {}
    if getattr(args, "oauth_token", None):
        cli_config.setdefault("runner", {})["oauth_token"] = args.oauth_token
    if getattr(args, "api_key", None):
        cli_config.setdefault("runner", {})["api_key"] = args.api_key
    if getattr(args, "base_url", None):
        cli_config.setdefault("runner", {})["base_url"] = args.base_url
    if getattr(args, "plugin", None):
        cli_config["plugin_name"] = args.plugin
    return merge_config(cli_config, env_config, dotenv_config, config or {}, defaults)


def _select_provider_plugin(args, full_config: Dict[str, Any]) -> str:
    return str(full_config.get("plugin_name") or getattr(args, "plugin", "claude-code"))


def _apply_mode(
    args, oauth_token: Optional[str], api_key: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    mode = getattr(args, "mode", None)
    if mode == "api":
        return (None, api_key)
    if mode == "subscription":
        return (oauth_token, None)
    # auto mode: prefer oauth if present
    if oauth_token:
        return (oauth_token, None)
    return (None, api_key)


def get_auth_config(args, config: Optional[Dict[str, Any]] = None) -> AuthConfig:
    """Derive AuthConfig from CLI/env/dotenv/config (plugin-aware)."""
    full_config = _merge_sources(args, config)
    runner_cfg = full_config.get("runner", {})
    plugin_name = _select_provider_plugin(args, full_config)
    if plugin_name == "codex":
        api_key = runner_cfg.get("api_key") or runner_cfg.get("openai_api_key")
        base_url = runner_cfg.get("base_url") or runner_cfg.get("openai_base_url")
        oauth_token = None
    else:
        oauth_token = runner_cfg.get("oauth_token")
        api_key = runner_cfg.get("api_key") or runner_cfg.get("anthropic_api_key")
        base_url = runner_cfg.get("base_url") or runner_cfg.get("anthropic_base_url")
    oauth_token, api_key = _apply_mode(args, oauth_token, api_key)
    return AuthConfig(oauth_token=oauth_token, api_key=api_key, base_url=base_url)
