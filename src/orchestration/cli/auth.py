"""Auth configuration helpers for CLI.

Validates credentials early so runs fail fast with clear guidance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import argparse
import os

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


def get_auth_config(
    args: argparse.Namespace, config: Optional[Dict[str, Any]] = None
) -> AuthConfig:
    """Derive AuthConfig from CLI/env/dotenv/config (plugin-aware).

    Raises:
        ValueError: when the selected mode lacks required credentials.
    """
    full_config = _merge_sources(args, config)
    runner_cfg = full_config.get("runner", {})
    plugin_name = _select_provider_plugin(args, full_config)
    if plugin_name == "codex":
        api_key = runner_cfg.get("api_key") or runner_cfg.get("openai_api_key")
        base_url = runner_cfg.get("base_url") or runner_cfg.get("openai_base_url")
        # Fallback: detect provider-specific env vars (OpenRouter, Groq, etc.)
        if not api_key:
            _env_key, _api, _url = _detect_codex_api_from_env()
            if _api:
                api_key = _api
                if not base_url and _url:
                    base_url = _url
        oauth_token = None
    else:
        oauth_token = runner_cfg.get("oauth_token")
        api_key = runner_cfg.get("api_key") or runner_cfg.get("anthropic_api_key")
        base_url = runner_cfg.get("base_url") or runner_cfg.get("anthropic_base_url")
    oauth_token, api_key = _apply_mode(args, oauth_token, api_key)
    # Validate presence according to mode and plugin
    mode = getattr(args, "mode", None)
    if mode == "api" and not api_key:
        # API mode requires an API key
        if plugin_name == "codex":
            raise ValueError(
                "missing API key for Codex; set a provider key (e.g., OPENAI_API_KEY/OPENROUTER_API_KEY) or use --api-key"
            )
        raise ValueError("missing API key; set ANTHROPIC_API_KEY (or use --api-key)")
    if mode == "subscription" and not oauth_token:
        raise ValueError(
            "missing OAuth token for subscription mode; set CLAUDE_CODE_OAUTH_TOKEN or use --oauth-token"
        )
    if not mode and not (oauth_token or api_key):
        # Auto-detect mode: require at least one credential depending on plugin
        if plugin_name == "codex":
            raise ValueError(
                "no Codex credentials found; set a provider API key (e.g., OPENAI_API_KEY/OPENROUTER_API_KEY) or pass --api-key"
            )
        raise ValueError(
            "no credentials found; set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY"
        )
    return AuthConfig(oauth_token=oauth_token, api_key=api_key, base_url=base_url)


# Known OpenAI-compatible provider env pairs
_PROVIDER_ENV_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"),
    ("GROQ_API_KEY", "GROQ_BASE_URL"),
    ("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    ("GEMINI_API_KEY", "GEMINI_BASE_URL"),
    ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    ("OLLAMA_API_KEY", "OLLAMA_BASE_URL"),
    ("ARCEEAI_API_KEY", "ARCEEAI_BASE_URL"),
)


def _detect_codex_api_from_env() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Detect a provider API key and base URL from environment variables.

    Returns: (env_key_name, api_key, base_url) or (None, None, None).
    Honors overrides via CODEX_ENV_KEY/CODEX_BASE_URL when set.
    """
    override = os.environ.get("CODEX_ENV_KEY")
    if override:
        api = os.environ.get(override)
        base = os.environ.get("CODEX_BASE_URL")
        if not base and override.endswith("_API_KEY"):
            bkey = f"{override[:-7]}_BASE_URL"
            base = os.environ.get(bkey)
        if api:
            return override, api, base
    # Prefer OPENAI_API_KEY
    if "OPENAI_API_KEY" in os.environ:
        return (
            "OPENAI_API_KEY",
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_BASE_URL"),
        )
    for k, b in _PROVIDER_ENV_CANDIDATES:
        if k in os.environ:
            return k, os.environ.get(k), os.environ.get(b)
    return None, None, None
