"""
Environment selection helpers for the Codex plugin.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

ENV_API_KEY = "OPENAI_API_KEY"
ENV_BASE_URL = "OPENAI_BASE_URL"
ENV_PROVIDER_OVERRIDE = "CODEX_ENV_KEY"
ENV_BASE_OVERRIDE = "CODEX_BASE_URL"
ENV_PROVIDER_NAME_OVERRIDE = "CODEX_MODEL_PROVIDER"
ENV_CODEX_API_KEY = "CODEX_API_KEY"

PROVIDER_ENV_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    (ENV_CODEX_API_KEY, ENV_BASE_OVERRIDE),
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


def collect_codex_env(auth_config: Optional[Dict[str, str]]) -> Dict[str, str]:
    env: Dict[str, str] = {}

    if auth_config:
        api_key = auth_config.get("api_key")
        if api_key:
            env[ENV_CODEX_API_KEY] = str(api_key)
            env.setdefault(ENV_API_KEY, str(api_key))
        base_url = auth_config.get("base_url")
        if base_url:
            val = str(base_url).strip()
            if val:
                env[ENV_BASE_URL] = val

    for env_key, base_key in PROVIDER_ENV_CANDIDATES:
        if val := os.environ.get(env_key):
            env.setdefault(env_key, val)
        if base_val := os.environ.get(base_key):
            env.setdefault(base_key, base_val)

    override_env = os.environ.get(ENV_PROVIDER_OVERRIDE)
    if override_env and os.environ.get(override_env):
        env.setdefault(override_env, os.environ.get(override_env))
    base_override = os.environ.get(ENV_BASE_OVERRIDE)
    if base_override:
        env.setdefault(ENV_BASE_URL, base_override)

    return env


def select_provider_env_key() -> Optional[str]:
    """Choose which provider API key env to use, ignoring empties."""
    override = os.environ.get(ENV_PROVIDER_OVERRIDE)
    if override and os.environ.get(override):
        return override
    if os.environ.get(ENV_CODEX_API_KEY):
        return ENV_CODEX_API_KEY
    if os.environ.get(ENV_API_KEY):
        return ENV_API_KEY
    for env_key, _ in PROVIDER_ENV_CANDIDATES:
        if os.environ.get(env_key):
            return env_key
    return None


def select_provider_base_url(env_key: Optional[str]) -> Optional[str]:
    base = os.environ.get(ENV_BASE_OVERRIDE)
    if base:
        return base
    if env_key:
        for candidate_key, base_key in PROVIDER_ENV_CANDIDATES:
            if candidate_key == env_key:
                if bval := os.environ.get(base_key):
                    return bval
    return os.environ.get(ENV_BASE_URL) or None


def select_provider_name(env_key: Optional[str]) -> Optional[str]:
    override = os.environ.get(ENV_PROVIDER_NAME_OVERRIDE)
    if override:
        return override
    if not env_key or env_key == ENV_API_KEY:
        return None
    prefix = env_key.lower().replace("_api_key", "")
    return prefix or None
