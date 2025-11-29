"""Retry classification helper for instance attempts."""

from __future__ import annotations

from ..shared import RetryConfig


def is_retryable_error(
    error_str: str, error_type: str, retry_config: RetryConfig
) -> bool:
    error_lower = error_str.lower()
    if error_type == "timeout":
        return True
    if error_type == "docker" and any(
        p.lower() in error_lower for p in retry_config.docker_error_patterns
    ):
        return True
    if error_type in ("agent",) and any(
        p.lower() in error_lower for p in retry_config.agent_error_patterns
    ):
        return True
    return any(p.lower() in error_lower for p in retry_config.general_error_patterns)


__all__ = ["is_retryable_error"]
