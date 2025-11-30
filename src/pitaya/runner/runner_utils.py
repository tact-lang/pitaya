"""Utility helpers for instance runner orchestration."""

from __future__ import annotations

from pitaya.shared import RetryConfig


def is_retryable_error(
    error_str: str, error_type: str, retry_config: RetryConfig
) -> bool:
    """Return True when the error matches a retry pattern or is a timeout."""
    error_lower = error_str.lower()

    if error_type == "timeout":
        return True

    if error_type == "docker":
        return any(
            pat.lower() in error_lower for pat in retry_config.docker_error_patterns
        )

    if error_type == "agent":
        return any(
            pat.lower() in error_lower for pat in retry_config.agent_error_patterns
        )

    return any(
        pat.lower() in error_lower for pat in retry_config.general_error_patterns
    )
