"""Configuration dataclasses shared across layers."""

from dataclasses import dataclass
from typing import Optional

from .type_aliases import ErrorPatterns


@dataclass
class ContainerLimits:
    """Resource limits for Docker containers."""

    cpu_count: int = 2
    memory_gb: int = 4
    memory_swap_gb: int = 4  # Total memory + swap


@dataclass
class AuthConfig:
    """Authentication configuration for AI tools."""

    oauth_token: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_attempts: int = 3
    initial_delay_seconds: float = 10.0  # Per spec: 10s, 60s, 360s
    max_delay_seconds: float = 360.0
    exponential_base: float = 6.0  # To get 10s -> 60s -> 360s progression
    # Pattern-based retry logic as per spec
    docker_error_patterns: ErrorPatterns = (
        "connection refused",
        "no such host",
        "timeout",
        "daemon",  # broader match per spec
        "Cannot connect to the Docker daemon",
        "already in use",  # name conflict
        "409 Client Error",  # explicit status hint
        'Conflict ("Conflict.',  # docker's overlap wording
    )
    agent_error_patterns: ErrorPatterns = (
        "rate limit",
        "API error",
        "connection reset",
        "overloaded_error",
        "429",  # Rate limit status code
    )
    general_error_patterns: ErrorPatterns = (
        "ECONNREFUSED",
        "ETIMEDOUT",
        "ENETUNREACH",
        "Connection timed out",
    )


__all__ = ["AuthConfig", "ContainerLimits", "RetryConfig"]
