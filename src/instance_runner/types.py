"""
Common types and data classes for the instance runner.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
    docker_error_patterns: tuple = (
        "connection refused",
        "no such host",
        "timeout",
        "Cannot connect to the Docker daemon",
        "already in use",  # name conflict
        "409 Client Error",  # explicit status hint
        'Conflict ("Conflict.',  # docker's overlap wording
    )
    agent_error_patterns: tuple = (
        "rate limit",
        "API error",
        "connection reset",
        "429",  # Rate limit status code
    )
    general_error_patterns: tuple = (
        "ECONNREFUSED",
        "ETIMEDOUT",
        "ENETUNREACH",
        "Connection timed out",
    )


@dataclass
class InstanceResult:
    """Result from running a single instance."""

    success: bool
    branch_name: Optional[str] = None
    has_changes: bool = False
    final_message: Optional[str] = None
    session_id: Optional[str] = None
    container_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    # Additional fields from specification
    commit_statistics: Optional[Dict[str, Any]] = None  # count, lines added/deleted
    started_at: Optional[str] = None  # ISO timestamp
    completed_at: Optional[str] = None  # ISO timestamp
    retry_attempts: int = 0
    log_path: Optional[str] = None
    workspace_path: Optional[str] = None  # Until cleanup
    status: str = "unknown"  # success/failed/timeout/canceled
