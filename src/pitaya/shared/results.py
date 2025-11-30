"""Instance result data structures shared across layers."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .type_aliases import Metrics


@dataclass
class InstanceResult:
    """Result from running a single instance."""

    success: bool
    branch_name: Optional[str] = None
    has_changes: bool = False
    final_message: Optional[str] = None
    session_id: Optional[str] = None
    container_name: Optional[str] = None
    metrics: Metrics = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    # Additional fields from specification
    commit_statistics: Optional[Metrics] = None  # count, lines added/deleted
    started_at: Optional[str] = None  # ISO timestamp
    completed_at: Optional[str] = None  # ISO timestamp
    retry_attempts: int = 0
    log_path: Optional[str] = None
    workspace_path: Optional[str] = None  # Until cleanup
    status: str = "unknown"  # success/failed/timeout/canceled
    # Artifact extras for orchestration mapping
    commit: Optional[str] = None
    duplicate_of_branch: Optional[str] = None
    dedupe_reason: Optional[str] = None
    # Strategy-specific metadata
    metadata: Metrics = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """Convenience: total cost from metrics (spec)."""
        # Metrics use 'total_cost' throughout the codebase
        return float(self.metrics.get("total_cost", self.metrics.get("cost", 0.0)))

    @property
    def tokens(self) -> int:
        """Convenience: total tokens from metrics (spec)."""
        return int(self.metrics.get("total_tokens", 0))

    @property
    def token_breakdown(self) -> Dict[str, int]:
        """Token breakdown dict: input/output/total."""
        return {
            "input": int(self.metrics.get("input_tokens", 0)),
            "output": int(self.metrics.get("output_tokens", 0)),
            "total": int(self.metrics.get("total_tokens", 0)),
        }


__all__ = ["InstanceResult"]
