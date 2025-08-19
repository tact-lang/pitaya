"""
Data models for TUI state tracking.

These models represent the in-memory state that the TUI maintains,
derived from the event stream and periodic state queries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class InstanceStatus(Enum):
    """Visual status for instance display."""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def emoji(self) -> str:
        """Get emoji representation for status."""
        emoji_map = {
            InstanceStatus.QUEUED: "⏳",
            InstanceStatus.RUNNING: "🔄",
            InstanceStatus.INTERRUPTED: "⏸️",
            InstanceStatus.COMPLETED: "✅",
            InstanceStatus.FAILED: "❌",
        }
        return emoji_map[self]

    @property
    def color(self) -> str:
        """Get Rich color for status."""
        color_map = {
            InstanceStatus.QUEUED: "dim",
            InstanceStatus.RUNNING: "yellow",
            InstanceStatus.INTERRUPTED: "magenta",
            InstanceStatus.COMPLETED: "green",
            InstanceStatus.FAILED: "red",
        }
        return color_map[self]


@dataclass
class InstanceDisplay:
    """Display state for a single instance."""

    instance_id: str
    strategy_name: str
    status: InstanceStatus = InstanceStatus.QUEUED
    branch_name: Optional[str] = None
    prompt: Optional[str] = None
    model: str = ""

    # Progress tracking
    current_activity: Optional[str] = None
    last_tool_use: Optional[str] = None

    # Metrics
    duration_seconds: float = 0.0
    cost: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Final message
    final_message: Optional[str] = None
    final_message_truncated: bool = False
    final_message_path: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if instance is currently active."""
        return self.status in (InstanceStatus.QUEUED, InstanceStatus.RUNNING)

    @property
    def display_name(self) -> str:
        """Get short display name for instance."""
        return self.instance_id[:8]


@dataclass
class StrategyDisplay:
    """Display state for a strategy execution."""

    strategy_id: str
    strategy_name: str
    config: Dict[str, Any] = field(default_factory=dict)

    # Instance tracking
    instance_ids: List[str] = field(default_factory=list)

    # Progress
    total_instances: int = 0
    completed_instances: int = 0
    failed_instances: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    is_complete: bool = False

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_instances == 0:
            return 0.0
        return (
            (self.completed_instances + self.failed_instances)
            / self.total_instances
            * 100
        )

    @property
    def status_summary(self) -> str:
        """Get status summary string."""
        if self.is_complete:
            return f"✓ {self.completed_instances}/{self.total_instances}"
        else:
            return f"🔄 {self.completed_instances + self.failed_instances}/{self.total_instances}"


@dataclass
class RunDisplay:
    """Display state for the entire run."""

    run_id: str
    prompt: str = ""
    repo_path: str = ""
    base_branch: str = "main"

    # Strategy tracking
    strategies: Dict[str, StrategyDisplay] = field(default_factory=dict)

    # Instance tracking
    instances: Dict[str, InstanceDisplay] = field(default_factory=dict)

    # Aggregate metrics
    total_cost: float = 0.0
    total_tokens: int = 0
    total_instances: int = 0
    active_instances: int = 0
    completed_instances: int = 0
    failed_instances: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # UI session start (for global runtime and clamping)
    ui_started_at: Optional[datetime] = None

    # Display preferences
    force_detail_level: Optional[str] = None  # detailed, compact, dense

    @property
    def duration_seconds(self) -> float:
        """Calculate run duration."""
        if not self.started_at:
            return 0.0
        # Use a tz-aware/naive match to avoid subtraction errors
        start = self.started_at
        now = datetime.now(start.tzinfo) if start.tzinfo is not None else datetime.now()
        end_time = self.completed_at or now
        # If completed_at exists but tzinfo mismatches, align to start's tzinfo
        if end_time.tzinfo != start.tzinfo:
            try:
                end_time = end_time.replace(tzinfo=start.tzinfo)
            except Exception:
                pass
        return (end_time - start).total_seconds()

    @property
    def overall_progress(self) -> float:
        """Calculate overall completion percentage."""
        if self.total_instances == 0:
            return 0.0
        return (
            (self.completed_instances + self.failed_instances)
            / self.total_instances
            * 100
        )

    @property
    def cost_per_hour(self) -> float:
        """Calculate cost burn rate."""
        if self.duration_seconds == 0:
            return 0.0
        return self.total_cost / (self.duration_seconds / 3600)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total_finished = self.completed_instances + self.failed_instances
        if total_finished == 0:
            return 0.0
        return self.completed_instances / total_finished * 100

    def get_display_mode(self) -> str:
        """Determine display mode based on instance count."""
        if self.force_detail_level:
            return self.force_detail_level

        if self.total_instances <= 5:
            return "detailed"
        elif self.total_instances <= 30:
            return "compact"
        else:
            return "dense"


@dataclass
class TUIState:
    """Complete TUI state."""

    # Current run being displayed
    current_run: Optional[RunDisplay] = None

    # Event tracking (byte position before the last applied event's start)
    last_event_start_offset: int = 0
    events_processed: int = 0

    # Display state
    selected_instance_id: Optional[str] = None
    last_updated_instance_id: Optional[str] = None
    show_help: bool = False

    # Error/warning messages
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Connection state
    connected_to_orchestrator: bool = False
    last_state_poll: Optional[datetime] = None

    def add_warning(self, message: str) -> None:
        """Add a warning message (keeps last 5)."""
        self.warnings.append(message)
        if len(self.warnings) > 5:
            self.warnings.pop(0)

    def add_error(self, message: str) -> None:
        """Add an error message (keeps last 5)."""
        self.errors.append(message)
        if len(self.errors) > 5:
            self.errors.pop(0)
