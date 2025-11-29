"""Mixins for event emission and failure handling in attempt execution."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .runner_utils import is_retryable_error
from ..shared import InstanceResult


class EventEmitterMixin:
    """Provides event emission helpers."""

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.params.event_callback:
            return
        try:
            if event_type.endswith("agent_result"):
                if msg := data.get("final_message"):
                    self.final_message = msg  # type: ignore[assignment]
                if isinstance(data.get("metrics"), dict):
                    self.metrics.update(data["metrics"])
        except Exception:
            pass
        self.params.event_callback(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": self.params.instance_id,
                "data": data,
            }
        )

    def _emit_phase_completed(self, phase: str) -> None:
        self._emit_event("instance.phase_completed", {"phase": phase})

    def _emit_failure(self, error_type: str, error: str, *, will_retry: bool) -> None:
        self._emit_event(
            "instance.failed",
            {
                "error": error,
                "error_type": error_type,
                "attempt": self.attempt_number,
                "total_attempts": self.total_attempts,
                "will_retry": will_retry,
            },
        )

    def _emit_canceled(self) -> None:
        self._emit_event(
            "instance.canceled",
            {
                "error": "canceled",
                "error_type": "canceled",
                "attempt": self.attempt_number,
                "total_attempts": self.total_attempts,
                "will_retry": False,
            },
        )

    def _emit_runner_completion(self, final_branch: Optional[str]) -> None:
        try:
            self._emit_event(
                "runner.instance.completed",
                {
                    "instance_id": self.params.instance_id,
                    "workspace_path": (
                        str(self.workspace_dir) if self.workspace_dir else None
                    ),
                    "branch_imported": final_branch,
                    "duration_seconds": time.time() - self.start_time,
                },
            )
        except Exception:
            pass

    def _emit_started(self) -> None:
        self._emit_event(
            "instance.started",
            {
                "prompt": self.params.prompt,
                "model": self.params.model,
                "attempt": self.attempt_number,
                "total_attempts": self.total_attempts,
                "is_retry": self.attempt_number > 1,
                "session_id": self.session_id,
                "log_path": self.log_path,
            },
        )


class FailureHandlingMixin(EventEmitterMixin):
    """Shared failure handling and cleanup helpers."""

    def _will_retry(self, error_type: str, error: str) -> bool:
        return is_retryable_error(error, error_type, self.params.retry_config)

    async def _cleanup_success(self, final_branch) -> None:
        self._emit_event(
            "instance.completed",
            {
                "success": True,
                "branch_name": final_branch,
                "duration_seconds": time.time() - self.start_time,
                "metrics": self.metrics,
            },
        )
        self._emit_runner_completion(final_branch)
        if getattr(self, "workspace_manager", None):
            await self.workspace_manager.cleanup()
        if getattr(self, "container_manager", None) and self.params.finalize:
            try:
                await self.container_manager.stop_and_cleanup(remove_home_volume=True)
                self._emit_event(
                    "instance.container_stopped",
                    {"container_name": self.params.container_name},
                )
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to cleanup container %s: %s",
                    self.params.container_name,
                    exc,
                )
        self._emit_phase_completed("cleanup_decision")

    async def _cleanup_failure(self, *, remove_home_volume: bool) -> None:
        if getattr(self, "container_manager", None):
            await self.container_manager.stop_and_cleanup(
                remove_home_volume=remove_home_volume
            )
        if getattr(self, "workspace_manager", None):
            try:
                await self.workspace_manager.cleanup()
            except Exception:
                pass

    def _failure_result(
        self, *, error: str, error_type: str, status: str
    ) -> InstanceResult:
        completed_at = datetime.now(timezone.utc).isoformat()
        return InstanceResult(
            success=False,
            error=error,
            error_type=error_type,
            session_id=self.agent_session_id,
            container_name=self.params.container_name,
            duration_seconds=time.time() - self.start_time,
            started_at=self.started_at,
            completed_at=completed_at,
            retry_attempts=self.attempt_number - 1,
            log_path=self.log_path,
            workspace_path=None,
            status=status,
            final_message=self.final_message,
            metrics=self.metrics,
        )

    async def _close_docker(self) -> None:
        if not getattr(self, "docker_manager", None):
            return
        try:
            self.docker_manager.close()
        except Exception:
            pass
