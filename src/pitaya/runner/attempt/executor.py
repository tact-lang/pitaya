"""Execute a single instance attempt (no retries)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pitaya.exceptions import (
    AgentError,
    DockerError,
    GitError,
    TimeoutError,
    ValidationError,
)
from pitaya.runner.agent_runner import AgentRunner
from pitaya.runner.docker.container_manager import ContainerManager
from pitaya.runner.workspace.git_operations import GitOperations
from pitaya.runner.workspace.import_manager import ImportManager
from pitaya.runner.attempt.mixins import FailureHandlingMixin
from pitaya.runner.attempt.timeout import TimeoutCleanupMixin
from pitaya.runner.runner_params import RunnerParams
from pitaya.runner.workspace.manager import WorkspaceManager
from pitaya.shared import InstanceResult
from pitaya.runner.docker.manager import DockerManager

logger = logging.getLogger(__name__)


class AttemptExecutor(FailureHandlingMixin, TimeoutCleanupMixin):
    """Encapsulates a single attempt, including all phases and cleanup."""

    def __init__(
        self,
        *,
        params: RunnerParams,
        plugin,
        resolved_model_id: Optional[str],
        attempt_number: int,
        total_attempts: int,
        reuse_container: bool,
        session_id: Optional[str],
    ) -> None:
        self.params = params
        self.plugin = plugin
        self.resolved_model_id = resolved_model_id or params.model
        self.attempt_number = attempt_number
        self.total_attempts = total_attempts
        self.reuse_container = reuse_container
        self.session_id = session_id

        self.git_ops = GitOperations()
        self.workspace_dir = None
        self.docker_manager: Optional[DockerManager] = None
        self.container = None
        self.agent_session_id: Optional[str] = None
        self.final_message = ""
        self.metrics: Dict[str, Any] = {}
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.start_time = time.time()
        self.log_path = (
            f"./logs/{params.run_id}/instance_{params.instance_id}.log"
            if params.run_id and params.instance_id
            else None
        )

    async def run(self) -> InstanceResult:
        try:
            self._emit_started()
            self._validate_inputs()
            await self._init_docker()
            await self._prepare_workspace()
            await self._create_container()
            await self._execute_agent()
            result = await self._collect_and_import()
            await self._cleanup_success(result.branch_name)
            return result
        except TimeoutError as exc:
            return await self._handle_timeout(exc)
        except (DockerError, GitError, AgentError, ValidationError) as exc:
            return await self._handle_known_failure(exc)
        except asyncio.CancelledError:
            return await self._handle_cancel()
        except (OSError, IOError) as exc:
            return await self._handle_system_failure(exc)
        finally:
            await self._close_docker()

    def _validate_inputs(self) -> None:
        if not self.params.repo_path.exists():
            raise ValidationError(
                f"Repository path does not exist: {self.params.repo_path}"
            )
        if not self.params.repo_path.is_dir():
            raise ValidationError(
                f"Repository path is not a directory: {self.params.repo_path}"
            )
        if not (self.params.repo_path / ".git").exists():
            raise ValidationError(f"Not a git repository: {self.params.repo_path}")
        if not self.params.container_name:
            raise ValidationError("Container name must be provided by orchestration")
        if not self.params.branch_name:
            raise ValidationError("Branch name must be provided by orchestration")

    async def _init_docker(self) -> None:
        self.docker_manager = DockerManager()
        await self.docker_manager.initialize()
        logger.info("Validating Docker environment...")
        if not await self.docker_manager.validate_environment():
            raise DockerError("Docker daemon is not accessible")
        logger.info("Docker validation successful")
        self._emit_phase_completed("validation")

    async def _prepare_workspace(self) -> None:
        manager = WorkspaceManager(
            git_ops=self.git_ops,
            params=self.params,
            emit_event=self._emit_event,
            session_id=self.session_id,
        )
        self.workspace_dir = await manager.prepare()
        self.workspace_manager = manager

    async def _create_container(self) -> None:
        assert self.workspace_dir and self.docker_manager
        container_mgr = ContainerManager(
            docker_manager=self.docker_manager,
            plugin=self.plugin,
            params=self.params,
            emit_event=self._emit_event,
            session_id=self.session_id,
            reuse_container=self.reuse_container,
        )
        await container_mgr.prepare(self.workspace_dir)
        self.container_manager = container_mgr
        self.container = container_mgr.container

    async def _execute_agent(self) -> None:
        runner = AgentRunner(
            plugin=self.plugin,
            params=self.params,
            emit_event=self._emit_event,
            resolved_model_id=self.resolved_model_id,
            session_id=self.session_id,
            log_path=self.log_path,
            container=self.container,
            docker_manager=self.docker_manager,
            env_vars=getattr(self.container_manager, "env_vars", None),
        )
        result_data = await runner.run()
        self.agent_session_id = result_data.get("session_id")
        self.final_message = result_data.get("final_message", "")
        self.metrics = result_data.get("metrics", {}) or {}
        self._emit_event(
            "instance.agent_completed",
            {
                "session_id": self.agent_session_id,
                "metrics": self.metrics,
            },
        )
        self._emit_phase_completed("agent_execution")

    async def _collect_and_import(self) -> InstanceResult:
        logger.info("Collecting results and importing branch")
        self._emit_event("instance.result_collection_started", {})
        import_mgr = ImportManager(
            git_ops=self.git_ops,
            params=self.params,
            workspace_dir=self.workspace_dir,
            emit_event=self._emit_event,
        )
        await import_mgr.force_commit_if_needed()
        import_info = await import_mgr.import_branch()
        has_changes = str(import_info.get("has_changes", "false")).lower() == "true"
        final_branch = (
            import_info.get("target_branch") or self.params.branch_name
            if has_changes
            else import_info.get("target_branch")
        )
        import_mgr.emit_after_import(import_info, final_branch, has_changes)
        duration = time.time() - self.start_time
        commit_statistics = await import_mgr.gather_stats(has_changes)
        completed_at = datetime.now(timezone.utc).isoformat()
        return InstanceResult(
            success=True,
            branch_name=final_branch,
            has_changes=has_changes,
            final_message=self.final_message,
            session_id=self.agent_session_id,
            container_name=self.params.container_name,
            metrics=self.metrics,
            duration_seconds=duration,
            commit_statistics=commit_statistics,
            started_at=self.started_at,
            completed_at=completed_at,
            retry_attempts=self.attempt_number - 1,
            log_path=self.log_path,
            workspace_path=(
                str(self.workspace_dir)
                if self.workspace_dir and not has_changes
                else None
            ),
            status="success",
            commit=import_info.get("commit"),
            duplicate_of_branch=import_info.get("duplicate_of_branch"),
            dedupe_reason=import_info.get("dedupe_reason"),
        )

    async def _handle_timeout(self, exc: TimeoutError) -> InstanceResult:
        logger.error("Instance %s timed out: %s", self.params.instance_id, exc)
        self._emit_failure(
            "timeout", str(exc), will_retry=self._will_retry("timeout", str(exc))
        )
        await self._cleanup_timeout()
        completed_at = datetime.now(timezone.utc).isoformat()
        return InstanceResult(
            success=False,
            error=str(exc),
            error_type="timeout",
            session_id=self.agent_session_id,
            container_name=self.params.container_name,
            duration_seconds=time.time() - self.start_time,
            started_at=self.started_at,
            completed_at=completed_at,
            retry_attempts=self.attempt_number - 1,
            log_path=self.log_path,
            workspace_path=str(self.workspace_dir) if self.workspace_dir else None,
            status="timeout",
            final_message=self.final_message,
            metrics=self.metrics,
        )

    async def _handle_known_failure(self, exc: Exception) -> InstanceResult:
        err_type = type(exc).__name__.lower().replace("error", "")
        logger.error(
            "Instance %s failed: %s: %s",
            self.params.instance_id,
            type(exc).__name__,
            exc,
        )
        self._emit_failure(
            err_type, str(exc), will_retry=self._will_retry(err_type, str(exc))
        )
        await self._cleanup_failure(remove_home_volume=True)
        return self._failure_result(
            error=str(exc), error_type=err_type, status="failed"
        )

    async def _handle_cancel(self) -> InstanceResult:
        logger.info(
            "Instance %s canceled; recording interruption", self.params.instance_id
        )
        self._emit_canceled()
        if self.docker_manager and self.container:
            try:
                await self.docker_manager.stop_heartbeat(self.container)
            except Exception:
                pass
        completed_at = datetime.now(timezone.utc).isoformat()
        return InstanceResult(
            success=False,
            error="canceled",
            error_type="canceled",
            session_id=self.agent_session_id,
            container_name=self.params.container_name,
            duration_seconds=time.time() - self.start_time,
            started_at=self.started_at,
            completed_at=completed_at,
            retry_attempts=self.attempt_number - 1,
            log_path=self.log_path,
            workspace_path=str(self.workspace_dir) if self.workspace_dir else None,
            status="canceled",
            final_message=self.final_message,
            metrics=self.metrics,
        )

    async def _handle_system_failure(self, exc: Exception) -> InstanceResult:
        logger.exception("System error in instance %s", self.params.instance_id)
        self._emit_failure("system", str(exc), will_retry=False)
        await self._cleanup_failure(remove_home_volume=True)
        return self._failure_result(
            error=f"Unexpected error: {str(exc)}",
            error_type="unexpected",
            status="failed",
        )
