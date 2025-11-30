from typing import Any, Dict

import pytest

from pitaya.runner.attempt.mixins import FailureHandlingMixin
from pitaya.runner.runner_params import RunnerParams
from pitaya.shared import ContainerLimits, RetryConfig


class _FakeContainerMgr:
    def __init__(self) -> None:
        self.called_with: Dict[str, Any] = {}

    async def stop_and_cleanup(self, *, remove_home_volume: bool) -> None:
        self.called_with["remove_home_volume"] = remove_home_volume


class _DummyFailure(FailureHandlingMixin):
    def __init__(self, cmgr: _FakeContainerMgr) -> None:
        self.container_manager = cmgr
        self.workspace_manager = None
        self.params = RunnerParams(
            prompt="p",
            repo_path=None,  # type: ignore[arg-type]
            base_branch="main",
            branch_name="b",
            run_id=None,
            strategy_execution_id=None,
            instance_id="iid",
            container_name="c",
            model="m",
            session_id=None,
            operator_resume=False,
            session_group_key=None,
            event_callback=None,
            startup_semaphore=None,
            timeout_seconds=1,
            container_limits=ContainerLimits(),
            auth_config=None,
            reuse_container=True,
            finalize=True,
            docker_image="img",
            retry_config=RetryConfig(),
            plugin_name="claude-code",
            system_prompt=None,
            append_system_prompt=None,
            import_policy="auto",
            import_conflict_policy="fail",
            skip_empty_import=True,
            network_egress=None,
            max_turns=None,
            allow_overwrite_protected_refs=False,
            allow_global_session_volume=False,
            agent_cli_args=None,
            force_commit=False,
            workspace_include_branches=None,
            task_key=None,
        )
        self.attempt_number = 1
        self.total_attempts = 1
        self.session_id = None
        self.log_path = None
        self.workspace_dir = None
        self.start_time = 0.0
        self.started_at = "now"
        self.agent_session_id = None
        self.final_message = ""
        self.metrics = {}


@pytest.mark.asyncio
async def test_cleanup_failure_removes_home_volume() -> None:
    cmgr = _FakeContainerMgr()
    dummy = _DummyFailure(cmgr)

    await dummy._cleanup_failure(remove_home_volume=True)

    assert cmgr.called_with.get("remove_home_volume") is True
