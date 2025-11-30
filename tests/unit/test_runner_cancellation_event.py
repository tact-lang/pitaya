from pathlib import Path
from typing import Any, Dict, List

from pitaya.runner.attempt.mixins import FailureHandlingMixin
from pitaya.runner.runner_params import RunnerParams
from pitaya.shared import ContainerLimits, RetryConfig


class _Dummy(FailureHandlingMixin):
    def __init__(self, callback_list: List[Dict[str, Any]]) -> None:
        self.params = RunnerParams(
            prompt="p",
            repo_path=Path("/tmp"),
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
            event_callback=lambda ev: callback_list.append(ev),
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
            logs_dir=Path(".pitaya/logs"),
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


def test_emit_canceled_event() -> None:
    events: list[dict[str, Any]] = []
    dummy = _Dummy(events)

    dummy._emit_canceled()

    assert any(ev["type"] == "instance.canceled" for ev in events)
    assert not any(ev["type"] == "instance.failed" for ev in events)
