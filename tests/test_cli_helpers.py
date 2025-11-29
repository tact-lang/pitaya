from __future__ import annotations

from argparse import Namespace
from io import StringIO

from rich.console import Console

from src.orchestration.cli import headless
from src.orchestration.cli.runner_setup import apply_resume_overrides
from src.orchestration.cli.results_display import (
    Totals,
    _coalesce_totals,
    _totals_from_results,
)


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False, color_system=None)


def test_exit_code_ignores_canceled_instances() -> None:
    class Result:
        def __init__(self, success: bool, status: str):
            self.success = success
            self.status = status

    results = [
        Result(success=True, status="completed"),
        Result(success=False, status="canceled"),
    ]
    assert headless._exit_code(results) == 0


def test_exit_code_detects_failures() -> None:
    class Result:
        def __init__(self, success: bool, status: str):
            self.success = success
            self.status = status

    results = [Result(success=False, status="failed")]
    assert headless._exit_code(results) == 3


def test_apply_resume_overrides_respects_safe_and_unsafe_paths() -> None:
    args = Namespace(
        resume="run_1234",
        max_parallel=None,
        max_startup_parallel=None,
        randomize_queue=False,
        timeout=123,
        force_commit=False,
        docker_image=None,
        model="new-model",
        plugin=None,
        oauth_token=None,
        api_key=None,
        base_url=None,
        override_config=False,
        resume_key_policy="strict",
    )
    full_config = {"runner": {"timeout": 3600}, "model": "base", "orchestration": {}}

    updated = apply_resume_overrides(full_config, _console(), args)
    assert updated["runner"]["timeout"] == 123  # safe override applied
    assert updated["model"] == "base"  # unsafe override ignored without flag

    args.override_config = True
    args.resume_key_policy = "suffix"
    updated_forced = apply_resume_overrides(full_config, _console(), args)
    assert updated_forced["model"] == "new-model"
    assert updated_forced["orchestration"].get("resume_key_suffix", "").startswith("r")


def test_coalesce_totals_prefers_state_when_present() -> None:
    class Result:
        def __init__(
            self, duration: int, cost: float, tokens: int, success: bool, status: str
        ):
            self.duration_seconds = duration
            self.metrics = {"total_cost": cost, "total_tokens": tokens}
            self.success = success
            self.status = status

    results = [Result(10, 5.0, 100, True, "completed")]
    fallback = _totals_from_results(results)
    state_totals = Totals(
        duration_seconds=50.0,
        total_cost=12.0,
        total_tokens=200,
        success_count=2,
        canceled_count=1,
        failed_count=0,
        total_count=3,
    )
    merged = _coalesce_totals(state_totals, fallback)

    assert merged.duration_seconds == 50.0
    assert merged.total_cost == 12.0
    assert merged.total_tokens == 200
    assert merged.total_count == 3

    merged_fallback = _coalesce_totals(None, fallback)
    assert merged_fallback == fallback
