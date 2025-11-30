"""Handlers for run.* and strategy.* events."""

from __future__ import annotations

from typing import Any, Dict

from ..models import RunDisplay, StrategyDisplay
from .logging_config import logger


class RunStrategyEventHandlers:
    """Run and strategy lifecycle handlers."""

    def _handle_run_started(self, event: Dict[str, Any]) -> None:
        """Handle run.started event."""
        data = event.get("data", {})

        self.state.current_run = RunDisplay(
            run_id=data.get("run_id", "unknown"),
            prompt=data.get("prompt", ""),
            repo_path=data.get("repo_path", ""),
            base_branch=data.get("base_branch", "main"),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.connected_to_orchestrator = True

    def _handle_run_completed(self, event: Dict[str, Any]) -> None:
        """Handle run.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        run = self.state.current_run

        run.completed_at = self._parse_timestamp(event.get("timestamp"))
        run.total_cost = data.get("total_cost", 0.0)
        run.total_tokens = data.get("total_tokens", 0)

    def _handle_run_failed(self, event: Dict[str, Any]) -> None:
        """Handle run.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        self.state.add_error(f"Run failed: {data.get('error', 'Unknown error')}")

    def _handle_strategy_started(self, event: Dict[str, Any]) -> None:
        """Handle strategy.started event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        # Canonical envelope carries the strategy execution id
        strategy_id = event.get("strategy_execution_id") or data.get("strategy_id")
        if not strategy_id:
            return

        strategy = StrategyDisplay(
            strategy_id=strategy_id,
            strategy_name=data.get("name", data.get("strategy_name", "unknown")),
            config=data.get("params", data.get("config", {})),
            started_at=self._parse_timestamp(event.get("timestamp")),
        )

        self.state.current_run.strategies[strategy_id] = strategy
        # Prime run start time on first strategy start
        try:
            if not self.state.current_run.started_at:
                self.state.current_run.started_at = strategy.started_at
        except Exception:
            pass
        # Log with the canonical 'name' when present
        logger.info(
            f"Created strategy {strategy_id} ({strategy.strategy_name or data.get('name','unknown')})"
        )

    def _handle_strategy_completed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.completed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = event.get("strategy_execution_id") or data.get("strategy_id")
        if not strategy_id or strategy_id not in self.state.current_run.strategies:
            return

        strategy = self.state.current_run.strategies[strategy_id]
        strategy.completed_at = self._parse_timestamp(event.get("timestamp"))
        strategy.is_complete = True

    def _handle_strategy_failed(self, event: Dict[str, Any]) -> None:
        """Handle strategy.failed event."""
        if not self.state.current_run:
            return

        data = event.get("data", {})
        strategy_id = data.get("strategy_id")
        if strategy_id and strategy_id in self.state.current_run.strategies:
            strategy = self.state.current_run.strategies[strategy_id]
            strategy.is_complete = True
            self.state.add_error(
                f"Strategy {strategy.strategy_name} failed: {data.get('error', 'Unknown')}"
            )


__all__ = ["RunStrategyEventHandlers"]
