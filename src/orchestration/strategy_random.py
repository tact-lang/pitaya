"""Deterministic random utilities for strategies."""

from __future__ import annotations

import hashlib
from typing import List


class DeterministicRand:
    """Provide deterministic rand sequence and emit canonical events."""

    def __init__(self, orchestrator, strategy_execution_id: str) -> None:
        self._orch = orchestrator
        self._sid = strategy_execution_id
        self._seq: List[float] = []
        self._index = 0

    def rand(self) -> float:
        self._index += 1
        h = hashlib.sha256(f"{self._sid}:{self._index}".encode("utf-8")).hexdigest()
        v_int = int(h[:8], 16)
        v = (v_int % 10_000_000) / 10_000_000.0
        self._seq.append(v)
        self._emit(v)
        return v

    def _emit(self, value: float) -> None:
        try:
            if getattr(self._orch, "event_bus", None) and getattr(
                self._orch, "state_manager", None
            ):
                run_id = self._orch.state_manager.current_state.run_id
                self._orch.event_bus.emit_canonical(
                    type="strategy.rand",
                    run_id=run_id,
                    strategy_execution_id=self._sid,
                    payload={"seq": self._index, "value": value},
                )
                try:
                    self._orch.state_manager.update_strategy_rand(
                        self._sid, self._index, value
                    )
                except Exception:
                    pass
        except Exception:
            pass
