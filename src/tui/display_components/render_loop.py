"""Render and polling loops for the TUI."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone


class RenderLoopMixin:
    """Main render loop plus periodic state polling."""

    async def _render_loop(self) -> None:
        iterations = 0
        while not self._shutdown:
            try:
                iterations += 1
                if iterations % 50 == 0:
                    self._logger.debug("Render loop iteration %s", iterations)

                t0 = time.perf_counter()
                now_tick = int(time.time())
                events_changed = (
                    getattr(self.state, "events_processed", -1)
                    != self._last_render_events
                )
                tick_changed = now_tick != self._last_runtime_tick
                sel_changed = (
                    getattr(self.state, "last_updated_instance_id", None)
                    != self._last_updated_iid
                )
                if not (events_changed or tick_changed or sel_changed):
                    await asyncio.sleep(self.refresh_rate)
                    continue

                self._frame_now = datetime.now(timezone.utc)
                self._snapshot_state_for_render()
                t_snap = time.perf_counter()

                self._update_header()
                t_head = time.perf_counter()
                self._update_dashboard()
                t_body = time.perf_counter()
                self._update_footer()
                t_foot = time.perf_counter()

                if self._live:
                    self._live.update(self._layout)
                    try:
                        self._live.refresh()
                    except Exception:
                        pass

                self._last_render_events = getattr(self.state, "events_processed", -1)
                self._last_runtime_tick = now_tick
                self._last_updated_iid = getattr(
                    self.state, "last_updated_instance_id", None
                )

                try:
                    stats = self.event_stream.get_stats()
                except Exception:
                    stats = {}

                self._logger.debug(
                    "frame_ms=%.2f snap_ms=%.2f head_ms=%.2f body_ms=%.2f foot_ms=%.2f mode=%s instances=%d events_processed=%d last_off=%d qsize=%s enq=%s proc=%s rot=%s trunc=%s pos=%s size=%s",
                    (time.perf_counter() - t0) * 1000.0,
                    (t_snap - t0) * 1000.0,
                    (t_head - t_snap) * 1000.0 if t_head and t_snap else 0.0,
                    (t_body - t_head) * 1000.0 if t_body and t_head else 0.0,
                    (t_foot - t_body) * 1000.0 if t_foot and t_body else 0.0,
                    self._render_run.get_display_mode() if self._render_run else "-",
                    len(self._render_run.instances) if self._render_run else 0,
                    self.state.events_processed,
                    self.state.last_event_start_offset,
                    stats.get("queue_size", -1),
                    stats.get("lines_enqueued", -1),
                    stats.get("lines_processed", -1),
                    stats.get("rotations", -1),
                    stats.get("truncations", -1),
                    stats.get("last_position", -1),
                    stats.get("prev_size", -1),
                )

                await asyncio.sleep(self.refresh_rate)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                raise
            except OSError as e:
                self.state.add_error(f"Render error: {e}")
                self._logger.error("Render error: %s", e, exc_info=True)
            except Exception as e:
                self.state.add_error(f"Render loop error: {type(e).__name__}: {e}")
                self._logger.error("Render loop exception", exc_info=True)

    async def _state_polling_loop(self) -> None:
        while not self._shutdown:
            try:
                if self._orchestrator and hasattr(
                    self._orchestrator, "get_current_state"
                ):
                    state = self._orchestrator.get_current_state()
                    if state:
                        self._reconcile_state(state)

                if (
                    self.state.current_run
                    and not self.state.current_run.force_detail_level
                ):
                    mode = (
                        (self._force_display_mode_cli or "").strip().lower()
                        if self._force_display_mode_cli
                        else None
                    )
                    if mode in ("detailed", "compact", "dense"):
                        self.state.current_run.force_detail_level = mode

                self.state.last_state_poll = datetime.now()
                await asyncio.sleep(self.state_poll_interval)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                raise
            except (OSError, AttributeError) as e:
                self.state.add_error(f"State poll error: {e}")


__all__ = ["RenderLoopMixin"]
