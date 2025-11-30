"""Footer rendering for the TUI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.text import Text


class FooterMixin:
    """Render footer summary (metrics, paths)."""

    def _update_footer(self) -> None:
        try:
            footer_lines = []
            run_src = self._render_run
            if run_src:
                footer_lines.extend(self._build_footer_lines(run_src))
            else:
                line = Text()
                line.append("Not Connected", style="bold white")
                line.append("  •  ", style="white")
                line.append("Events: ", style="bold white")
                line.append(str(self.state.events_processed), style="bright_white")
                footer_lines.append(line)

            footer_content = Group(*footer_lines)
            try:
                self._layout["footer"].size = max(1, len(footer_lines) + 2)
            except Exception:
                pass
            self._layout["footer"].update(Panel(footer_content, style="blue"))
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            self._layout["footer"].update(Panel(f"Footer Error: {e}", style="red"))

    def _build_footer_lines(self, run_src):
        lines = []
        run = run_src
        inst_list = list(run.instances.values())
        total_tokens_in = sum(i.input_tokens for i in inst_list)
        total_tokens_out = sum(i.output_tokens for i in inst_list)
        total_tokens_cached = sum(
            getattr(i, "cached_input_tokens", 0) for i in inst_list
        )
        total_tokens = sum(i.total_tokens for i in inst_list)
        total_cost = sum(i.cost for i in inst_list)
        duration = "--:--:--"
        try:
            start_dt = self._ui_started_at
            if start_dt:
                now_dt = self._frame_now or datetime.now(timezone.utc)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                elapsed = now_dt - start_dt
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            pass
        burn = 0.0
        try:
            total_secs = (
                (self._frame_now - self._ui_started_at).total_seconds()
                if self._frame_now and self._ui_started_at
                else 0
            )
            if total_secs > 0:
                burn = total_cost / (total_secs / 3600.0)
        except Exception:
            pass

        line1 = Text()
        line1.append("Events: ", style="bold white")
        line1.append(str(self.state.events_processed), style="bright_white")
        line1.append("  •  ", style="white")
        line1.append("Tokens: ", style="bold white")
        line1.append(f"{total_tokens:,}", style="bright_white")
        line1.append(" (", style="white")
        line1.append(f"↓{total_tokens_out:,}", style="bright_white")
        line1.append(" ", style="white")
        line1.append(f"↑{total_tokens_in:,}", style="bright_white")
        if total_tokens_cached:
            line1.append(" ", style="white")
            line1.append(f"+{total_tokens_cached:,} cached", style="bright_white")
        line1.append(")  •  ", style="white")
        line1.append("Cost: ", style="bold white")
        line1.append(f"${total_cost:.4f}", style="bright_magenta")
        line1.append("  •  ", style="white")
        line1.append("Burn: ", style="bold white")
        line1.append(f"${burn:.2f}/h", style="bright_yellow")
        line1.append("  •  ", style="white")
        line1.append("Runtime: ", style="bold white")
        line1.append(duration, style="bright_yellow")
        lines.append(line1)

        try:
            if getattr(self, "_events_file", None):
                run_logs_dir = self._events_file.parent
                logs_root = run_logs_dir.parent
                results_path = logs_root.parent / "results" / run.run_id
                logs_path = str(run_logs_dir)
            else:
                logs_root = Path(".pitaya/logs")
                logs_path = str(logs_root / run.run_id)
                results_path = Path(".pitaya/results") / run.run_id
        except Exception:
            logs_path = f".pitaya/logs/{run.run_id}"
            results_path = Path(".pitaya/results") / run.run_id
        line2 = Text()
        line2.append("Logs: ", style="bold white")
        line2.append(str(logs_path), style="bright_blue")
        line2.append("  •  ", style="white")
        line2.append("Results: ", style="bold white")
        line2.append(str(results_path), style="bright_blue")
        lines.append(line2)

        return lines


__all__ = ["FooterMixin"]
