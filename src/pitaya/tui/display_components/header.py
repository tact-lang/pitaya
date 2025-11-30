"""Header rendering for the TUI."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

from ..models import InstanceStatus


class HeaderMixin:
    """Render the header zone."""

    def _update_header(self) -> None:
        try:
            run_src = self._render_run
            if not run_src:
                header_content = Text("Pitaya TUI - No Active Run", style="bold yellow")
            else:
                header_content = self._build_header_content(run_src)

            try:
                lines = max(1, header_content.plain.count("\n") + 1)
                self._layout["header"].size = max(1, lines + 2)
            except Exception:
                pass
            self._layout["header"].update(
                Panel(Align.left(header_content), style="blue")
            )
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            self._layout["header"].update(Panel(f"Header Error: {e}", style="red"))

    def _build_header_content(self, run_src):
        run = run_src
        header_content = Text()
        rows: list[Text] = []

        r1 = Text()
        r1.append("Run: ", style="bold white")
        r1.append(f"{run.run_id}", style="bold bright_cyan")
        self._append_strategy_info(r1, run)
        r1.append("  •  ", style="white")
        r1.append("Base: ", style="bold white")
        r1.append(run.base_branch or "-", style="bright_blue")
        rows.append(r1)

        rows.append(self._build_runtime_row())
        rows.append(self._build_counts_row(run))

        try:
            if getattr(self.state, "errors", None):
                err = str(self.state.errors[-1])[:160]
                rows.append(Text(f"ERR: {err}", style="red"))
        except Exception:
            pass

        for idx, row in enumerate(rows):
            if idx:
                header_content.append("\n")
            header_content.append(row)
        return header_content

    def _append_strategy_info(self, r1: Text, run) -> None:
        try:
            if run.strategies:
                first = next(iter(list(run.strategies.values())))
                params = []
                cfg = getattr(first, "config", {}) or {}
                for k in sorted(cfg.keys()):
                    v = cfg[k]
                    if isinstance(v, (str, int, float, bool)):
                        params.append(f"{k}={v}")
                strat_label = first.strategy_name
                if params:
                    strat_label += f"({','.join(params)})"
                r1.append("  •  ", style="white")
                r1.append("Strategy: ", style="bold white")
                r1.append(strat_label, style="bold green")
                model = cfg.get("model")
                if model:
                    r1.append("  •  ", style="white")
                    r1.append("Model: ", style="bold white")
                    r1.append(str(model), style="bright_magenta")
        except Exception:
            pass

    def _build_runtime_row(self) -> Text:
        r2 = Text()
        try:
            r2.append("Started: ", style="bold white")
            r2.append(self._format_time(self._ui_started_at), style="white")
            st = self._ui_started_at
            if st and st.tzinfo is None:
                st = st.replace(tzinfo=timezone.utc)
            if st:
                now_dt = self._frame_now or datetime.now(timezone.utc)
                elapsed = now_dt - st
                hours, rem = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(rem, 60)
                duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                r2.append("  •  ")
                r2.append("Runtime: ", style="bold white")
                r2.append(duration, style="bright_yellow")
        except Exception:
            pass
        return r2

    def _build_counts_row(self, run) -> Text:
        r3 = Text()
        try:
            inst_list = list(run.instances.values())
            total = len(inst_list)
            running = sum(1 for i in inst_list if i.status == InstanceStatus.RUNNING)
            queued = sum(1 for i in inst_list if i.status == InstanceStatus.QUEUED)
            done = sum(
                1
                for i in inst_list
                if i.status
                in (
                    InstanceStatus.COMPLETED,
                    InstanceStatus.FAILED,
                    InstanceStatus.INTERRUPTED,
                )
            )
            interrupted = sum(
                1 for i in inst_list if i.status == InstanceStatus.INTERRUPTED
            )
            r3.append("Tasks: ", style="bold white")
            r3.append(f"R:{running} ", style="yellow")
            r3.append(f"Q:{queued} ", style="bright_white")
            r3.append(f"D:{done} ", style="green")
            r3.append(f"I:{interrupted} ", style="magenta")
            r3.append("• ", style="white")
            r3.append("Total: ", style="bold white")
            r3.append(f"{total}", style="bright_white")
            if getattr(self, "_ids_full", False):
                r3.append("  •  ")
                r3.append("IDs: full", style="white")
        except Exception:
            pass
        return r3

    def _get_strategy_summary(self) -> str:
        if not self._render_run:
            return "-"

        strategy_names = set()
        try:
            for s in self._render_run.strategies.values():
                if s.strategy_name and s.strategy_name.lower() != "unknown":
                    strategy_names.add(s.strategy_name)
        except Exception:
            pass

        if not strategy_names:
            try:
                for inst in self._render_run.instances.values():
                    if inst.strategy_name and inst.strategy_name.lower() != "unknown":
                        strategy_names.add(inst.strategy_name)
            except Exception:
                pass

        if not strategy_names:
            return "-"

        if len(strategy_names) == 1:
            name = next(iter(strategy_names))
            exec_count = len(self._render_run.strategies) or sum(
                1
                for i in self._render_run.instances.values()
                if i.strategy_name == name
            )
            return f"{name} (x{exec_count})" if exec_count > 1 else name
        return f"Multiple ({len(strategy_names)})"


__all__ = ["HeaderMixin"]
