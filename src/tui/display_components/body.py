"""Dashboard and details rendering for the TUI."""

from __future__ import annotations

from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class BodyMixin:
    """Render the main dashboard and optional details pane."""

    def _update_dashboard(self) -> None:
        try:
            run_src = self._render_run
            if not run_src:
                target = self._layout["body"]
                if self._body_split and self._details_mode == "none":
                    self._layout["body"] = Layout(name="body", ratio=1)
                    self._body_split = False
                    target = self._layout["body"]
                target.update(
                    Panel(
                        Align.center(Text("Waiting for Pitaya run...", style="dim")),
                        style="dim",
                    )
                )
                return

            display_mode = run_src.get_display_mode()
            dashboard_content = self.adaptive_display.render_dashboard(
                run_src, display_mode, frame_now=getattr(self, "_frame_now", None)
            )

            if self._details_mode == "right":
                if not self._body_split:
                    body = self._layout["body"]
                    body.split_row(
                        Layout(name="dashboard", ratio=3),
                        Layout(name="details", size=48),
                    )
                    self._body_split = True
                self._layout["body"]["dashboard"].update(dashboard_content)
                self._layout["body"]["details"].update(self._render_details_panel())
            else:
                self._layout["body"].update(dashboard_content)
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            self._layout["body"].update(Panel(f"Dashboard Error: {e}", style="red"))

    def _render_details_panel(self):
        try:
            run = self._render_run
            if not run or not getattr(self, "state", None):
                return Panel("No details", style="dim")
            iid = getattr(self.state, "last_updated_instance_id", None)
            inst = run.instances.get(iid) if iid else None
            if not inst:
                return Panel("No task selected", style="dim")

            tbl = Table(show_header=False, box=None, pad_edge=False, show_edge=False)
            tbl.add_row("Instance:", (inst.instance_id or "")[:16])
            if inst.branch_name:
                tbl.add_row("Branch:", inst.branch_name)
            if inst.current_activity:
                tbl.add_row("Activity:", inst.current_activity)
            if inst.error:
                tbl.add_row("Error:", inst.error)
            if inst.duration_seconds:
                tbl.add_row("Time:", self._format_duration(inst.duration_seconds))
            if inst.total_tokens:
                token_str = f"{inst.total_tokens:,} (↓{inst.output_tokens:,} ↑{inst.input_tokens:,}"
                if getattr(inst, "cached_input_tokens", 0):
                    token_str += f" +{inst.cached_input_tokens:,} cached"
                token_str += ")"
                tbl.add_row("Tokens:", token_str)
            if inst.cost:
                tbl.add_row("Cost:", f"${inst.cost:.2f}")
            try:
                if getattr(inst, "final_message_path", None):
                    note = (
                        " (truncated)"
                        if getattr(inst, "final_message_truncated", False)
                        else ""
                    )
                    tbl.add_row("Final:", f"{inst.final_message_path}{note}")
                elif getattr(inst, "final_message", None):
                    preview = inst.final_message.strip().replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:197] + "..."
                    note = (
                        " (truncated)"
                        if getattr(inst, "final_message_truncated", False)
                        else ""
                    )
                    tbl.add_row("Final:", f"{preview}{note}")
            except Exception:
                pass
            try:
                msgs = getattr(self.event_processor, "_messages", {}).get(
                    inst.instance_id, []
                )
                if msgs:
                    tbl.add_row("Messages:", "\n".join(f"• {m}" for m in msgs))
            except Exception:
                pass
            return Panel(tbl, title="Details", border_style="blue")
        except Exception as e:
            return Panel(f"Details error: {e}", style="red")


__all__ = ["BodyMixin"]
