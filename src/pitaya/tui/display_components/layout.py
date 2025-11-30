"""Layout creation for the TUI."""

from __future__ import annotations

from rich.layout import Layout
from rich.panel import Panel


class LayoutMixin:
    """Provides the base three-zone layout."""

    def _create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["header"].update(Panel("Initializing...", style="blue"))
        layout["body"].update(Panel("Loading...", style="dim"))
        layout["footer"].update(Panel("Starting...", style="blue"))
        return layout


__all__ = ["LayoutMixin"]
