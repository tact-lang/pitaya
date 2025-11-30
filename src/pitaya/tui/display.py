"""
Rich-based TUI display for Pitaya.

TUIDisplay is composed from mixins in :mod:`tui.display_components`
to keep modules small and maintainable. The public API remains
`from tui.display import TUIDisplay`.
"""

from .display_main import TUIDisplay

__all__ = ["TUIDisplay"]
