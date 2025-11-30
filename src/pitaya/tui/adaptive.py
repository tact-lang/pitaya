"""
Adaptive display logic for the Pitaya TUI.

Provides the AdaptiveDisplay class, which delegates detailed/compact/dense
rendering to dedicated renderer mixins to keep modules small and readable.
"""

from .adaptive_renderers import (
    AdaptiveBase,
    CompactRenderer,
    DenseRenderer,
    DetailedRenderer,
)


class AdaptiveDisplay(
    DetailedRenderer,
    CompactRenderer,
    DenseRenderer,
    AdaptiveBase,
):
    """Handles adaptive display rendering based on instance count."""

    def __init__(self) -> None:
        super().__init__()
        self._mode_renderers = {
            "detailed": self._render_detailed,
            "compact": self._render_compact,
            "dense": self._render_dense,
        }


__all__ = ["AdaptiveDisplay"]
