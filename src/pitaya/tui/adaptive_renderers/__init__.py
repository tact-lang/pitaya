"""Adaptive dashboard renderers used by the TUI."""

from .base import AdaptiveBase
from .detailed import DetailedRenderer
from .compact import CompactRenderer
from .dense import DenseRenderer

__all__ = ["AdaptiveBase", "DetailedRenderer", "CompactRenderer", "DenseRenderer"]
