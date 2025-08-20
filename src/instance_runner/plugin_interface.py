"""
Runner Plugin Interface for supporting multiple AI coding tools.

This module defines the abstract interface that all AI tool plugins must implement,
as specified in section 3.8 of the Pitaya specification.
"""

# Re-export from shared types to maintain compatibility
from ..shared import PluginCapabilities, RunnerPlugin

__all__ = ["PluginCapabilities", "RunnerPlugin"]
