"""Pitaya public API surface.

This package intentionally exposes only the stable entry points needed by
consumers; everything else should be considered internal and may change.
"""

from .orchestration.orchestrator import Orchestrator
from .runner.api import run_instance
from .shared.results import InstanceResult
from .version import __version__

__all__ = ["Orchestrator", "run_instance", "InstanceResult", "__version__"]
