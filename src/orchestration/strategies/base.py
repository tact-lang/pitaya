"""
Base strategy interface for orchestrating multiple AI coding instances.

All strategies inherit from this base class and implement ``execute``.
This base provides sensible defaults so custom strategies remain minimal:

- No need to import ``logging``; use ``self.logger`` for logs
- Optional strategy name via class attribute ``NAME``
- Default config class is ``StrategyConfig``; override only if needed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import re as _re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...shared import InstanceResult

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""

    # Common configuration all strategies share
    model: str = "sonnet"

    def validate(self) -> None:
        """Validate configuration, raising ValueError if invalid."""
        pass


class Strategy(ABC):
    """
    Abstract base class for orchestration strategies.

    Strategies encapsulate patterns for coordinating multiple instances.
    They decide how many instances to run, in what order, and how to
    process the results.
    """

    # Optional class attribute to set a stable name without overriding the property
    NAME: Optional[str] = None

    def __init__(self):
        """Initialize strategy with empty config and a namespaced logger."""
        self._config_overrides: Dict[str, Any] = {}
        # Provide a per-strategy logger so authors don't need to import logging
        try:
            self.logger = logging.getLogger(f"pitaya.strategy.{self.name}")
        except Exception:  # pragma: no cover - defensive
            self.logger = logging.getLogger("pitaya.strategy")

    @property
    def name(self) -> str:
        """Return the strategy name for identification.

        Resolved in order: class attribute ``NAME`` -> derived from class name.
        Example: ``BestOfNStrategy`` -> ``best-of-n``.
        """
        if getattr(self, "NAME", None):
            return str(self.NAME)
        # Derive from class name by removing 'Strategy' and converting CamelCase to kebab-case
        cname = self.__class__.__name__
        if cname.endswith("Strategy"):
            cname = cname[: -len("Strategy")]
        parts = _re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", cname)
        derived = "-".join(p.lower() for p in parts if p)
        return derived or "custom"

    def get_config_class(self) -> type[StrategyConfig]:
        """Return the configuration class for this strategy (default: StrategyConfig)."""
        return StrategyConfig

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """
        Execute the strategy with given prompt and context.

        This is the main entry point for strategy execution. The strategy
        should spawn instances through the context, coordinate their
        execution, and return the final results.

        Args:
            prompt: The instruction for AI agents
            base_branch: Starting branch
            ctx: Strategy context providing access to instance spawning

        Returns:
            List of final results from the strategy
        """
        pass

    def create_config(self, **kwargs) -> StrategyConfig:
        """
        Create and validate configuration for this strategy.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Validated configuration instance
        """
        config_class = self.get_config_class()
        # Merge stored overrides with any provided kwargs
        all_config = {**self._config_overrides, **kwargs}
        # Filter unknown keys to keep strategies agnostic to extra fields (e.g., plugin_name)
        try:
            valid_keys = set(getattr(config_class, "__dataclass_fields__", {}).keys())
            if valid_keys:
                filtered = {k: v for k, v in all_config.items() if k in valid_keys}
            else:
                filtered = all_config
        except Exception:
            filtered = all_config
        config = config_class(**filtered)
        config.validate()
        return config

    def set_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Set configuration overrides for this strategy."""
        self._config_overrides = overrides
