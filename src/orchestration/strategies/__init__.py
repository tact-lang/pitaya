"""
Orchestration strategies for coordinating multiple AI coding instances.
"""

from .base import Strategy, StrategyConfig
from .simple import SimpleStrategy
from .scoring import ScoringStrategy, ScoringConfig
from .best_of_n import BestOfNStrategy, BestOfNConfig
from .iterative import IterativeStrategy, IterativeConfig
from .bug_finding import BugFindingStrategy, BugFindingConfig
from .doc_review import DocReviewStrategy, DocReviewConfig
from .pr_review import PRReviewStrategy, PRReviewConfig

# Registry of available strategies
AVAILABLE_STRATEGIES = {
    "simple": SimpleStrategy,
    "scoring": ScoringStrategy,
    "best-of-n": BestOfNStrategy,
    "iterative": IterativeStrategy,
    "bug-finding": BugFindingStrategy,
    "doc-review": DocReviewStrategy,
    "pr-review": PRReviewStrategy,
}

__all__ = [
    "Strategy",
    "StrategyConfig",
    "SimpleStrategy",
    "ScoringStrategy",
    "ScoringConfig",
    "BestOfNStrategy",
    "BestOfNConfig",
    "IterativeStrategy",
    "IterativeConfig",
    "BugFindingStrategy",
    "BugFindingConfig",
    "DocReviewStrategy",
    "DocReviewConfig",
    "PRReviewStrategy",
    "PRReviewConfig",
    "AVAILABLE_STRATEGIES",
]
