"""Built-in orchestration strategies."""

from pitaya.orchestration.strategy.builtin.simple import SimpleStrategy
from pitaya.orchestration.strategy.builtin.scoring import ScoringStrategy
from pitaya.orchestration.strategy.builtin.best_of_n import BestOfNStrategy
from pitaya.orchestration.strategy.builtin.iterative import IterativeStrategy
from pitaya.orchestration.strategy.builtin.bug_finding import BugFindingStrategy
from pitaya.orchestration.strategy.builtin.doc_review import DocReviewStrategy
from pitaya.orchestration.strategy.builtin.pr_review import PRReviewStrategy

AVAILABLE_STRATEGIES = {
    "simple": SimpleStrategy,
    "scoring": ScoringStrategy,
    "best_of_n": BestOfNStrategy,
    "iterative": IterativeStrategy,
    "bug_finding": BugFindingStrategy,
    "doc_review": DocReviewStrategy,
    "pr_review": PRReviewStrategy,
}

__all__ = [
    "AVAILABLE_STRATEGIES",
    "SimpleStrategy",
    "ScoringStrategy",
    "BestOfNStrategy",
    "IterativeStrategy",
    "BugFindingStrategy",
    "DocReviewStrategy",
    "PRReviewStrategy",
]
