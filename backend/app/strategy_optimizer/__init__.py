"""Module 6 — strategy optimizer: backtest candidate strategies and adopt the best (Phase 6)."""

from app.strategy_optimizer.models import (
    CandidateResult,
    OptimizeResult,
    StrategySpec,
    Weighting,
)
from app.strategy_optimizer.optimizer import optimize

__all__ = [
    "CandidateResult",
    "OptimizeResult",
    "StrategySpec",
    "Weighting",
    "optimize",
]
