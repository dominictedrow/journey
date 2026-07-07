"""Pydantic response models for /strategies (Module 6).

List/detail mirror the `strategies` table: `params` is the re-runnable `StrategySpec`, and
`rationale` (detail only) is the optimizer's transparency record for the run that produced it.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = [
    "StrategyOut",
    "StrategyDetailOut",
    "StrategyListOut",
    "OptimizeQueuedOut",
]


class StrategyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_active: bool
    params: dict
    created_at: datetime
    adopted_at: datetime | None


class StrategyDetailOut(StrategyOut):
    rationale: dict | None


class StrategyListOut(BaseModel):
    items: list[StrategyOut]
    total: int
    limit: int
    offset: int


class OptimizeQueuedOut(BaseModel):
    task_id: str
    message: str
