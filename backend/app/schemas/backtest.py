"""Pydantic request/response models for /backtests.

The create body IS `BacktestConfig` (validated once, stored verbatim in `params`).
List items omit the potentially large `equity_curve`/`trade_log`; the detail view
carries them.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.backtesting.models import BacktestConfig

__all__ = ["BacktestConfig", "BacktestOut", "BacktestDetailOut", "BacktestListOut"]


class BacktestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    params: dict
    started_at: datetime | None
    finished_at: datetime | None
    metrics: dict | None


class BacktestDetailOut(BacktestOut):
    equity_curve: list | None
    trade_log: list | None


class BacktestListOut(BaseModel):
    items: list[BacktestOut]
    total: int
    limit: int
    offset: int
