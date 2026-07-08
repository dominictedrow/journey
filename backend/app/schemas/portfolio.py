"""Pydantic models for /portfolio: positions, snapshots (equity curve), and the summary."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class PortfolioPositionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    mode: str
    market_id: str | None
    token_id: str | None
    source_wallet: str | None
    size: Decimal
    avg_entry_price: Decimal | None
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    is_open: bool
    opened_at: datetime | None
    closed_at: datetime | None


class PortfolioSnapshotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    mode: str
    cash: Decimal
    total_value: Decimal
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    open_positions: int | None


class PortfolioSummaryOut(BaseModel):
    mode: str
    cash: Decimal
    total_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    open_positions: int
    last_snapshot_at: datetime | None
    positions: list[PortfolioPositionOut]


class PortfolioHistoryOut(BaseModel):
    mode: str
    items: list[PortfolioSnapshotOut]


class ClosePositionOut(BaseModel):
    """Result of a manual close: the execution outcome and the position's final state."""

    status: str  # filled | partial | rejected
    filled_size: Decimal
    fill_price: Decimal | None
    error: str | None
    position: PortfolioPositionOut
