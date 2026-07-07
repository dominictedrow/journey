"""Pydantic response/request models for /wallets and /discovery.

Decimal fields serialize to JSON strings (Pydantic v2 default) — the frontend parses them
for display, which keeps money values exact on the wire.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class WalletMetricsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    computed_at: datetime
    roi_lifetime: Decimal | None
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    win_rate: Decimal | None
    avg_trade_return: Decimal | None
    profit_factor: Decimal | None
    cagr: Decimal | None
    roi_30d: Decimal | None
    roi_90d: Decimal | None
    monthly_consistency: Decimal | None
    weekly_consistency: Decimal | None
    max_drawdown: Decimal | None
    losing_streak_max: int | None
    volatility: Decimal | None
    concentration: Decimal | None
    sharpe: Decimal | None
    sortino: Decimal | None
    avg_hold_hours: Decimal | None
    trade_frequency: Decimal | None
    avg_position_size: Decimal | None
    max_concurrent_positions: int | None
    category_mix: dict | None
    early_entry_score: Decimal | None
    exit_timing_score: Decimal | None
    avg_prob_improvement: Decimal | None
    sample_size: int | None


class WalletOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    address: str
    first_seen: datetime | None
    last_active: datetime | None
    discovery_source: str | None
    discovered_at: datetime | None
    total_volume: Decimal | None
    total_trades: int | None
    is_tracked: bool
    copy_enabled: bool
    disabled_reason: str | None


class WalletWithMetrics(WalletOut):
    metrics: WalletMetricsOut | None = None


class WalletListOut(BaseModel):
    items: list[WalletWithMetrics]
    total: int
    limit: int
    offset: int


class TradeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    market_id: str | None
    token_id: str | None
    side: str
    price: Decimal
    size: Decimal
    usdc_amount: Decimal | None
    timestamp: datetime
    tx_hash: str


class PositionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    market_id: str | None
    token_id: str | None
    net_size: Decimal
    avg_entry_price: Decimal | None
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    is_open: bool
    opened_at: datetime | None
    closed_at: datetime | None


class WalletDetailOut(WalletWithMetrics):
    open_positions: list[PositionOut]
    recent_trades: list[TradeOut]


class WalletUpdate(BaseModel):
    is_tracked: bool | None = None
    copy_enabled: bool | None = None


class DiscoverySourceCount(BaseModel):
    source: str | None  # None = auto-created by the chain indexer, no attribution
    count: int


class DiscoveryStatusOut(BaseModel):
    total_wallets: int
    tracked_wallets: int
    last_discovered_at: datetime | None
    by_source: list[DiscoverySourceCount]


class TaskEnqueuedOut(BaseModel):
    task_id: str
