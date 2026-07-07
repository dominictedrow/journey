"""Pydantic models for /copy: copy-order feed, simulate input, run/simulate results."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class CopyOrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    mode: str
    source_wallet: str | None
    source_trade_id: int | None
    market_id: str | None
    token_id: str | None
    side: str
    requested_size: Decimal
    filled_size: Decimal | None
    requested_price: Decimal | None
    fill_price: Decimal | None
    delay_ms: int | None
    slippage: Decimal | None
    status: str
    error: str | None
    external_order_id: str | None = None


class CopyOrderListOut(BaseModel):
    items: list[CopyOrderOut]
    total: int
    limit: int
    offset: int


class CopyStatusOut(BaseModel):
    mode: str
    total_orders: int
    by_status: dict[str, int]
    last_order_at: datetime | None


class SimulateTradeIn(BaseModel):
    """Inject a synthetic source fill and run it through detect→risk→paper-execute→portfolio."""

    source_wallet: str = Field(min_length=1)
    token_id: str = Field(min_length=1)
    side: str = Field(pattern="^(BUY|SELL)$")
    size: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0, lt=1)
    market_id: str | None = None
    usdc_amount: Decimal | None = None


class SimulateResultOut(BaseModel):
    order: CopyOrderOut
    detail: str


class TaskEnqueuedOut(BaseModel):
    task_id: str
