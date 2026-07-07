"""Executor interface (Module 9).

An `Executor` turns a risk-approved size into a fill. `PaperExecutor` (Phase 5) simulates
against the live order book; `LiveExecutor` (Phase 8) submits real orders. Both return the
same `ExecutionResult`, so the copy engine is executor-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from app.risk.models import TradeIntent


@dataclass
class ExecutionResult:
    status: str  # filled | partial | rejected
    filled_size: Decimal
    fill_price: Decimal | None
    slippage: Decimal | None  # signed fraction vs the intent's reference price (adverse = +)
    delay_ms: int | None
    error: str | None = None
    external_order_id: str | None = None  # CLOB order id for a live fill (None for paper)


class Executor(ABC):
    mode: str

    @abstractmethod
    async def execute(self, intent: TradeIntent, size: Decimal) -> ExecutionResult:
        """Fill up to `size` shares of `intent`; `size` is the risk-adjusted quantity."""
