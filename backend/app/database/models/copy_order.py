from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class CopyOrder(Base):
    """A copy-trade execution record (Module 9), paper or live. Captures requested vs filled
    plus execution quality (delay, slippage) used by the copy-quality score component."""

    __tablename__ = "copy_orders"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    mode: Mapped[str] = mapped_column(String(8), default="paper", index=True)
    source_wallet: Mapped[str | None] = mapped_column(ForeignKey("wallets.address"), index=True)
    source_trade_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"))
    market_id: Mapped[str | None] = mapped_column(ForeignKey("markets.id"), index=True)
    token_id: Mapped[str | None] = mapped_column(String(80))
    side: Mapped[str] = mapped_column(String(4))  # BUY / SELL
    requested_size: Mapped[Decimal] = mapped_column(Numeric)
    filled_size: Mapped[Decimal | None] = mapped_column(Numeric)
    requested_price: Mapped[Decimal | None] = mapped_column(Numeric)
    fill_price: Mapped[Decimal | None] = mapped_column(Numeric)
    delay_ms: Mapped[int | None] = mapped_column(BigInteger)
    slippage: Mapped[Decimal | None] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    # CLOB order id for a live fill; anchors post-crash reconciliation (None for paper orders).
    external_order_id: Mapped[str | None] = mapped_column(String(80), index=True)
