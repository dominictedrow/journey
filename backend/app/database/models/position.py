from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class Position(Base):
    """A source wallet's per-token position, derived from its trades."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(ForeignKey("wallets.address"), index=True)
    market_id: Mapped[str | None] = mapped_column(ForeignKey("markets.id"), index=True)
    token_id: Mapped[str | None] = mapped_column(String(80), index=True)
    net_size: Mapped[Decimal] = mapped_column(Numeric)
    avg_entry_price: Mapped[Decimal | None] = mapped_column(Numeric)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
