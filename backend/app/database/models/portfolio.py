from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class PortfolioSnapshot(Base):
    """Periodic mark-to-market of the copy portfolio (Module 10). `mode` is paper or live."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    mode: Mapped[str] = mapped_column(String(8), default="paper", index=True)
    cash: Mapped[Decimal] = mapped_column(Numeric)
    total_value: Mapped[Decimal] = mapped_column(Numeric)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    open_positions: Mapped[int | None] = mapped_column(BigInteger)


class PortfolioPosition(Base):
    """An open/closed position held by the copy portfolio itself (paper or live)."""

    __tablename__ = "portfolio_positions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(8), default="paper", index=True)
    market_id: Mapped[str | None] = mapped_column(ForeignKey("markets.id"), index=True)
    token_id: Mapped[str | None] = mapped_column(String(80), index=True)
    source_wallet: Mapped[str | None] = mapped_column(ForeignKey("wallets.address"), index=True)
    size: Mapped[Decimal] = mapped_column(Numeric)
    avg_entry_price: Mapped[Decimal | None] = mapped_column(Numeric)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
