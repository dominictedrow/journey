from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class Ranking(Base):
    """A ranking generation entry (Module 5). Categories: overall/politics/crypto/sports/
    economics/hft/long_term/early_info/low_risk. API serves the latest computed_at generation."""

    __tablename__ = "rankings"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    wallet_address: Mapped[str] = mapped_column(ForeignKey("wallets.address"), index=True)
    score: Mapped[Decimal] = mapped_column(Numeric)
