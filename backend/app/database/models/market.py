from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, JsonB


class Market(Base):
    """A Polymarket market, keyed by CTF condition_id (Gamma API sync)."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)  # condition_id
    question: Mapped[str] = mapped_column(Text)
    slug: Mapped[str | None] = mapped_column(String(255), index=True)
    category: Mapped[str | None] = mapped_column(String(64), index=True)
    outcomes: Mapped[list | None] = mapped_column(JsonB)
    token_ids: Mapped[list | None] = mapped_column(JsonB)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    closed: Mapped[bool] = mapped_column(Boolean, default=False)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    liquidity: Mapped[Decimal | None] = mapped_column(Numeric)
    volume: Mapped[Decimal | None] = mapped_column(Numeric)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
