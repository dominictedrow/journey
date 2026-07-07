from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Wallet(Base):
    """A tracked (or candidate) Polymarket wallet. Addresses stored lowercase."""

    __tablename__ = "wallets"

    address: Mapped[str] = mapped_column(String(42), primary_key=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovery_source: Mapped[str | None] = mapped_column(String(64))
    discovered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_volume: Mapped[Decimal | None] = mapped_column(Numeric)
    total_trades: Mapped[int | None] = mapped_column(BigInteger)
    is_tracked: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    copy_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled_reason: Mapped[str | None] = mapped_column(Text)
