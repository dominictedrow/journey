from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK, JsonB


class WalletScore(Base):
    """Composite score history (Module 4): 0.40*historical + 0.30*recent
    + 0.20*copy_quality + 0.10*risk. `explanation` records each input's contribution."""

    __tablename__ = "wallet_scores"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(ForeignKey("wallets.address"), index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    composite: Mapped[Decimal] = mapped_column(Numeric)
    historical_component: Mapped[Decimal | None] = mapped_column(Numeric)
    recent_component: Mapped[Decimal | None] = mapped_column(Numeric)
    copy_quality_component: Mapped[Decimal | None] = mapped_column(Numeric)
    risk_component: Mapped[Decimal | None] = mapped_column(Numeric)
    explanation: Mapped[dict | None] = mapped_column(JsonB)
