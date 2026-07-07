from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK, JsonB


class RiskEvent(Base):
    """Every risk-gate denial/adjustment and kill-switch action (Module 8, Transparency)."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")
    context: Mapped[dict | None] = mapped_column(JsonB)
    action_taken: Mapped[str | None] = mapped_column(String(255))
