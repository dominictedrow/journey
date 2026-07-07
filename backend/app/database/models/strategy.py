from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK, JsonB


class Strategy(Base):
    """A copy-portfolio strategy (Module 6). `params` holds selection rule, sizing model,
    weighting, and limits; `rationale` explains why the optimizer adopted it."""

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    params: Mapped[dict] = mapped_column(JsonB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    adopted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rationale: Mapped[dict | None] = mapped_column(JsonB)
