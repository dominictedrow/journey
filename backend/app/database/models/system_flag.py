from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class SystemFlag(Base):
    """A named runtime switch shared across every process via the database.

    The kill switch (`key="kill_switch"`) lives here: a DB flag makes the halt
    durable across restarts and visible to the API, workers, and copy engine alike
    (Module 8). `reason` records who/why for the audit trail (Transparency)."""

    __tablename__ = "system_flags"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
