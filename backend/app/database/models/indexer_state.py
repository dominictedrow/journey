from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class IndexerState(Base):
    """Block cursor per indexer stream (historical/live) for crash recovery (Module 1)."""

    __tablename__ = "indexer_state"
    __table_args__ = (UniqueConstraint("stream", name="uq_indexer_state_stream"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    # Wide enough for per-wallet streams: "history:0x<40>" / "backfill:0x<40>" (~51 chars).
    stream: Mapped[str] = mapped_column(String(80))
    last_block: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
