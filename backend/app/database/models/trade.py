from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK


class Trade(Base):
    """A single fill, from on-chain events ('chain') or the Polymarket Data-API ('data_api').

    Unique (tx_hash, log_index) makes ingestion idempotent (dedup guarantee). The two sources
    cover disjoint block ranges split at a per-wallet handoff block, so a tx_hash only ever
    comes from one source — the constraint holds even though Data-API rows carry a synthetic,
    deterministic log_index rather than a real on-chain one. See app/ingest/history.py.
    """

    __tablename__ = "trades"
    __table_args__ = (UniqueConstraint("tx_hash", "log_index", name="uq_trades_tx_hash_log_index"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    tx_hash: Mapped[str] = mapped_column(String(66))
    log_index: Mapped[int] = mapped_column(Integer)
    wallet_address: Mapped[str] = mapped_column(ForeignKey("wallets.address"), index=True)
    market_id: Mapped[str | None] = mapped_column(ForeignKey("markets.id"), index=True)
    token_id: Mapped[str | None] = mapped_column(String(80), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY / SELL
    price: Mapped[Decimal] = mapped_column(Numeric)
    size: Mapped[Decimal] = mapped_column(Numeric)
    usdc_amount: Mapped[Decimal | None] = mapped_column(Numeric)
    block_number: Mapped[int | None] = mapped_column(BigInteger, index=True)  # null for data_api
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trade_type: Mapped[str | None] = mapped_column(String(16))  # open/increase/decrease/close
    source: Mapped[str] = mapped_column(String(16), server_default="chain")  # chain | data_api
