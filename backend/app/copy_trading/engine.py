"""Module 9 — copy engine.

Ties the pipeline together: a fresh source-wallet fill → size it per the copy sizing config →
`RiskEngine.check` → `Executor.execute` → persist a `copy_orders` row → fold the fill into the
portfolio. Every processed source trade yields exactly one copy order (keyed by
`source_trade_id`), so detection is idempotent and the audit trail is complete: rejected by
risk, skipped (nothing to mirror), or filled/partial.

BUY sizing mirrors the backtester: fixed USDC, a fraction of equity, or a multiple of the
source notional. SELL mirrors the *fraction* of the source's position being closed onto our
own holding, so exits track the source proportionally.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.copy_trading.executors import Executor, LiveExecutor, PaperExecutor, build_live_client
from app.database.models import CopyOrder, Market, PortfolioSnapshot, Position, Trade, Wallet
from app.portfolio.service import apply_fill, get_position
from app.risk.engine import RiskEngine
from app.risk.models import RiskDecision, TradeIntent

QUANTUM = Decimal("0.000001")
MIN_SPEND = Decimal("0.01")


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


def select_executor(settings: Settings) -> Executor:
    """Pick the executor for the current deployment. Live only when explicitly armed
    (`EXECUTION_MODE=live` + `LIVE_TRADING_ARMED=true`); every other configuration — including
    an unarmed live box — trades on paper, so live is opt-in and fails safe."""
    if settings.execution_mode == "live" and settings.live_trading_armed:
        return LiveExecutor(client=build_live_client(settings), settings=settings)
    return PaperExecutor()


class CopyEngine:
    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        executor: Executor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.risk = risk_engine or RiskEngine()
        self.executor = executor or select_executor(self.settings)
        self.mode = self.executor.mode

    async def _equity(self, session: AsyncSession) -> Decimal:
        snapshot = await session.scalar(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.mode == self.mode)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        )
        if snapshot is not None:
            return Decimal(snapshot.total_value)
        return self.settings.paper_starting_cash if self.mode == "paper" else Decimal(0)

    async def _buy_size(self, session: AsyncSession, trade: Trade) -> Decimal:
        """Shares to buy per the configured sizing model (before risk adjustment)."""
        price = trade.price
        if price <= 0:
            return Decimal(0)
        mode = self.settings.copy_sizing_mode
        value = self.settings.copy_sizing_value
        if mode == "fixed":
            spend = value
        elif mode == "percentage":
            spend = value * await self._equity(session)
        else:  # proportional to the source trade's own notional
            source_notional = (
                trade.usdc_amount if trade.usdc_amount is not None else price * trade.size
            )
            spend = source_notional * value
        return _dec(spend / price) if spend >= MIN_SPEND else Decimal(0)

    async def _sell_size(self, session: AsyncSession, trade: Trade) -> Decimal:
        """Shares to sell: the fraction of our holding matching the source's own close."""
        held = await get_position(session, self.mode, trade.token_id or "", trade.wallet_address)
        if held is None or held.size <= 0:
            return Decimal(0)
        source = await session.scalar(
            select(Position).where(
                Position.wallet_address == trade.wallet_address,
                Position.token_id == trade.token_id,
            )
        )
        source_remaining = source.net_size if source is not None else Decimal(0)
        source_before = source_remaining + trade.size
        fraction = trade.size / source_before if source_before > 0 else Decimal(1)
        if fraction >= 1:
            return held.size
        return _dec(held.size * fraction)

    async def process_trade(
        self, session: AsyncSession, trade: Trade, *, now: datetime | None = None
    ) -> CopyOrder:
        """Run one source fill through the full pipeline; commits and returns the copy order."""
        now = now or datetime.now(UTC)
        # SQLite (test suite) returns naive datetimes; Postgres returns aware. Normalize.
        ts = trade.timestamp if trade.timestamp.tzinfo else trade.timestamp.replace(tzinfo=UTC)
        delay_seconds = max(0.0, (now - ts).total_seconds())

        if trade.side == "BUY":
            size = await self._buy_size(session, trade)
        else:
            size = await self._sell_size(session, trade)

        order = CopyOrder(
            created_at=now,
            mode=self.mode,
            source_wallet=trade.wallet_address,
            source_trade_id=trade.id,
            market_id=trade.market_id,
            token_id=trade.token_id,
            side=trade.side,
            requested_size=_dec(size),
            requested_price=trade.price,
            status="pending",
        )
        session.add(order)

        if size <= 0:
            order.status = "skipped"
            order.filled_size = Decimal(0)
            order.error = "nothing to mirror"
            await session.commit()
            return order

        market = await session.get(Market, trade.market_id) if trade.market_id else None
        intent = TradeIntent(
            source_wallet=trade.wallet_address,
            token_id=trade.token_id or "",
            side=trade.side,  # type: ignore[arg-type]
            size=size,
            price=trade.price,
            market_id=trade.market_id,
            source_trade_id=trade.id,
            delay_seconds=delay_seconds,
            market_liquidity=market.liquidity if market is not None else None,
            mode=self.mode,
        )

        decision: RiskDecision = await self.risk.check(session, intent)
        if not decision.allow:
            order.status = "rejected"
            order.filled_size = Decimal(0)
            order.error = decision.reason
            await session.commit()
            return order

        result = await self.executor.execute(intent, decision.adjusted_size)
        order.filled_size = result.filled_size
        order.fill_price = result.fill_price
        order.delay_ms = result.delay_ms
        order.slippage = result.slippage
        order.status = result.status
        order.error = result.error
        order.external_order_id = result.external_order_id

        if result.filled_size > 0 and result.fill_price is not None:
            await apply_fill(
                session,
                mode=self.mode,
                source_wallet=trade.wallet_address,
                market_id=trade.market_id,
                token_id=trade.token_id or "",
                side=trade.side,
                filled_size=result.filled_size,
                fill_price=result.fill_price,
                now=now,
            )
        await session.commit()
        return order

    async def detect_and_copy(
        self, session: AsyncSession, now: datetime | None = None
    ) -> list[CopyOrder]:
        """Process every fresh, not-yet-copied fill from copy-enabled wallets, oldest first."""
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.settings.copy_max_trade_age_seconds)
        already_copied = (
            select(CopyOrder.source_trade_id)
            .where(CopyOrder.mode == self.mode, CopyOrder.source_trade_id.is_not(None))
            .scalar_subquery()
        )
        trades = (
            await session.scalars(
                select(Trade)
                .join(Wallet, Wallet.address == Trade.wallet_address)
                .where(
                    Wallet.copy_enabled.is_(True),
                    Trade.timestamp >= cutoff,
                    Trade.id.not_in(already_copied),
                )
                # Coalesce null block_number to 0 so Data-API rows sort exactly as
                # compute_positions orders them (which uses `block_number or 0`); Postgres would
                # otherwise sort nulls last and disagree if a data_api fill fell in this window.
                .order_by(Trade.timestamp, func.coalesce(Trade.block_number, 0), Trade.log_index)
            )
        ).all()
        orders = []
        for trade in trades:
            orders.append(await self.process_trade(session, trade, now=now))
        return orders
