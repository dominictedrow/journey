"""Module 8 — the risk gate.

`RiskEngine.check` is the single entry point the copy engine calls before every trade:
kill switch first, then the ordered rule chain over a `RiskContext` built from the current
paper/live portfolio. It returns a `RiskDecision` and records a `risk_event` for any denial
or size adjustment (Transparency). The engine adds event rows to the session but leaves the
commit to the caller, so a copy attempt and its risk trail persist atomically.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import PortfolioPosition, PortfolioSnapshot, RiskEvent, Wallet
from app.risk.killswitch import is_engaged
from app.risk.models import RiskContext, RiskDecision, RiskLimits, TradeIntent
from app.risk.rules import Rule, default_rules

_WINDOWS = {"day": timedelta(days=1), "week": timedelta(weeks=1), "month": timedelta(days=30)}


def _cost(position: PortfolioPosition) -> Decimal:
    return position.size * (position.avg_entry_price or Decimal(0))


async def _period_pnl(session: AsyncSession, mode: str, now: datetime) -> dict[str, Decimal | None]:
    """Mark-to-market equity change over each trailing window, from portfolio snapshots.

    The change is `latest_equity - equity_at_window_start`; a window with no earlier snapshot
    to anchor against yields None (the loss-limit rule then skips it)."""
    latest = await session.scalar(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.mode == mode)
        .order_by(PortfolioSnapshot.timestamp.desc())
        .limit(1)
    )
    if latest is None:
        return {window: None for window in _WINDOWS}
    result: dict[str, Decimal | None] = {}
    for window, span in _WINDOWS.items():
        baseline = await session.scalar(
            select(PortfolioSnapshot)
            .where(
                PortfolioSnapshot.mode == mode,
                PortfolioSnapshot.timestamp >= now - span,
            )
            .order_by(PortfolioSnapshot.timestamp.asc())
            .limit(1)
        )
        result[window] = latest.total_value - baseline.total_value if baseline is not None else None
    return result


async def build_context(
    session: AsyncSession, intent: TradeIntent, mode: str, now: datetime | None = None
) -> RiskContext:
    """Assemble the portfolio state the rules evaluate against."""
    now = now or datetime.now(UTC)
    positions = (
        await session.scalars(
            select(PortfolioPosition).where(
                PortfolioPosition.mode == mode,
                PortfolioPosition.is_open.is_(True),
                PortfolioPosition.size > 0,
            )
        )
    ).all()

    total = Decimal(0)
    token_exposure = Decimal(0)
    market_exposure = Decimal(0)
    wallet_exposure = Decimal(0)
    holds_token = False
    for pos in positions:
        cost = _cost(pos)
        total += cost
        if pos.token_id == intent.token_id:
            token_exposure += cost
            holds_token = True
        if intent.market_id is not None and pos.market_id == intent.market_id:
            market_exposure += cost
        if pos.source_wallet == intent.source_wallet:
            wallet_exposure += cost

    wallet = await session.get(Wallet, intent.source_wallet)
    return RiskContext(
        total_exposure=total,
        open_positions=len(positions),
        token_exposure=token_exposure,
        market_exposure=market_exposure,
        wallet_exposure=wallet_exposure,
        holds_token=holds_token,
        wallet_copy_enabled=bool(wallet.copy_enabled) if wallet is not None else False,
        wallet_disabled_reason=wallet.disabled_reason if wallet is not None else "unknown wallet",
        period_pnl=await _period_pnl(session, mode, now),
    )


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None, rules: list[Rule] | None = None) -> None:
        self.limits = limits or RiskLimits.from_settings(get_settings())
        self.rules = rules if rules is not None else default_rules(self.limits)

    async def check(
        self, session: AsyncSession, intent: TradeIntent, *, record: bool = True
    ) -> RiskDecision:
        now = datetime.now(UTC)
        if await is_engaged(session):
            decision = RiskDecision(
                allow=False,
                adjusted_size=Decimal(0),
                reason="kill switch engaged",
                triggered_rule="kill_switch",
            )
            if record:
                self._record(session, intent, decision, now, "critical", "blocked")
            return decision

        ctx = await build_context(session, intent, intent.mode, now)
        size = intent.size
        adjustments: list[dict] = []
        for rule in self.rules:
            result = rule.evaluate(ctx, intent, size)
            if result.action == "deny":
                decision = RiskDecision(
                    allow=False,
                    adjusted_size=Decimal(0),
                    reason=result.reason,
                    triggered_rule=rule.name,
                    adjustments=adjustments,
                )
                if record:
                    self._record(session, intent, decision, now, "warning", "blocked")
                return decision
            if result.action == "adjust":
                adjustments.append(
                    {
                        "rule": rule.name,
                        "from": float(size),
                        "to": float(result.size),
                        "reason": result.reason,
                    }
                )
                size = result.size

        decision = RiskDecision(allow=True, adjusted_size=size, adjustments=adjustments)
        if record and adjustments:
            self._record(session, intent, decision, now, "info", "reduced")
        return decision

    @staticmethod
    def _record(
        session: AsyncSession,
        intent: TradeIntent,
        decision: RiskDecision,
        now: datetime,
        severity: str,
        action: str,
    ) -> None:
        session.add(
            RiskEvent(
                timestamp=now,
                type=decision.triggered_rule or "size_adjusted",
                severity=severity,
                context={
                    "source_wallet": intent.source_wallet,
                    "token_id": intent.token_id,
                    "market_id": intent.market_id,
                    "side": intent.side,
                    "requested_size": float(intent.size),
                    "adjusted_size": float(decision.adjusted_size),
                    "reason": decision.reason,
                    "adjustments": decision.adjustments,
                },
                action_taken=action,
            )
        )
