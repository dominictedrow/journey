"""Module 6 — candidate generation and spec → `BacktestConfig` resolution.

`candidate_specs` enumerates the strategy grid the optimizer explores (top-N × weighting ×
sizing), pre-clamped to the configured risk limits so no candidate can be adopted that would
breach them. `resolve_config` binds a spec to the *latest* ranking generation: it picks the
top-N wallets of the spec's category, computes per-wallet weights (normalized to mean 1), and
produces the deterministic `BacktestConfig` the Phase-4 engine replays.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.models import BacktestConfig, SizingMode
from app.config import Settings
from app.database.models import Ranking, WalletMetrics
from app.strategy_optimizer.models import StrategySpec, Weighting

WEIGHTINGS: tuple[Weighting, ...] = ("equal", "score", "volatility")
SIZING_MODES: tuple[SizingMode, ...] = ("fixed", "percentage", "proportional")

# Default per-mode sizing values (as fractions of `initial_capital` where notional-denominated).
# fixed:        5% of capital spent per copied BUY.
# percentage:   5% of current equity spent per copied BUY.
# proportional: 1x the source trade's own notional.
_SIZING_FRACTION = {
    "fixed": Decimal("0.05"),
    "percentage": Decimal("0.05"),
    "proportional": Decimal("1.0"),
}


def _default_sizing_value(mode: SizingMode, initial_capital: Decimal) -> Decimal:
    if mode == "fixed":
        return (initial_capital * _SIZING_FRACTION["fixed"]).quantize(Decimal("0.01"))
    return _SIZING_FRACTION[mode]


def candidate_specs(
    settings: Settings, category: str = "overall", *, extra_top_n: int | None = None
) -> list[StrategySpec]:
    """Every strategy the optimizer evaluates this run, clamped to risk limits.

    `extra_top_n` — the activity governor's (Module D) widened selection size, read through
    the runtime-override layer by the caller — is folded into the grid alongside the static
    choices when the book has gone quiet, so a wider top-N gets a chance to be adopted."""
    max_positions = settings.risk_max_open_positions
    # The live rule caps a position at a % of *current equity*; the backtest replays against
    # a fixed simulated capital, so the same % is applied to that instead for an equivalent cap.
    max_size = settings.optimizer_initial_capital * settings.risk_max_position_size_pct
    choices = list(settings.optimizer_top_n_list)
    if extra_top_n:
        choices.append(extra_top_n)
    top_ns = [min(n, max_positions) for n in choices]
    top_ns = sorted({n for n in top_ns if n >= 1})

    specs: list[StrategySpec] = []
    for top_n in top_ns:
        for weighting in WEIGHTINGS:
            for mode in SIZING_MODES:
                specs.append(
                    StrategySpec(
                        category=category,
                        top_n=top_n,
                        weighting=weighting,
                        sizing_mode=mode,
                        sizing_value=_default_sizing_value(
                            mode, settings.optimizer_initial_capital
                        ),
                        max_open_positions=max_positions,
                        max_position_size=max_size,
                    )
                )
    return specs


async def _latest_ranking_rows(session: AsyncSession, category: str, top_n: int) -> list[Ranking]:
    """Top-N entries of the latest ranking generation for `category` (by rank)."""
    latest = await session.scalar(select(func.max(Ranking.computed_at)))
    if latest is None:
        return []
    return list(
        (
            await session.scalars(
                select(Ranking)
                .where(Ranking.computed_at == latest, Ranking.category == category)
                .order_by(Ranking.rank)
                .limit(top_n)
            )
        ).all()
    )


def _normalize(raw: dict[str, Decimal]) -> dict[str, Decimal]:
    """Scale weights to mean 1 (sum == count); fall back to equal if the total is non-positive."""
    n = len(raw)
    total = sum(raw.values(), Decimal(0))
    if n == 0 or total <= 0:
        return {addr: Decimal(1) for addr in raw}
    factor = Decimal(n) / total
    return {addr: (w * factor) for addr, w in raw.items()}


async def _weights(
    session: AsyncSession, rows: list[Ranking], weighting: Weighting
) -> dict[str, Decimal]:
    """Per-wallet spend multipliers for the weighting scheme, normalized to mean 1."""
    addresses = [r.wallet_address for r in rows]
    if weighting == "equal":
        return {addr: Decimal(1) for addr in addresses}
    if weighting == "score":
        raw = {r.wallet_address: max(r.score, Decimal(0)) for r in rows}
        return _normalize(raw)
    # volatility — weight by inverse metric volatility (lower vol ⇒ larger allocation).
    vols = {
        m.wallet_address: m.volatility
        for m in (
            await session.scalars(
                select(WalletMetrics).where(WalletMetrics.wallet_address.in_(addresses))
            )
        ).all()
    }

    def inverse_vol(addr: str) -> Decimal:
        vol = vols.get(addr)
        return Decimal(1) / vol if vol is not None and vol > 0 else Decimal(1)

    return _normalize({addr: inverse_vol(addr) for addr in addresses})


async def resolve_config(
    session: AsyncSession,
    spec: StrategySpec,
    *,
    initial_capital: Decimal,
    lookback_days: int,
    now: datetime | None = None,
) -> BacktestConfig | None:
    """Bind `spec` to the latest rankings → a replayable `BacktestConfig`, or None if the
    category has no ranked wallets yet."""
    rows = await _latest_ranking_rows(session, spec.category, spec.top_n)
    if not rows:
        return None
    now = now or datetime.now(UTC)
    weights = await _weights(session, rows, spec.weighting)
    return BacktestConfig(
        wallets=[r.wallet_address for r in rows],
        start=now - timedelta(days=lookback_days),
        end=now,
        initial_capital=initial_capital,
        sizing_mode=spec.sizing_mode,
        sizing_value=spec.sizing_value,
        weights=weights,
        delay_seconds=spec.delay_seconds,
        slippage_pct=spec.slippage_pct,
        fee_pct=spec.fee_pct,
        max_open_positions=spec.max_open_positions,
        max_position_size=spec.max_position_size,
    )
