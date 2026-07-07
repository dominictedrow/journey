"""Module 4 — adaptive intelligence: composite wallet scoring.

`score_cohort` is a pure function of the scored cohort's metrics rows (plus per-wallet
copy-execution stats): each input metric is normalized to [0, 1] as its percentile rank
*within the cohort*, sub-scores average their available inputs, and the composite applies
the fixed weighting from docs/plan.md:

    composite = 0.40*historical + 0.30*recent + 0.20*copy_quality + 0.10*risk

`explanation` records every input's raw value and normalized score plus each component's
weight and contribution (Transparency principle). Copy-quality uses real `copy_orders`
execution stats when the wallet has been copied; otherwise a neutral prior of 0.5,
flagged in the explanation. `score_all_wallets` is the DB wrapper that loads the cohort,
aggregates copy stats, and appends one `wallet_scores` generation (shared `computed_at`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import CopyOrder, WalletMetrics, WalletScore

QUANTUM = Decimal("0.000001")
NEUTRAL_SCORE = Decimal("0.5")

WEIGHTS = {
    "historical": Decimal("0.40"),
    "recent": Decimal("0.30"),
    "copy_quality": Decimal("0.20"),
    "risk": Decimal("0.10"),
}

# WalletMetrics columns feeding each percentile-normalized component: (column, higher_is_better).
COMPONENT_INPUTS: dict[str, tuple[tuple[str, bool], ...]] = {
    "historical": (
        ("roi_lifetime", True),
        ("profit_factor", True),
        ("win_rate", True),
        ("sharpe", True),
        ("cagr", True),
    ),
    "recent": (
        ("roi_30d", True),
        ("roi_90d", True),
        ("monthly_consistency", True),
        ("weekly_consistency", True),
    ),
    "risk": (
        ("max_drawdown", False),
        ("volatility", False),
        ("concentration", False),
        ("losing_streak_max", False),
        ("sortino", True),
    ),
}


def _dec(value: float | Decimal) -> Decimal:
    return Decimal(str(float(value))).quantize(QUANTUM)


@dataclass
class CopyExecutionStats:
    """Aggregated execution quality of copying one source wallet (from `copy_orders`)."""

    fill_rate: Decimal | None  # sum(filled_size) / sum(requested_size)
    avg_slippage: Decimal | None
    avg_delay_ms: Decimal | None


@dataclass
class ScoreResult:
    composite: Decimal
    components: dict[str, Decimal]  # keyed like WEIGHTS
    explanation: dict


def _percentile(value: float, cohort: Sequence[float]) -> Decimal:
    """Mean-rank percentile in [0, 1]; a single-member cohort scores 0.5 (no information)."""
    less = sum(1 for v in cohort if v < value)
    equal = sum(1 for v in cohort if v == value)
    return _dec((less + equal / 2) / len(cohort))


def _component(
    scores: list[tuple[str, float, Decimal]],
) -> tuple[Decimal, dict]:
    """Average the (metric, raw value, normalized score) inputs; neutral if none apply."""
    if not scores:
        return NEUTRAL_SCORE, {"score": float(NEUTRAL_SCORE), "inputs": {}, "neutral_prior": True}
    mean = _dec(sum((s for _, _, s in scores), Decimal(0)) / len(scores))
    inputs = {metric: {"value": value, "score": float(s)} for metric, value, s in scores}
    return mean, {"score": float(mean), "inputs": inputs}


def _copy_quality(
    stats: CopyExecutionStats | None,
    slippage_cohort: Sequence[float],
    delay_cohort: Sequence[float],
) -> tuple[Decimal, dict]:
    """Fill rate is an absolute [0, 1] ratio; slippage/delay normalize against the cohort
    of wallets that have been copied (lower is better)."""
    if stats is None:
        return NEUTRAL_SCORE, {"score": float(NEUTRAL_SCORE), "inputs": {}, "neutral_prior": True}
    scores: list[tuple[str, float, Decimal]] = []
    if stats.fill_rate is not None:
        rate = min(max(stats.fill_rate, Decimal(0)), Decimal(1))
        scores.append(("fill_rate", float(stats.fill_rate), _dec(rate)))
    if stats.avg_slippage is not None and slippage_cohort:
        value = float(stats.avg_slippage)
        scores.append(("avg_slippage", value, 1 - _percentile(value, slippage_cohort)))
    if stats.avg_delay_ms is not None and delay_cohort:
        value = float(stats.avg_delay_ms)
        scores.append(("avg_delay_ms", value, 1 - _percentile(value, delay_cohort)))
    return _component(scores)


def score_cohort(
    metrics_rows: Sequence[WalletMetrics],
    copy_stats: Mapping[str, CopyExecutionStats] | None = None,
) -> dict[str, ScoreResult]:
    """Score every wallet in the cohort; returns results keyed by wallet address."""
    copy_stats = copy_stats or {}
    rows = sorted(metrics_rows, key=lambda m: m.wallet_address)

    cohorts: dict[str, list[float]] = {}
    for inputs in COMPONENT_INPUTS.values():
        for metric, _ in inputs:
            cohorts[metric] = [float(v) for m in rows if (v := getattr(m, metric)) is not None]
    slippage_cohort = [
        float(s.avg_slippage) for s in copy_stats.values() if s.avg_slippage is not None
    ]
    delay_cohort = [
        float(s.avg_delay_ms) for s in copy_stats.values() if s.avg_delay_ms is not None
    ]

    results: dict[str, ScoreResult] = {}
    for m in rows:
        components: dict[str, Decimal] = {}
        expl_components: dict[str, dict] = {}
        for name, inputs in COMPONENT_INPUTS.items():
            scores: list[tuple[str, float, Decimal]] = []
            for metric, higher_is_better in inputs:
                raw = getattr(m, metric)
                if raw is None:
                    continue
                value = float(raw)
                pct = _percentile(value, cohorts[metric])
                scores.append((metric, value, pct if higher_is_better else 1 - pct))
            components[name], expl_components[name] = _component(scores)
        components["copy_quality"], expl_components["copy_quality"] = _copy_quality(
            copy_stats.get(m.wallet_address), slippage_cohort, delay_cohort
        )

        composite = _dec(sum((WEIGHTS[name] * components[name] for name in WEIGHTS), Decimal(0)))
        for name in WEIGHTS:
            expl_components[name]["weight"] = float(WEIGHTS[name])
            expl_components[name]["contribution"] = float(_dec(WEIGHTS[name] * components[name]))
        results[m.wallet_address] = ScoreResult(
            composite=composite,
            components=components,
            explanation={"cohort_size": len(rows), "components": expl_components},
        )
    return results


async def _load_copy_stats(session: AsyncSession) -> dict[str, CopyExecutionStats]:
    """Aggregate execution quality per source wallet over all its copy orders. Unfilled
    orders drag the fill rate down (filled_size coalesced to 0), which is the point."""
    rows = await session.execute(
        select(
            CopyOrder.source_wallet,
            func.sum(func.coalesce(CopyOrder.filled_size, 0)),
            func.sum(CopyOrder.requested_size),
            func.avg(CopyOrder.slippage),
            func.avg(CopyOrder.delay_ms),
        )
        .where(CopyOrder.source_wallet.is_not(None))
        .group_by(CopyOrder.source_wallet)
    )
    stats: dict[str, CopyExecutionStats] = {}
    for wallet, filled, requested, slippage, delay in rows:
        stats[wallet] = CopyExecutionStats(
            fill_rate=_dec(Decimal(str(filled)) / Decimal(str(requested))) if requested else None,
            avg_slippage=_dec(slippage) if slippage is not None else None,
            avg_delay_ms=_dec(delay) if delay is not None else None,
        )
    return stats


async def score_all_wallets(
    session: AsyncSession, now: datetime | None = None
) -> list[WalletScore]:
    """Score every wallet with computed metrics; append one `wallet_scores` generation."""
    now = now or datetime.now(UTC)
    metrics_rows = (
        await session.scalars(select(WalletMetrics).where(WalletMetrics.sample_size > 0))
    ).all()
    results = score_cohort(metrics_rows, await _load_copy_stats(session))
    score_rows = [
        WalletScore(
            wallet_address=address,
            computed_at=now,
            composite=result.composite,
            historical_component=result.components["historical"],
            recent_component=result.components["recent"],
            copy_quality_component=result.components["copy_quality"],
            risk_component=result.components["risk"],
            explanation=result.explanation,
        )
        for address, result in sorted(results.items())
    ]
    session.add_all(score_rows)
    await session.commit()
    return score_rows
