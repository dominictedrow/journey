"""Golden tests for Module 4: cohort-normalized composite scoring.

Two-wallet cohort where A beats B on every input: mean-rank percentiles are exactly
0.75/0.25, so every component and the 0.40/0.30/0.20/0.10 composite is verifiable by hand.
"""

from datetime import UTC, datetime
from decimal import Decimal

from app.adaptive_intelligence.scoring import (
    WEIGHTS,
    CopyExecutionStats,
    score_cohort,
)
from app.database.models import WalletMetrics

WALLET_A = "0x" + "aa" * 20
WALLET_B = "0x" + "bb" * 20
NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _metrics(address: str, **values) -> WalletMetrics:
    return WalletMetrics(wallet_address=address, computed_at=NOW, sample_size=50, **values)


def fixture_cohort() -> list[WalletMetrics]:
    strong = _metrics(
        WALLET_A,
        roi_lifetime=Decimal("0.5"),
        profit_factor=Decimal("3"),
        win_rate=Decimal("0.6"),
        sharpe=Decimal("2"),
        cagr=Decimal("1.0"),
        roi_30d=Decimal("0.2"),
        roi_90d=Decimal("0.3"),
        monthly_consistency=Decimal("1"),
        weekly_consistency=Decimal("0.8"),
        max_drawdown=Decimal("0.05"),
        volatility=Decimal("0.2"),
        concentration=Decimal("0.3"),
        losing_streak_max=1,
        sortino=Decimal("3"),
    )
    weak = _metrics(
        WALLET_B,
        roi_lifetime=Decimal("-0.1"),
        profit_factor=Decimal("0.5"),
        win_rate=Decimal("0.3"),
        sharpe=Decimal("-1"),
        cagr=Decimal("-0.5"),
        roi_30d=Decimal("-0.05"),
        roi_90d=Decimal("-0.1"),
        monthly_consistency=Decimal("0.25"),
        weekly_consistency=Decimal("0.2"),
        max_drawdown=Decimal("0.5"),
        volatility=Decimal("0.9"),
        concentration=Decimal("0.9"),
        losing_streak_max=5,
        sortino=Decimal("0.1"),
    )
    return [strong, weak]


def test_golden_composite_weighting() -> None:
    results = score_cohort(fixture_cohort())
    a, b = results[WALLET_A], results[WALLET_B]

    # A wins every percentile-normalized input: (1 + 0.5) / 2 = 0.75 across the board.
    assert a.components == {
        "historical": Decimal("0.75"),
        "recent": Decimal("0.75"),
        "risk": Decimal("0.75"),
        "copy_quality": Decimal("0.5"),  # never copied -> neutral prior
    }
    # 0.40*0.75 + 0.30*0.75 + 0.20*0.50 + 0.10*0.75 = 0.30 + 0.225 + 0.10 + 0.075
    assert a.composite == Decimal("0.700000")
    assert b.components["historical"] == Decimal("0.25")
    # 0.40*0.25 + 0.30*0.25 + 0.20*0.50 + 0.10*0.25
    assert b.composite == Decimal("0.300000")


def test_composite_reproduces_weights_exactly() -> None:
    for result in score_cohort(fixture_cohort()).values():
        expected = sum((WEIGHTS[name] * result.components[name] for name in WEIGHTS), Decimal(0))
        assert result.composite == Decimal(str(float(expected))).quantize(Decimal("0.000001"))


def test_explanation_surfaces_each_contribution() -> None:
    explanation = score_cohort(fixture_cohort())[WALLET_A].explanation
    assert explanation["cohort_size"] == 2
    components = explanation["components"]
    assert set(components) == {"historical", "recent", "copy_quality", "risk"}
    historical = components["historical"]
    assert historical["weight"] == 0.4
    assert historical["score"] == 0.75
    assert historical["contribution"] == 0.3
    assert historical["inputs"]["roi_lifetime"] == {"value": 0.5, "score": 0.75}
    # Lower-is-better inputs invert the percentile.
    assert components["risk"]["inputs"]["max_drawdown"] == {"value": 0.05, "score": 0.75}
    assert components["copy_quality"]["neutral_prior"] is True


def test_copy_quality_uses_execution_stats() -> None:
    stats = {
        WALLET_A: CopyExecutionStats(
            fill_rate=Decimal("0.9"),
            avg_slippage=Decimal("0.01"),
            avg_delay_ms=Decimal("500"),
        )
    }
    results = score_cohort(fixture_cohort(), stats)
    a = results[WALLET_A]
    # fill_rate is absolute; slippage/delay are sole cohort members -> 0.5 each.
    assert a.components["copy_quality"] == Decimal("0.633333")
    assert a.composite == Decimal("0.726667")
    inputs = a.explanation["components"]["copy_quality"]["inputs"]
    assert inputs["fill_rate"] == {"value": 0.9, "score": 0.9}
    assert "neutral_prior" not in a.explanation["components"]["copy_quality"]
    # The uncopied wallet keeps the neutral prior.
    assert results[WALLET_B].components["copy_quality"] == Decimal("0.5")


def test_missing_metrics_fall_back_to_neutral() -> None:
    lonely = _metrics(WALLET_A)  # every metric None
    result = score_cohort([lonely])[WALLET_A]
    assert result.components == {
        "historical": Decimal("0.5"),
        "recent": Decimal("0.5"),
        "risk": Decimal("0.5"),
        "copy_quality": Decimal("0.5"),
    }
    assert result.composite == Decimal("0.500000")
    assert result.explanation["components"]["historical"]["neutral_prior"] is True


def test_single_wallet_cohort_scores_neutral() -> None:
    result = score_cohort([fixture_cohort()[0]])[WALLET_A]
    # A one-member cohort carries no relative information: every percentile is 0.5.
    assert result.composite == Decimal("0.500000")


def test_deterministic() -> None:
    assert score_cohort(fixture_cohort()) == score_cohort(fixture_cohort())
