"""Golden test for Module 3: fixed fixture wallet -> byte-stable metric values.

The fixture is small enough to verify by hand (comments carry the arithmetic); the
annualized/statistical constants (cagr, sharpe, sortino) are captured from a verified
run and pinned to detect any behavioural drift in the engine.
"""

from datetime import UTC, datetime
from decimal import Decimal

from app.database.models import Market, Trade
from app.wallet_analytics.metrics import compute_metrics

WALLET = "0x" + "aa" * 20
NOW = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)


def _trade(
    block: int,
    ts: datetime,
    side: str,
    token_id: str,
    market_id: str,
    price: str,
    size: str,
) -> Trade:
    return Trade(
        tx_hash="0x" + f"{block:02x}" * 32,
        log_index=0,
        wallet_address=WALLET,
        market_id=market_id,
        token_id=token_id,
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        usdc_amount=Decimal(price) * Decimal(size),
        block_number=block,
        timestamp=ts,
    )


def fixture_markets() -> dict[str, Market]:
    return {
        "mA": Market(
            id="mA",
            question="A?",
            category="politics",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        "mB": Market(id="mB", question="B?", category="crypto"),
    }


def fixture_trades() -> list[Trade]:
    return [
        _trade(1, datetime(2026, 1, 10, 12, 0, tzinfo=UTC), "BUY", "tA", "mA", "0.40", "100"),
        _trade(2, datetime(2026, 1, 20, 12, 0, tzinfo=UTC), "BUY", "tA", "mA", "0.60", "100"),
        _trade(3, datetime(2026, 2, 1, 12, 0, tzinfo=UTC), "SELL", "tA", "mA", "0.70", "150"),
        _trade(4, datetime(2026, 2, 5, 12, 0, tzinfo=UTC), "BUY", "tB", "mB", "0.20", "50"),
        _trade(5, datetime(2026, 2, 10, 12, 0, tzinfo=UTC), "SELL", "tB", "mB", "0.10", "50"),
    ]


def test_golden_metrics() -> None:
    m = compute_metrics(fixture_trades(), fixture_markets(), now=NOW)

    assert m["sample_size"] == 5
    # tA: 200 @ avg 0.50, sell 150 @ 0.70 -> +30; tB: 50 @ 0.20, sell 50 @ 0.10 -> -5.
    assert m["realized_pnl"] == Decimal("25")
    # 50 tA left @ avg 0.50, marked at last observed price 0.70 -> +10.
    assert m["unrealized_pnl"] == Decimal("10")
    # (25 + 10) / (40 + 60 + 10) buys.
    assert m["roi_lifetime"] == Decimal("0.318182")
    assert m["win_rate"] == Decimal("0.5")
    # Returns 0.4 (tA) and -0.5 (tB).
    assert m["avg_trade_return"] == Decimal("-0.05")
    assert m["volatility"] == Decimal("0.45")
    # Gross profit 30 / gross loss 5.
    assert m["profit_factor"] == Decimal("6")
    # Both realizations within 30d of NOW: 25 pnl / 85 cost.
    assert m["roi_30d"] == Decimal("0.294118")
    assert m["roi_90d"] == Decimal("0.294118")
    # Both realizations land in February -> 1/1 positive months.
    assert m["monthly_consistency"] == Decimal("1")
    # ISO week of Feb 1 is +30, week of Feb 10 is -5 -> 1/2.
    assert m["weekly_consistency"] == Decimal("0.5")
    # Peak equity 30, trough 25 -> dd 5 over max invested 100 (200 tA @ 0.50).
    assert m["max_drawdown"] == Decimal("0.05")
    assert m["losing_streak_max"] == 1
    # Holds: 528h (Jan 10 -> Feb 1) and 120h (Feb 5 -> Feb 10).
    assert m["avg_hold_hours"] == Decimal("324")
    # 5 fills over a 31-day active span.
    assert m["trade_frequency"] == Decimal("0.161290")
    # Buy notionals 40, 60, 10.
    assert m["avg_position_size"] == Decimal("36.666667")
    # tB opened while 50 tA still held.
    assert m["max_concurrent_positions"] == 2
    # Volumes: mA 205 (40+60+105), mB 15 (10+5) of 220 total.
    assert m["category_mix"] == {"crypto": 0.068182, "politics": 0.931818}
    # HHI: (205/220)^2 + (15/220)^2.
    assert m["concentration"] == Decimal("0.872934")
    # mA entered 9.5 days into a 59-day lifetime: 1 - 9.5/59; mB has no dates.
    assert m["early_entry_score"] == Decimal("0.838983")
    # tA exit at range top (0.40-0.70), tB exit at range bottom (0.10-0.20).
    assert m["exit_timing_score"] == Decimal("0.5")
    # Mean of (0.70 - 0.50) and (0.10 - 0.20).
    assert m["avg_prob_improvement"] == Decimal("0.05")
    # Annualized constants pinned from a verified run (31-day span, daily PnL series).
    assert m["cagr"] == Decimal("24.916727")
    assert m["sharpe"] == Decimal("2.805916")
    assert m["sortino"] == Decimal("16.886570")


def test_metrics_deterministic() -> None:
    a = compute_metrics(fixture_trades(), fixture_markets(), now=NOW)
    b = compute_metrics(fixture_trades(), fixture_markets(), now=NOW)
    assert a == b


def test_empty_wallet_yields_nulls() -> None:
    m = compute_metrics([], {}, now=NOW)
    assert m["sample_size"] == 0
    assert m["roi_lifetime"] is None
    assert m["win_rate"] is None
    assert m["category_mix"] is None


def test_buys_only_wallet_has_no_realizations() -> None:
    trades = [t for t in fixture_trades() if t.side == "BUY"]
    m = compute_metrics(trades, fixture_markets(), now=NOW)
    assert m["sample_size"] == 3
    assert m["realized_pnl"] == Decimal("0")
    assert m["win_rate"] is None
    assert m["profit_factor"] is None
    # tA marked at its last observed price 0.60: (0.60 - 0.50) * 200; tB flat at 0.20.
    assert m["unrealized_pnl"] == Decimal("20")
