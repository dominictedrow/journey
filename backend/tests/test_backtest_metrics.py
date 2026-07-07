"""Unit tests for Module-7 metrics over a synthetic 60-day equity curve (hand-checkable
except the annualized daily-series constants, pinned from a verified run)."""

from decimal import Decimal

from app.backtesting.metrics import compute_backtest_metrics

CURVE = [
    {"t": "2026-01-01T00:00:00+00:00", "equity": 1000.0},
    {"t": "2026-01-21T00:00:00+00:00", "equity": 1200.0},
    {"t": "2026-02-10T00:00:00+00:00", "equity": 900.0},
    {"t": "2026-03-02T00:00:00+00:00", "equity": 1080.0},
]
LOG = [
    {"status": "filled", "notional": 500.0, "fee": 1.0, "realized_pnl": None},
    {"status": "filled", "notional": 600.0, "fee": 1.2, "realized_pnl": 200.0},
    {"status": "filled", "notional": 300.0, "fee": 0.6, "realized_pnl": -50.0},
    {"status": "skipped", "notional": None, "fee": None, "realized_pnl": None},
]


def test_metrics_golden() -> None:
    metrics = compute_backtest_metrics(CURVE, LOG, Decimal("1000"))
    assert metrics == {
        "initial_capital": 1000.0,
        "final_equity": 1080.0,
        "total_return": 0.08,
        "cagr": 0.597597,  # 1.08 ** (365.25 / 60) - 1
        "max_drawdown": 0.25,  # 1200 peak -> 900 trough
        "sharpe": 0.981357,  # pinned (daily forward-filled series, sqrt(365))
        "sortino": 1.479865,  # pinned
        "win_rate": 0.5,
        "profit_factor": 4.0,  # 200 gross profit / 50 gross loss
        "turnover": 1.4,  # 1400 filled notional / 1000 capital
        "fees_paid": 2.8,
        "trades_executed": 3,
        "trades_skipped": 1,
    }


def test_metrics_short_span_skips_annualization() -> None:
    curve = [
        {"t": "2026-01-01T00:00:00+00:00", "equity": 1000.0},
        {"t": "2026-01-02T00:00:00+00:00", "equity": 1100.0},
    ]
    metrics = compute_backtest_metrics(curve, [], Decimal("1000"))
    assert metrics["cagr"] is None  # 1-day span < 30-day floor
    assert metrics["sharpe"] is None  # a single daily return has no dispersion
    assert metrics["total_return"] == 0.1


def test_metrics_empty_inputs() -> None:
    metrics = compute_backtest_metrics([], [], Decimal("500"))
    assert metrics["final_equity"] == 500.0
    assert metrics["total_return"] == 0.0
    assert metrics["max_drawdown"] is None
    assert metrics["trades_executed"] == 0
