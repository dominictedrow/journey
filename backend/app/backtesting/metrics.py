"""Module 7 — performance metrics over a backtest's equity curve and trade log.

Pure and deterministic: same curve + log = same metrics dict (values quantized to 6dp).
Daily statistics resample the event-level equity curve to calendar days (last equity of
each day, forward-filled over quiet days) before computing returns, mirroring the
annualization conventions of `wallet_analytics.metrics`.
"""

import statistics
from datetime import datetime
from decimal import Decimal
from math import sqrt

QUANTUM = Decimal("0.000001")
DAYS_PER_YEAR = 365.25
MIN_CAGR_SPAN_DAYS = 30  # annualizing shorter histories produces absurd numbers


def _dec(value: float | Decimal) -> Decimal:
    return Decimal(str(float(value))).quantize(QUANTUM)


def _f(value: float | Decimal) -> float:
    return float(_dec(value))


def _daily_equity(equity_curve: list[dict]) -> list[float]:
    """Last observed equity per calendar day, forward-filled across the active span."""
    by_day: dict = {}
    for point in equity_curve:
        day = datetime.fromisoformat(point["t"]).date()
        by_day[day] = point["equity"]  # points are ordered; later wins
    days = sorted(by_day)
    series: list[float] = []
    current = by_day[days[0]]
    day = days[0]
    while day <= days[-1]:
        current = by_day.get(day, current)
        series.append(current)
        day = day.fromordinal(day.toordinal() + 1)
    return series


def compute_backtest_metrics(
    equity_curve: list[dict], trade_log: list[dict], initial_capital: Decimal
) -> dict:
    """Every `backtests.metrics` value from the replay output."""
    metrics: dict = {
        "initial_capital": _f(initial_capital),
        "final_equity": _f(initial_capital),
        "total_return": 0.0,
        "cagr": None,
        "max_drawdown": None,
        "sharpe": None,
        "sortino": None,
        "win_rate": None,
        "profit_factor": None,
        "turnover": 0.0,
        "fees_paid": 0.0,
        "trades_executed": 0,
        "trades_skipped": 0,
    }
    filled = [e for e in trade_log if e["status"] == "filled"]
    metrics["trades_executed"] = len(filled)
    metrics["trades_skipped"] = len(trade_log) - len(filled)
    metrics["turnover"] = _f(sum(e["notional"] for e in filled) / float(initial_capital))
    metrics["fees_paid"] = _f(sum(e["fee"] for e in filled))

    if not equity_curve:
        return metrics
    final = equity_curve[-1]["equity"]
    metrics["final_equity"] = _f(final)
    metrics["total_return"] = _f(final / float(initial_capital) - 1)

    peak = drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        if peak > 0:
            drawdown = max(drawdown, (peak - point["equity"]) / peak)
    metrics["max_drawdown"] = _f(drawdown)

    first_ts = datetime.fromisoformat(equity_curve[0]["t"])
    last_ts = datetime.fromisoformat(equity_curve[-1]["t"])
    span_days = (last_ts - first_ts).total_seconds() / 86400
    if span_days >= MIN_CAGR_SPAN_DAYS and final > 0:
        growth = final / float(initial_capital)
        metrics["cagr"] = _f(growth ** (DAYS_PER_YEAR / span_days) - 1)

    daily = _daily_equity(equity_curve)
    returns = [
        (daily[i] - daily[i - 1]) / daily[i - 1] for i in range(1, len(daily)) if daily[i - 1] > 0
    ]
    if len(returns) >= 2:
        mean_r, std_r = statistics.fmean(returns), statistics.pstdev(returns)
        if std_r > 0:
            metrics["sharpe"] = _f(mean_r / std_r * sqrt(365))
        downside = sqrt(statistics.fmean(min(r, 0.0) ** 2 for r in returns))
        if downside > 0:
            metrics["sortino"] = _f(mean_r / downside * sqrt(365))

    realizations = [e for e in filled if e["realized_pnl"] is not None]
    if realizations:
        wins = [e for e in realizations if e["realized_pnl"] > 0]
        losses = [e for e in realizations if e["realized_pnl"] < 0]
        metrics["win_rate"] = _f(len(wins) / len(realizations))
        gross_profit = sum(e["realized_pnl"] for e in wins)
        gross_loss = -sum(e["realized_pnl"] for e in losses)
        if gross_loss > 0:
            metrics["profit_factor"] = _f(gross_profit / gross_loss)
    return metrics
