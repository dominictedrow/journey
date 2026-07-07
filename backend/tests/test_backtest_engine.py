"""Golden-file tests for Module 7: fixed fixture + fixed config -> byte-stable output.

The fixture is one source wallet on one token with a rising price path into the exit
(observations at +0/5/15/30/60s) and a falling path after it, so longer execution delay
is adverse on both legs — the acceptance criterion that delay/slippage/fee scenarios
change results monotonically holds by construction. Hand-checkable anchor: with no
delay/slippage/fees, five $50 buys at 0.40/0.41/0.42/0.43/0.45 and a full exit at 0.80
end at exactly 1224.711216. The golden run's constants are pinned from a verified run.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.engine import run_backtest
from app.backtesting.models import BacktestConfig
from app.database.models import Trade

WALLET = "0x" + "aa" * 20
BASE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _trade(
    block: int, offset_s: int, side: str, price: str, size: str, token: str = "tokA"
) -> Trade:
    return Trade(
        tx_hash="0x" + f"{block:02x}" * 32,
        log_index=0,
        wallet_address=WALLET,
        market_id=None,
        token_id=token,
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        usdc_amount=Decimal(price) * Decimal(size),
        block_number=block,
        timestamp=BASE + timedelta(seconds=offset_s),
    )


def fixture_trades() -> list[Trade]:
    return [
        _trade(1, 0, "BUY", "0.40", "100"),
        _trade(2, 5, "BUY", "0.41", "50"),
        _trade(3, 15, "BUY", "0.42", "50"),
        _trade(4, 30, "BUY", "0.43", "50"),
        _trade(5, 60, "BUY", "0.45", "50"),
        # Source exits everything at the top, then the price path decays — the four
        # trailing sells are skipped (source is flat) but stamp the post-exit prices.
        _trade(6, 3600, "SELL", "0.80", "300"),
        _trade(7, 3605, "SELL", "0.78", "10"),
        _trade(8, 3615, "SELL", "0.76", "10"),
        _trade(9, 3630, "SELL", "0.74", "10"),
        _trade(10, 3660, "SELL", "0.70", "10"),
    ]


def _config(**overrides) -> BacktestConfig:
    defaults = dict(
        wallets=[WALLET],
        initial_capital=Decimal("1000"),
        sizing_mode="fixed",
        sizing_value=Decimal("50"),
        delay_seconds=5,
        slippage_pct=Decimal("0.01"),
        fee_pct=Decimal("0.001"),
    )
    return BacktestConfig(**{**defaults, **overrides})


def _final_equity(config: BacktestConfig) -> float:
    return run_backtest(config, fixture_trades()).metrics["final_equity"]


def test_golden_run() -> None:
    result = run_backtest(_config(), fixture_trades())

    assert result.metrics == {
        "initial_capital": 1000.0,
        "final_equity": 1200.645914,
        "total_return": 0.200646,
        "cagr": None,  # 1-hour span, far below the 30-day annualization floor
        "max_drawdown": 0.001752,  # first buy's slippage+fee dents equity below par
        "sharpe": None,  # single-day run has no daily return series
        "sortino": None,
        "win_rate": 1.0,
        "profit_factor": None,  # no losing realization -> undefined
        "turnover": 0.701347,
        "fees_paid": 0.701347,
        "trades_executed": 6,
        "trades_skipped": 4,
    }

    assert len(result.equity_curve) == 11  # initial point + one per source fill
    assert result.equity_curve[0] == {"t": "2026-06-01T12:00:00+00:00", "equity": 1000.0}
    assert result.equity_curve[1]["equity"] == 998.247513
    assert result.equity_curve[-1]["equity"] == 1200.645914

    first_buy = result.trade_log[0]
    # Delayed 5s onto the +5s print (0.41), then 1% adverse slippage -> 0.4141.
    assert first_buy["exec_price"] == 0.4141
    assert first_buy["notional"] == 50.0
    assert first_buy["fee"] == 0.05
    assert first_buy["size"] == 120.743782
    exit_fill = result.trade_log[5]
    # Source closes 300 of 300 -> we mirror a full close at 0.78 print less slippage.
    assert exit_fill["exec_price"] == 0.7722
    assert exit_fill["size"] == 584.495287
    assert exit_fill["realized_pnl"] == 200.896174
    assert result.trade_log[6] == {
        "t": "2026-06-01T13:00:05+00:00",
        "wallet": WALLET,
        "token": "tokA",
        "market_id": None,
        "side": "SELL",
        "source_price": 0.78,
        "source_size": 10.0,
        "status": "skipped",
        "reason": "no_source_position",
        "exec_price": None,
        "size": None,
        "notional": None,
        "fee": None,
        "realized_pnl": None,
        "cash": 1200.645914,
        "equity": 1200.645914,
    }


def test_deterministic() -> None:
    a = run_backtest(_config(), fixture_trades())
    b = run_backtest(_config(), fixture_trades())
    assert a.metrics == b.metrics
    assert a.equity_curve == b.equity_curve
    assert a.trade_log == b.trade_log


def test_delay_scenarios_monotonically_worse() -> None:
    zero = dict(slippage_pct=Decimal("0"), fee_pct=Decimal("0"))
    finals = [_final_equity(_config(delay_seconds=d, **zero)) for d in (0, 5, 15, 30, 60)]
    assert finals[0] == 1224.711216  # hand-checkable no-friction anchor
    assert finals == sorted(finals, reverse=True)
    assert len(set(finals)) == len(finals)  # strictly decreasing on this fixture


def test_slippage_scenarios_monotonically_worse() -> None:
    finals = [
        _final_equity(_config(delay_seconds=0, slippage_pct=Decimal(s), fee_pct=Decimal("0")))
        for s in ("0", "0.005", "0.01", "0.02")
    ]
    assert finals == sorted(finals, reverse=True)
    assert len(set(finals)) == len(finals)


def test_fee_scenarios_monotonically_worse() -> None:
    finals = [
        _final_equity(_config(delay_seconds=0, slippage_pct=Decimal("0"), fee_pct=Decimal(f)))
        for f in ("0", "0.001", "0.01")
    ]
    assert finals == sorted(finals, reverse=True)
    assert len(set(finals)) == len(finals)


def test_sizing_modes() -> None:
    frictionless = dict(delay_seconds=0, slippage_pct=Decimal("0"), fee_pct=Decimal("0"))
    fixed = run_backtest(_config(**frictionless), fixture_trades())
    assert fixed.trade_log[0]["notional"] == 50.0

    pct = run_backtest(
        _config(sizing_mode="percentage", sizing_value=Decimal("0.1"), **frictionless),
        fixture_trades(),
    )
    assert pct.trade_log[0]["notional"] == 100.0  # 10% of the initial 1000 equity

    prop = run_backtest(
        _config(sizing_mode="proportional", sizing_value=Decimal("0.5"), **frictionless),
        fixture_trades(),
    )
    assert prop.trade_log[0]["notional"] == 20.0  # half of the source's 40 USDC

    weighted = run_backtest(
        _config(weights={WALLET: Decimal("0.5")}, **frictionless), fixture_trades()
    )
    assert weighted.trade_log[0]["notional"] == 25.0  # fixed 50 x 0.5 wallet weight


def test_max_open_positions_skips_new_tokens() -> None:
    trades = [
        _trade(1, 0, "BUY", "0.40", "100", token="tokA"),
        _trade(2, 10, "BUY", "0.50", "100", token="tokB"),
        _trade(3, 20, "BUY", "0.42", "50", token="tokA"),  # increase is still allowed
    ]
    result = run_backtest(_config(max_open_positions=1, delay_seconds=0), trades)
    statuses = [(e["token"], e["status"], e["reason"]) for e in result.trade_log]
    assert statuses == [
        ("tokA", "filled", None),
        ("tokB", "skipped", "max_open_positions"),
        ("tokA", "filled", None),
    ]


def test_insufficient_cash_clips_then_skips() -> None:
    config = _config(
        initial_capital=Decimal("60"),
        delay_seconds=0,
        slippage_pct=Decimal("0"),
        fee_pct=Decimal("0"),
    )
    result = run_backtest(config, fixture_trades())
    buys = [e for e in result.trade_log if e["side"] == "BUY"]
    assert buys[0]["notional"] == 50.0  # full size
    assert buys[1]["notional"] == 10.0  # clipped to remaining cash
    assert {e["reason"] for e in buys[2:]} == {"insufficient_cash"}


def test_partial_sell_mirrors_source_fraction() -> None:
    trades = [
        _trade(1, 0, "BUY", "0.40", "100"),
        _trade(2, 60, "SELL", "0.50", "40"),  # source closes 40% of 100
    ]
    config = _config(delay_seconds=0, slippage_pct=Decimal("0"), fee_pct=Decimal("0"))
    result = run_backtest(config, trades)
    buy, sell = result.trade_log
    assert sell["status"] == "filled"
    assert sell["size"] == round(buy["size"] * 0.4, 6)


def test_sell_without_source_position_is_skipped() -> None:
    trades = [_trade(1, 0, "SELL", "0.50", "10")]
    result = run_backtest(_config(), trades)
    assert result.trade_log[0]["status"] == "skipped"
    assert result.trade_log[0]["reason"] == "no_source_position"
    assert result.metrics["final_equity"] == 1000.0


def test_empty_trades() -> None:
    result = run_backtest(_config(), [])
    assert result.equity_curve == []
    assert result.trade_log == []
    assert result.metrics["final_equity"] == 1000.0
    assert result.metrics["total_return"] == 0.0
