from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import Scripted, make_bars
from trader.backtest.passive import PassiveBacktester
from trader.instruments import EURUSD
from trader.modes.base import Enter, Exit, Side

H = 0.00005  # 0.5 pip half-spread
P = 1.10000
DIST = 0.00100  # 10-pip stop distance
FLAT = (P, P + 0.00002, P - 0.00002, P)
FX = replace(EURUSD, slippage=0.00002)


def bar(o, h=None, low=None, c=None):
    return (o, h if h is not None else o + 0.00002, low if low is not None else o - 0.00002,
            c if c is not None else o)  # fmt: skip


def run(rows, script, *, feat=None, **kw):
    bars = make_bars(rows)
    mode = Scripted(script)
    if feat is not None:
        mode.feat = feat
    bt = PassiveBacktester(FX, **kw)
    return bt.run(mode, bars, np.full(len(bars), H)), bars


def long_signal():
    return {0: [Enter(Side.LONG, stop=P - DIST)]}


def test_buy_limit_fills_only_when_ask_crosses_bid():
    # Order placed at minute 1 open: limit = bid = P - H. Minute 2 mid low P - 1.5H: ask low
    # P - 0.5H > limit, no fill. Minute 3 mid low P - 2H: ask low P - H = limit, fills.
    rows = [FLAT, FLAT, bar(P, low=P - 1.5 * H), bar(P, low=P - 2 * H), FLAT, FLAT]
    res, bars = run(rows, long_signal(), ttl="10min")
    t = res.trades.iloc[0]
    assert t.entry_time == bars.index[3]
    assert t.entry_price == pytest.approx(P - H)
    assert t.initial_stop == pytest.approx(P - H - DIST)
    assert bool(res.signals.filled.iloc[0])


def test_unfilled_order_expires_and_is_logged():
    rows = [FLAT, FLAT, FLAT, FLAT, bar(P, low=P - 5 * H), FLAT]
    res, _ = run(rows, long_signal(), ttl="2min")  # live on minutes 1-2 only
    assert res.trades.empty
    assert len(res.signals) == 1 and not res.signals.filled.iloc[0]


def test_through_rule_fills_where_cross_does_not():
    rows = [FLAT, FLAT, bar(P, low=P - 0.00002), FLAT, FLAT]
    cross, _ = run(rows, long_signal(), ttl="10min", fill_rule="cross")
    thru, _ = run(rows, long_signal(), ttl="10min", fill_rule="through", through=0.00001)
    assert cross.trades.empty
    assert len(thru.trades) == 1


def test_long_stop_triggers_on_bid_and_pays_slippage():
    stop = P - H - DIST
    # Minute 3 mid low = stop + 0.9H: bid low = stop - 0.1H <= stop -> stopped.
    rows = [FLAT, FLAT, bar(P, low=P - 2 * H), FLAT, bar(P, low=stop + 0.9 * H), FLAT]
    res, bars = run(rows, long_signal(), ttl="10min")
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop"
    assert t.exit_time == bars.index[4]
    assert t.exit_price == pytest.approx(stop - FX.slippage)
    assert t.r_multiple < -1


def test_stop_can_hit_in_the_fill_minute():
    stop = P - H - DIST
    rows = [FLAT, FLAT, bar(P, low=stop - H), FLAT]
    res, bars = run(rows, long_signal(), ttl="10min")
    t = res.trades.iloc[0]
    assert t.entry_time == t.exit_time == bars.index[2]
    assert t.exit_price == pytest.approx(stop - FX.slippage)


def test_take_profit_limit_fills_on_bid_cross_after_entry_minute():
    # Anchor = close - z*sigma = P + 0.0005 at the signal bar.
    feat = {"z": np.full(8, -2.0), "sigma": np.full(8, 0.00025)}
    target = P + 0.0005
    rows = [FLAT, FLAT, bar(P, low=P - 2 * H, h=target + 2 * H), FLAT,
            bar(P, h=target + H), FLAT]  # fmt: skip
    res, bars = run(rows, long_signal(), feat=feat, take_profit=True, ttl="10min")
    t = res.trades.iloc[0]
    assert t.entry_time == bars.index[2]
    assert t.exit_reason == "target"
    assert t.exit_time == bars.index[4]  # not the entry minute, despite its high
    assert t.exit_price == pytest.approx(target)


def test_stop_wins_over_target_in_same_minute():
    feat = {"z": np.full(6, -2.0), "sigma": np.full(6, 0.00025)}
    stop = P - H - DIST
    rows = [FLAT, FLAT, bar(P, low=P - 2 * H), bar(P, h=P + 0.002, low=stop - H), FLAT]
    res, _ = run(rows, long_signal(), feat=feat, take_profit=True, ttl="10min")
    assert res.trades.iloc[0].exit_reason == "stop"


def test_market_exit_sells_at_bid_minus_slippage():
    script = {0: [Enter(Side.LONG, stop=P - DIST)], 3: [Exit("time")]}
    rows = [FLAT, FLAT, bar(P, low=P - 2 * H), FLAT, bar(P + 0.0003), FLAT]
    res, bars = run(rows, script, ttl="10min")
    t = res.trades.iloc[0]
    assert t.exit_time == bars.index[4]
    assert t.exit_price == pytest.approx(P + 0.0003 - H - FX.slippage)


def test_e2_ignores_reverted_exit_but_honours_time_exit():
    feat = {"z": np.full(8, -2.0), "sigma": np.full(8, 0.00025)}
    script = {0: [Enter(Side.LONG, stop=P - DIST)], 3: [Exit("reverted")], 5: [Exit("time")]}
    rows = [FLAT, FLAT, bar(P, low=P - 2 * H), FLAT, FLAT, FLAT, FLAT, FLAT]
    res, bars = run(rows, script, feat=feat, take_profit=True, ttl="10min")
    t = res.trades.iloc[0]
    assert t.exit_reason == "time"
    assert t.exit_time == bars.index[6]


def test_short_limit_fills_when_bid_crosses_ask():
    rows = [FLAT, FLAT, bar(P, h=P + 2 * H), FLAT, FLAT]
    res, bars = run(rows, {0: [Enter(Side.SHORT, stop=P + DIST)]}, ttl="10min")
    t = res.trades.iloc[0]
    assert t.side == -1
    assert t.entry_price == pytest.approx(P + H)
    assert t.entry_time == bars.index[2]


def test_half_spread_validated():
    bars = make_bars([FLAT, FLAT])
    with pytest.raises(ValueError):
        PassiveBacktester(FX).run(Scripted({}), bars, np.array([H]))
