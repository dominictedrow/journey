"""Every registered mode must give identical decisions whether or not later data exists."""

import pandas as pd
import pytest

from tests.conftest import random_walk
from trader.backtest.engine import Backtester
from trader.instruments import MES
from trader.modes import REGISTRY, build_mode

CHEAP_PARAMS = {"random": {"entry_prob": 0.5}}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_truncation_invariance(name):
    bars = random_walk(60 * 24 * 40, seed=11, step=2.0)  # 40 days of minutes
    cut = bars.index[0] + pd.Timedelta(days=25)
    full = Backtester(MES, initial_equity=1e6).run(
        build_mode(name, **CHEAP_PARAMS.get(name, {})), bars
    )
    part = Backtester(MES, initial_equity=1e6).run(
        build_mode(name, **CHEAP_PARAMS.get(name, {})), bars[bars.index < cut]
    )
    # Trades closed well before the cut cannot depend on data after it.
    safe = cut - pd.Timedelta(days=2)
    a = full.trades[full.trades.exit_time < safe].drop(columns=["exit_idx", "entry_idx"])
    b = part.trades[part.trades.exit_time < safe].drop(columns=["exit_idx", "entry_idx"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    if name != "no_trade":
        assert len(a) > 0, "test data should produce trades"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_prepared_features_are_causal(name):
    from trader.data.resample import resample_bars

    bars = random_walk(60 * 24 * 40, seed=5)
    mode = build_mode(name)
    htf = resample_bars(bars, mode.timeframe).drop(columns="end")
    mode.prepare(htf)
    full = {k: v.copy() for k, v in mode.feat.items()}
    k = len(htf) * 2 // 3
    mode.prepare(htf.iloc[:k])
    for key, arr in mode.feat.items():
        pd.testing.assert_series_equal(pd.Series(arr), pd.Series(full[key][:k]))
