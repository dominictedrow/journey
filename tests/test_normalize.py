"""Phase 3 step 5a: normalization, masks and staleness (component doc 03 sections 7 and 11)."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import TradingCalendar
from qt.features import market, normalize, price, registry
from qt.features.inputs import MacroState
from qt.features.market import MarketState

FCFG = config.load().config.features
NCFG = FCFG.normalize
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")


def _naive_z(x, i, w, cap, frac):
    if not math.isfinite(x[i]):
        return math.nan
    win = [v for v in x[max(0, i - w + 1) : i + 1] if math.isfinite(v)]
    if len(win) < max(math.ceil(frac * w), 3):
        return math.nan
    sd = float(np.std(win, ddof=1))
    if sd == 0:
        return math.nan
    return float(np.clip((x[i] - np.mean(win)) / sd, -cap, cap))


def test_rolling_z_hand_calc():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 400)
    x[[30, 31, 260]] = np.nan
    x[300] = 50.0  # winsorized
    w, cap, frac = 50, 4.0, 0.9
    got = normalize.rolling_z(x, w, cap, frac)
    want = [_naive_z(x, i, w, cap, frac) for i in range(len(x))]
    np.testing.assert_allclose(got, want, rtol=1e-10, equal_nan=True)
    assert got[300] == cap
    assert np.isnan(got[: math.ceil(frac * w) - 1]).all()
    assert normalize.z_last(x[:301], w, cap, frac) == cap
    assert normalize.z_last(x[:150], w, cap, frac) == pytest.approx(want[149])


def test_rolling_z_past_only():
    x = np.random.default_rng(1).normal(0, 1, 300)
    z = normalize.rolling_z(x, 252, 4.0, 0.9)
    y = x.copy()
    y[260:] += 100
    np.testing.assert_array_equal(normalize.rolling_z(y, 252, 4.0, 0.9)[:260], z[:260])


def test_constant_window_is_nan():
    assert np.isnan(normalize.rolling_z(np.ones(300), 252, 4.0, 0.9)).all()


@pytest.fixture
def clean_registry():
    saved = registry._clear_for_tests()
    yield
    registry._restore_for_tests(saved)


def test_register_ts_z_selects_groups(clean_registry):
    mspecs = market.register(FCFG.market)
    pspecs = price.register(FCFG.price)
    z = normalize.register_ts_z(mspecs + pspecs, NCFG, NCFG.market_groups + NCFG.asset_groups)
    names = {s.name for s in z}
    assert "credit_baa10y_tsz252" in names and "ret_21d_tsz252" in names
    assert "earn_days_next_tsz252" not in names  # earnings group is not normalized
    assert all(s.group in (normalize.G_TSZ_MARKET, normalize.G_TSZ_ASSET) for s in z)
    s = registry.get("credit_baa10y_tsz252", 1)
    assert normalize.base_key(s) == ("credit_baa10y", 1) and s.kind == "market"
    assert "+-4" in s.formula and "252" in s.formula


def test_asset_ts_z_values_and_ready(clean_registry):
    base = registry.feature("x_feat", 1, kind="asset", group="price_returns", formula="x")(
        lambda f: None)  # fmt: skip
    (spec,) = normalize.register_ts_z([registry.get("x_feat", 1)], NCFG, ["price_returns"])
    del base
    n = 400
    rng = np.random.default_rng(2)
    vals = rng.normal(0, 1, n)
    vals[5] = np.nan
    ready = np.arange(n, dtype="int64") * 10
    ready[100] = 10_000  # this row became usable late: z rows whose window holds it wait
    ready[5] = 99_999  # a NaN row's ready time doesn't hold anything back
    idx = pd.Index(range(n))
    cols = pd.Index([("x_feat", 1)], tupleize_cols=False)
    v = pd.DataFrame(vals[:, None], index=idx, columns=cols)
    r = pd.DataFrame(ready[:, None], index=idx, columns=cols)
    zv, zr = normalize.asset_ts_z(v, r, [spec])
    want = normalize.rolling_z(vals, 252, 4.0, 0.9)
    np.testing.assert_allclose(zv[spec.key].to_numpy(), want, equal_nan=True)
    zr = zr[spec.key].to_numpy()
    assert zr[99] == 990 and zr[100] == 10_000 and zr[100 + 251] == 10_000
    assert zr[100 + 252] == 3520  # window no longer holds row 100
    assert zr[10] == 100


def _frame(n):
    rng = np.random.default_rng(3)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    d = np.array(SESSIONS[-n - 10 : -10], dtype="datetime64[D]")
    return price.PriceFrame(
        asset_class="equity", dates=d, open=c, high=c * 1.01, low=c * 0.99, close=c,
        volume=np.full(n, 1e6), close_raw=c, volume_raw=np.full(n, 1e6),
        ready_ns=np.arange(n, dtype="int64"), ctx={}, ctx_ready_ns={}, asof_ns=10**19,
        earnings=pd.DataFrame(), sessions=SESSIONS, cfg=FCFG.price,
    )  # fmt: skip


def test_new_listing_masks(clean_registry):
    specs = {s.name: s for s in normalize.register_masks(NCFG)}
    f = _frame(30)  # a 30-day-old asset
    hist = specs["history_days"].fn(f)
    assert hist.tolist() == list(range(1, 31))
    ok21 = specs["hist_ok_21"].fn(f)
    assert ok21[-1] == 1.0 and ok21[20] == 0.0 and ok21[21] == 1.0
    for n in (63, 126, 252):
        assert (specs[f"hist_ok_{n}"].fn(f) == 0).all()
    # and the matching price features agree: 21-day set, 63/252 NaN
    pspecs = price.register(FCFG.price)
    vals, _ = price.compute_series(f, [s for s in pspecs if s.name in ("ret_21d", "ret_63d",
                                                                        "rvol_252d")])
    assert math.isfinite(vals[("ret_21d", 1)].iloc[-1])
    assert vals[("ret_63d", 1)].isna().all() and vals[("rvol_252d", 1)].isna().all()


# ---------------------------------------------------------------- staleness
def _macro_rows(rows):
    df = pd.DataFrame(rows, columns=["series_id", "period", "value", "vintage", "released_at"])
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    return df


def _ts(y, mo, d, hh=13):
    return datetime(y, mo, d, hh, tzinfo=UTC)


def _mstate(macro, t):
    ms = MacroState(macro)
    ms.advance(t)
    return MarketState(asof=pd.Timestamp(t), macro=ms, prices={}, sessions=SESSIONS,
                       cfg=FCFG.market)  # fmt: skip


def test_days_since_release_across_revision_and_ffill():
    rows = [
        ("UNRATE", date(2022, 1, 1), 4.0, date(2022, 2, 4), _ts(2022, 2, 4)),
        ("UNRATE", date(2022, 2, 1), 3.8, date(2022, 3, 4), _ts(2022, 3, 4)),
        # a revision of January, released later: not news about the newest period
        ("UNRATE", date(2022, 1, 1), 4.1, date(2022, 3, 20), _ts(2022, 3, 20)),
        ("UNRATE", date(2022, 3, 1), None, date(2022, 4, 1), _ts(2022, 4, 1)),  # missing value
        ("UNRATE", date(2022, 4, 1), 3.6, date(2022, 5, 6), _ts(2022, 5, 6)),
    ]
    macro = _macro_rows(rows)
    m = _mstate(macro, _ts(2022, 3, 10))
    assert normalize.days_since_release(m, "UNRATE") == pytest.approx(6.0)
    m = _mstate(macro, _ts(2022, 3, 25, 1))
    assert normalize.days_since_release(m, "UNRATE") == pytest.approx(20.5)
    assert m.mac("UNRATE").iloc[0] == 4.1  # the revision is visible, never back-filled earlier
    m = _mstate(macro, _ts(2022, 4, 10))  # March has no value: still counting from Feb's release
    assert normalize.days_since_release(m, "UNRATE") == pytest.approx(37.0)
    assert m.latest("UNRATE") == 3.8  # forward-filled until the next release
    m = _mstate(macro, _ts(2022, 5, 7))
    assert normalize.days_since_release(m, "UNRATE") == pytest.approx(1.0)
    assert math.isnan(normalize.days_since_release(m, "DGS10"))
    early = _mstate(macro, _ts(2022, 2, 1))
    assert math.isnan(normalize.days_since_release(early, "UNRATE"))


def test_staleness_registered_per_macro_series(clean_registry):
    mspecs = market.register(FCFG.market)
    st = normalize.register_staleness(mspecs, NCFG)
    names = {s.name for s in st}
    assert {"dsr_unrate", "dsr_baa10y", "dsr_dgs10", "dsr_walcl", "dsr_ir3tib01jpm156n"} <= names
    assert not any("px" in n or "breadth" in n for n in names)
    assert all(s.group == normalize.G_STALENESS and s.kind == "market" for s in st)
