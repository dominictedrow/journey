"""Phase 3 step 5b: the feature build, its writes and the matrix reader (doc 03 sections 8, 10).

A tiny point-in-time universe in a throwaway DB: 8 index members in two sectors, SPY/TLT, VIX and
VIX3M, BTC and ETH trading 7 days a week, daily BAA10Y / DGS10 / DGS3MO vintages. The central check
is that a daily run at T (inputs cut at T) writes exactly the rows a longer backfill writes for T
(inputs cut weeks later).
"""

from __future__ import annotations

import dataclasses
import math
import secrets
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest

from qt.common.calendar import TradingCalendar
from qt.db.conn import connect
from qt.db.migrate import migrate
from qt.features import build, catalog, matrix, registry
from qt.features.inputs import ET
from qt.features.regime import hmm

pytestmark = pytest.mark.db

CAL = TradingCalendar()
START = date(2022, 1, 3)
SINCE, UNTIL = date(2023, 5, 1), date(2023, 7, 14)  # month boundaries inside
MID = date(2023, 7, 6)


def with_features(settings, **sections):
    f = settings.config.features
    upd = {}
    for k, v in sections.items():
        cur = getattr(f, k)
        if isinstance(v, dict):
            sub = {}
            for kk, vv in v.items():
                part = getattr(cur, kk)
                sub[kk] = part.model_copy(update=vv) if isinstance(vv, dict) else vv
            upd[k] = cur.model_copy(update=sub)
        else:
            upd[k] = v
    feats = f.model_copy(update=upd)
    return dataclasses.replace(settings, config=settings.config.model_copy(
        update={"features": feats}))  # fmt: skip


def _seed(c: psycopg.Connection) -> dict[str, int]:
    rng = np.random.default_rng(5)
    ids = {}

    def asset(tk, cls, sector=None):
        ids[tk] = c.execute(
            "INSERT INTO assets (ticker, asset_class, gics_sector) VALUES (%s, %s, %s) "
            "RETURNING asset_id", (tk, cls, sector),
        ).fetchone()[0]  # fmt: skip

    for i in range(8):
        asset(f"EQ{i}", "equity", "Tech" if i < 5 else "Energy")
    asset("SPY", "etf")
    asset("TLT", "etf")
    asset("^VIX", "index")
    asset("^VIX3M", "index")
    asset("BTC/USD", "crypto")
    asset("ETH/USD", "crypto")
    sess = [pd.Timestamp(x).date() for x in CAL.cal.sessions
            if START <= pd.Timestamp(x).date() <= date(2023, 8, 31)]  # fmt: skip
    days = [START + timedelta(days=i) for i in range((date(2023, 8, 31) - START).days + 1)]
    rows = []
    for tk, aid in ids.items():
        crypto = "/" in tk
        dd = days if crypto else sess
        src = "yfinance" if tk.startswith("^") else "alpaca"
        lvl = 20.0 if tk.startswith("^") else 100.0
        x = lvl * np.exp(np.cumsum(rng.normal(0, 0.02 if crypto else 0.012, len(dd))))
        for d, px in zip(dd, x, strict=True):
            if crypto:
                ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
                rel = ts + timedelta(days=1, minutes=5)
            else:
                ts = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(UTC)
                rel = datetime(d.year, d.month, d.day, 16, 20, tzinfo=ET).astimezone(UTC)
            rows.append((aid, ts, px, px * 1.01, px * 0.99, px, 1e6 + rng.random() * 1e5, src,
                         rel))  # fmt: skip
    with c.cursor() as cur:
        cur.executemany(
            "INSERT INTO raw_prices (asset_id, ts, open, high, low, close, volume, source, "
            "released_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows,
        )  # fmt: skip
        mac = []
        for i, d in enumerate(sess):
            nxt = sess[i + 1] if i + 1 < len(sess) else d + timedelta(days=1)
            r = datetime(nxt.year, nxt.month, nxt.day, 8, tzinfo=ET).astimezone(UTC)
            for sid, v in (("BAA10Y", 2 + 0.01 * math.sin(i / 9)), ("DGS10", 3 + 0.002 * i % 1),
                           ("DGS3MO", 4 + 0.001 * (i % 30))):  # fmt: skip
                mac.append((sid, d, v + rng.normal(0, 0.01), nxt, r, "alfred"))
        cur.executemany(
            "INSERT INTO raw_macro (series_id, period, value, vintage, released_at, source) "
            "VALUES (%s, %s, %s, %s, %s, %s)", mac,
        )  # fmt: skip
        snap = datetime(2021, 12, 31, tzinfo=UTC)
        cur.executemany(
            "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, asset_id, "
            "released_at) VALUES ('SPX', %s, %s, 'wikipedia', %s, %s)",
            [(snap, f"EQ{i}", ids[f"EQ{i}"], snap) for i in range(7)],  # EQ7 is not a member
        )  # fmt: skip
        # EQ6 leaves the index on 2023-07-01 (snapshot released then)
        snap2 = datetime(2023, 7, 1, tzinfo=UTC)
        cur.executemany(
            "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, asset_id, "
            "released_at) VALUES ('SPX', %s, %s, 'wikipedia', %s, %s)",
            [(snap2, f"EQ{i}", ids[f"EQ{i}"], snap2) for i in range(6)],
        )  # fmt: skip
    return ids


def _db(settings):
    name = "qt_test_" + secrets.token_hex(4)
    with connect(settings, "primary", "postgres", autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    with connect(settings, "primary", name, autocommit=True) as c:
        migrate(c)
        ids = _seed(c)
    return name, ids


def _drop(settings, name):
    with connect(settings, "primary", "postgres", autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="module")
def cfg_settings():
    from qt.common import config

    s = config.load(profile="dev")
    try:
        with connect(s, "primary", "postgres", autocommit=True):
            pass
    except Exception:
        pytest.skip("primary DB not reachable: run `docker compose up -d`")
    return s


@pytest.fixture(scope="module")
def built(cfg_settings):
    """DB A: backfill SINCE..UNTIL (inputs cut at UNTIL). DB B: daily run at MID only."""
    a, ids = _db(cfg_settings)
    b, _ = _db(cfg_settings)
    try:
        with connect(cfg_settings, "primary", a) as c:
            registry.sync(c)
            res = build.build(c, cfg_settings, SINCE, UNTIL, run_id=None, compress=True)
            c.commit()
        with connect(cfg_settings, "primary", b) as c:
            res_b = build.build(c, cfg_settings, MID, MID, run_id=None, compress=False)
            c.commit()
        yield {"a": a, "b": b, "ids": ids, "res": res, "res_b": res_b}
    finally:
        _drop(cfg_settings, a)
        _drop(cfg_settings, b)


def _rows(settings, db, t):
    with connect(settings, "primary", db) as c:
        return {(r[0], r[1], r[2]): r[3] for r in c.execute(
            "SELECT asset_id, feature_name, feature_version, value FROM features WHERE asof = %s",
            (t,)).fetchall()}  # fmt: skip


def _t(d):
    return build.decision_time(d, "18:00")


def test_daily_equals_backfill(built, cfg_settings):
    t = _t(MID)
    a = _rows(cfg_settings, built["a"], t)
    b = _rows(cfg_settings, built["b"], t)
    assert len(a) > 500 and a.keys() == b.keys()
    diff = [k for k in a if not (a[k] == b[k] or (a[k] is None and b[k] is None)
                                 or (a[k] is not None and b[k] is not None
                                     and math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12)))]
    assert diff == []  # fmt: skip
    # and enough of them are real numbers
    assert sum(v is not None for v in a.values()) > 0.5 * len(a)
    # context-series features are live (SPY bench, DXY-free here, DGS10 from macro)
    ids = built["ids"]
    for name in ("beta_63d", "corr_spy_63d", "resid_vol_63d", "corr_y10_63d"):
        assert a[(ids["EQ0"], name, 1)] is not None, name
    assert a[(ids["BTC/USD"], "beta_63d", 1)] is not None  # coins: bench BTC


def test_members_only_and_classes(built, cfg_settings):
    ids = built["ids"]
    t = _t(MID)  # after EQ6 left the index
    a = _rows(cfg_settings, built["a"], t)
    assets = {k[0] for k in a}
    assert {ids[f"EQ{i}"] for i in range(6)} <= assets
    assert ids["EQ6"] not in assets and ids["EQ7"] not in assets  # left / never a member
    assert {ids["SPY"], ids["BTC/USD"]} <= assets
    assert ids["^VIX"] not in assets  # index levels are context, not assets
    before = {k[0] for k in _rows(cfg_settings, built["a"], _t(date(2023, 6, 29)))}
    assert ids["EQ6"] in before
    # ranks only for equities; ETF z only for ETFs/coins
    names_eq = {k[1] for k in a if k[0] == ids["EQ0"]}
    names_etf = {k[1] for k in a if k[0] == ids["SPY"]}
    assert "ret_21d_xs" in names_eq and "ret_21d_xs" not in names_etf
    assert "ret_21d_tsz252" in names_etf and "ret_21d_tsz252" not in names_eq
    assert "earnings_yield" in names_eq and "earnings_yield" not in names_etf
    # market rows: asset_id NULL, incl. regime and z
    mk = {k[1] for k in a if k[0] is None}
    assert {"stress_score", "hmm_p_calm", "quad_goldilocks", "dsr_baa10y",
            "credit_baa10y_tsz252"} <= mk  # fmt: skip


def test_weekend_rows_crypto_only(built, cfg_settings):
    ids = built["ids"]
    sat = _t(date(2023, 7, 8))
    a = _rows(cfg_settings, built["a"], sat)
    assets = {k[0] for k in a if k[0] is not None}
    assert assets == {ids["BTC/USD"], ids["ETH/USD"]}
    groups = {s.name: s.group for s in registry.all_specs()}
    mk = {groups[k[1]] for k in a if k[0] is None}
    assert mk <= set(cfg_settings.config.features.build.nonsession_market_groups)
    with connect(cfg_settings, "primary", built["a"]) as c:
        assert c.execute("SELECT count(*) FROM regime WHERE asof = %s", (sat,)).fetchone()[0] == 0


def test_rerun_writes_nothing_and_table_is_append_only(built, cfg_settings):
    with connect(cfg_settings, "primary", built["b"]) as c:
        res = build.build(c, cfg_settings, MID, MID, compress=False)
        c.commit()
        assert res.rows_offered > 0 and res.rows_written == 0
        with pytest.raises(psycopg.Error):
            c.execute("UPDATE features SET value = 0 WHERE asof = %s", (_t(MID),))
        c.rollback()
        with pytest.raises(psycopg.Error):
            c.execute("DELETE FROM regime")
        c.rollback()


def test_regime_rows_and_compression(built, cfg_settings):
    res = built["res"]
    n_sess = sum(CAL.is_session(SINCE + timedelta(days=i))
                 for i in range((UNTIL - SINCE).days + 1))  # fmt: skip
    assert res.regime_rows == n_sess
    with connect(cfg_settings, "primary", built["a"]) as c:
        n = c.execute("SELECT count(*) FROM regime").fetchone()[0]
        assert n == n_sess
        comp = c.execute(
            "SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = "
            "'features' AND is_compressed").fetchone()[0]  # fmt: skip
        assert comp >= 1  # the chunk before the July boundary was compressed mid-run
        seg = c.execute("SELECT segmentby FROM timescaledb_information.hypertable_compression_"
                        "settings WHERE hypertable = 'features'::regclass").fetchone()  # fmt: skip
        assert seg is not None and "feature_version" in str(seg[0])
    assert res.rows_written == res.rows_offered
    assert set(res.nan_rate) >= {"price_returns", "regime_stress"}


def test_matrix_reads_back(built, cfg_settings):
    ids = built["ids"]
    cat = catalog.register_all(cfg_settings.config.features)
    keys = cat.keys()
    with connect(cfg_settings, "primary", built["a"]) as c:
        mx = matrix.features_asof(c, _t(MID) + timedelta(hours=1), keys)
        # Saturday: stocks carry Friday's values, coins have their own Saturday rows
        sat = matrix.features_asof(c, _t(date(2023, 7, 8)), keys)
    raw = _rows(cfg_settings, built["a"], _t(MID))
    assert mx.assets.loc[ids["EQ0"], "ret_21d"] == pytest.approx(raw[(ids["EQ0"], "ret_21d", 1)])
    assert mx.market["stress_score"] == pytest.approx(raw[(None, "stress_score", 1)])
    fri = _rows(cfg_settings, built["a"], _t(date(2023, 7, 7)))
    s8 = _rows(cfg_settings, built["a"], _t(date(2023, 7, 8)))
    assert sat.assets.loc[ids["EQ0"], "ret_21d"] == pytest.approx(fri[(ids["EQ0"], "ret_21d", 1)])
    assert sat.assets.loc[ids["BTC/USD"], "ret_21d"] == pytest.approx(
        s8[(ids["BTC/USD"], "ret_21d", 1)])
    two = matrix.features_asof(c2 := connect(cfg_settings, "primary", built["a"]), _t(MID),
                               [("infl_mom_core_cpi", 1), ("infl_mom_core_cpi", 2)])  # fmt: skip
    c2.close()
    assert set(two.market.index) <= {"infl_mom_core_cpi@v1", "infl_mom_core_cpi@v2"}


# ---------------------------------------------------------------- unit pieces
def test_resolve_matches_compute_asof():
    rng = np.random.default_rng(0)
    n = 50
    vals = pd.DataFrame({("a", 1): rng.normal(size=n), ("b", 1): rng.normal(size=n)})
    ready = pd.DataFrame({("a", 1): np.arange(n) * 10, ("b", 1): np.arange(n) * 10 + 15})
    ready.iloc[20, 0] = 1000  # a late row: earlier rows win until it is ready
    t = np.array([5, 9, 200, 205, 999, 1000, 10_000], dtype="int64")
    got = build.resolve(vals, ready, t)
    for i, tt in enumerate(t):
        for j, k in enumerate(vals.columns):
            ok = np.flatnonzero(ready[k].to_numpy() <= tt)
            want = vals[k].iloc[ok[-1]] if len(ok) else np.nan
            assert (np.isnan(want) and np.isnan(got[i, j])) or got[i, j] == want


def test_grid_and_decision_time():
    g = build.Grid.build(date(2023, 7, 1), date(2023, 7, 5), "18:00", CAL)
    assert g.days[0] == date(2023, 7, 1) and len(g.days) == 5
    assert g.session.tolist() == [False, False, True, False, True]  # Jul 4 holiday
    assert pd.Timestamp(g.t[2], tz=UTC) == pd.Timestamp("2023-07-03 22:00", tz=UTC)


# ---------------------------------------------------------------- HMM refit job
def test_ensure_fits_and_stability_alert(cfg_settings, tmp_path, monkeypatch):
    s = with_features(cfg_settings, regime={"hmm": {"min_weeks": 40, "n_seeds": 2,
                                                    "first_obs": date(2022, 2, 1),
                                                    "model_dir": str(tmp_path)}})  # fmt: skip
    name, _ = _db(s)
    try:
        sent = []
        calls = iter([0.05, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        monkeypatch.setattr(hmm, "relabel_change", lambda *a, **k: next(calls))
        with connect(s, "primary", name) as c:
            until = datetime(2023, 4, 3, tzinfo=UTC)
            # the default catalog: the overridden HMM numbers would be a new definition
            cat = catalog.register_all(cfg_settings.config.features)
            inp = build.load_inputs(c, s, until, cat)
            fits, new = build.ensure_fits(c, s, inp, until,
                                          alert=lambda *a: sent.append(a) or True)  # fmt: skip
            c.commit()
            assert len(new) >= 3 and [x["fit_date"].weekday() for x in new] == [6] * len(new)
            assert new[0]["relabel_change"] is None  # first fit: nothing to compare with
            assert [x["flagged"] for x in new[1:3]] == [False, True]
            assert len(sent) == 1 and sent[0][0] == "warn" and "90%" in sent[0][2]
            rows = c.execute("SELECT fit_date, model_path FROM hmm_fits ORDER BY 1").fetchall()
            assert len(rows) == len(new)
            # rerun: nothing new; the stored files load back identically
            _, again = build.ensure_fits(c, s, inp, until)
            assert again == []
            loaded = build.load_fits(c, 1, until)
            assert [f.fit_date for f in loaded] == [f.fit_date for f in fits]
            np.testing.assert_array_equal(loaded[-1].means, fits[-1].means)
            p = build.stability_report(c, s, date(2023, 4, 3))
            text = p.read_text()
            assert "**ALERT**" in text and f"{len(new)} fits, 1 above" in text
            p.unlink()
    finally:
        _drop(s, name)
