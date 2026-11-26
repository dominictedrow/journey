"""Phase 3 step 1: registry, PIT inputs, bar adjustment (component doc 03 sections 3, 8, 11)."""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.common.config import AdjustCfg, FeatureBarsCfg
from qt.db.conn import connect
from qt.features import registry
from qt.features.adjust import adjust
from qt.features.inputs import MacroState, load_bars, map_actions_to_assets, with_name_changes

FEATURES_DIR = Path(__file__).resolve().parents[1] / "src" / "qt" / "features"
RAW_TABLES = {
    "raw_prices",
    "raw_macro",
    "raw_fundamentals",
    "raw_events",
    "raw_metrics",
    "raw_crypto_derivs",
    "index_membership",
    "articles",
}
CFG = AdjustCfg(ignore_kinds=["ca_spin_offs"])


def ts(d: date, hh: int = 21) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, tzinfo=UTC))


# ---------------------------------------------------------------- registry
@pytest.fixture
def clean_registry():
    saved = registry._clear_for_tests()
    try:
        yield
    finally:
        registry._restore_for_tests(saved)


def _register(name: str, version: int, formula: str = "x") -> None:
    registry.feature(name, version, kind="market", group="g", formula=formula, window=21)(
        lambda: None
    )


def test_registry_current_and_bad_names(clean_registry):
    _register("vix_z", 1)
    _register("vix_z", 2)
    _register("curve", 1)
    assert [(s.name, s.version) for s in registry.current_specs()] == [("curve", 1), ("vix_z", 2)]
    with pytest.raises(ValueError):
        _register("Bad-Name", 1)
    with pytest.raises(ValueError):
        _register("ok", 0)


def test_registry_sync_insert_mismatch_freeze(clean_registry, settings, db):
    with connect(settings, "primary", db, autocommit=True) as c:
        _register("vix_z", 1, "z of vix")
        assert registry.sync(c) == 1
        assert registry.sync(c) == 0
        # Same version, changed definition: refuse.
        registry._clear_for_tests()
        _register("vix_z", 1, "z of vix, winsorized")
        with pytest.raises(registry.RegistryError, match="without a version bump"):
            registry.sync(c)
        # Bump: v2 is inserted, v1 (unpinned, not current) is frozen.
        registry._clear_for_tests()
        _register("vix_z", 1, "z of vix")
        _register("vix_z", 2, "z of vix, winsorized")
        assert registry.sync(c) == 1
        rows = dict(
            c.execute("SELECT version, status FROM feature_registry ORDER BY version").fetchall()
        )
        assert rows == {1: "frozen", 2: "active"}
        assert registry.pinned_versions(c) == set()


def test_pinned_old_version_keeps_computing(clean_registry, settings, db):
    with connect(settings, "primary", db, autocommit=True) as c:
        c.execute(
            "CREATE TABLE model_feature_pins (feature_name text, feature_version int, active bool)"
        )
        c.execute("INSERT INTO model_feature_pins VALUES ('vix_z', 1, true)")
        _register("vix_z", 1)
        _register("vix_z", 2, "y")
        assert {s.key for s in registry.to_compute(c)} == {("vix_z", 1), ("vix_z", 2)}
        registry.sync(c)
        status = c.execute("SELECT status FROM feature_registry WHERE version = 1").fetchone()[0]
        assert status == "active"


# ---------------------------------------------------------------- adjustment
def _bars(closes: list[float], start: date = date(2026, 11, 16)) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "asset_id": 1,
            "date": days,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * len(closes),
            "released_at": [ts(d) for d in days],
        }
    )


def _action(kind, ex, released, **kw):
    base = {
        "external_id": f"{kind}-{ex}",
        "kind": kind,
        "symbol": "X",
        "ex_date": ex,
        "released_at": released,
        "old_rate": np.nan,
        "new_rate": np.nan,
        "rate": np.nan,
        "special": False,
        "asset_id": 1,
    }
    base.update(kw)
    return base


def test_split_adjusts_history_and_volume():
    b = _bars([800.0, 760.0, 95.0, 96.0])  # 8:1 split effective on the third bar
    ex = b["date"][2]
    acts = pd.DataFrame([_action("ca_forward_splits", ex, ts(ex, 5), old_rate=1, new_rate=8)])
    out = adjust(b, acts, ts(b["date"][3]), CFG)
    assert out["adj_close"].tolist() == [100.0, 95.0, 95.0, 96.0]
    assert out["adj_volume"].tolist() == [800.0, 800.0, 100.0, 100.0]
    assert out["close"].tolist() == [800.0, 760.0, 95.0, 96.0]  # raw kept


def test_dividend_total_return_factor():
    b = _bars([100.0, 100.0, 99.0])
    ex = b["date"][2]
    acts = pd.DataFrame([_action("ca_cash_dividends", ex, ts(ex, 5), rate=1.0)])
    out = adjust(b, acts, ts(ex), CFG)
    assert out["adj_close"].round(10).tolist() == [99.0, 99.0, 99.0]


def test_adjustment_pit_unreleased_or_future_action_ignored():
    b = _bars([800.0, 760.0, 95.0])
    ex = b["date"][2]
    split = _action("ca_forward_splits", ex, ts(ex, 5), old_rate=1, new_rate=8)
    # At T before the action's release (and before its ex date) nothing is adjusted.
    t_before = ts(b["date"][1])
    out = adjust(b[b["released_at"] <= t_before], pd.DataFrame([split]), t_before, CFG)
    assert out["adj_close"].tolist() == [800.0, 760.0]
    # Released late (after T): ignored at T even though the ex date has passed.
    late = dict(split, released_at=ts(ex) + pd.Timedelta(days=5))
    out = adjust(b, pd.DataFrame([late]), ts(ex), CFG)
    assert out["adj_close"].tolist() == [800.0, 760.0, 95.0]


def test_returns_invariant_to_later_actions():
    b = _bars([100.0, 102.0, 101.0, 50.0, 51.0])
    ex = b["date"][3]
    acts = pd.DataFrame([_action("ca_forward_splits", ex, ts(ex, 5), old_rate=1, new_rate=2)])
    t_early = ts(b["date"][2])
    early = adjust(b[b["released_at"] <= t_early], acts, t_early, CFG)
    late = adjust(b, acts, ts(b["date"][4]), CFG)
    r_early = np.log(early["adj_close"]).diff().dropna().to_numpy()
    r_late = np.log(late["adj_close"]).diff().dropna().to_numpy()[: len(r_early)]
    np.testing.assert_allclose(r_early, r_late)


def test_ignored_kind_and_oversized_dividend():
    b = _bars([10.0, 10.0])
    ex = b["date"][1]
    acts = pd.DataFrame(
        [
            _action("ca_spin_offs", ex, ts(ex, 5)),
            _action("ca_cash_dividends", ex, ts(ex, 5), rate=9.0),  # 90% of price: data error
        ]
    )
    assert adjust(b, acts, ts(ex), CFG)["adj_close"].tolist() == [10.0, 10.0]


# ---------------------------------------------------------------- action -> asset mapping
def test_map_actions_rename_and_reuse():
    tickers = pd.DataFrame(
        [
            (1, "FB", None, date(2022, 6, 9)),
            (1, "META", date(2022, 6, 9), None),
            (2, "SNDK", None, date(2016, 5, 12)),  # old SanDisk
            (3, "SNDK", None, None),  # new SanDisk (2025)
        ],
        columns=["asset_id", "symbol", "valid_from", "valid_to"],
    )
    assets = pd.DataFrame({"asset_id": [1, 2, 3, 4], "ticker": ["META", "SNDK", "SNDK", "SPY"]})
    bars = pd.DataFrame({"asset_id": [2, 3], "date": [date(2016, 3, 1), date(2026, 11, 19)]})
    acts = pd.DataFrame(
        [
            {"symbol": "FB", "ex_date": date(2020, 1, 2)},
            {"symbol": "META", "ex_date": date(2024, 1, 2)},
            {"symbol": "SNDK", "ex_date": date(2016, 3, 2)},  # both match; only old has bars
            {"symbol": "SNDK", "ex_date": date(2026, 11, 20)},  # old interval closed
            {"symbol": "SPY", "ex_date": date(2024, 1, 2)},  # no alias: falls back to ticker
            {"symbol": "ZZZZ", "ex_date": date(2024, 1, 2)},  # unknown: dropped
        ]
    )
    out = map_actions_to_assets(acts, tickers, assets, bars)
    assert out["asset_id"].tolist() == [1, 1, 2, 3, 4]


# ---------------------------------------------------------------- macro vintages
def test_macro_state_no_lookahead_and_forward_only():
    p = date(2026, 9, 1)
    rows = pd.DataFrame(
        {
            "series_id": ["CPI"] * 4,
            "period": [date(2026, 8, 1), p, p, p],
            "value": [1.0, 2.0, 2.5, None],
            "vintage": [
                date(2026, 9, 11),
                date(2026, 10, 15),
                date(2026, 11, 13),
                date(2026, 12, 1),
            ],
            "released_at": [
                ts(date(2026, 9, 11), 12),
                ts(date(2026, 10, 15), 12),
                ts(date(2026, 11, 13), 12),
                ts(date(2026, 12, 1), 12),
            ],
        }
    )
    st = MacroState(rows)
    st.advance(ts(date(2026, 10, 15), 11))  # one hour before the first September print
    assert st.series("CPI").to_dict() == {date(2026, 8, 1): 1.0}
    st.advance(ts(date(2026, 10, 15), 12))
    assert st.series("CPI")[p] == 2.0
    st.advance(ts(date(2026, 11, 20)))  # revision visible only after its release
    assert st.series("CPI")[p] == 2.5
    assert st.last_release("CPI") == ts(date(2026, 11, 13), 12)
    st.advance(ts(date(2026, 12, 2)))
    assert np.isnan(st.series("CPI")[p])  # the source's own missing marker, not a back-fill
    with pytest.raises(ValueError):
        st.advance(ts(date(2026, 1, 1)))


# ---------------------------------------------------------------- bars from the DB
def test_load_bars_source_priority_and_class_gate(settings, db):
    cfg = FeatureBarsCfg(source_priority=["alpaca", "yfinance"], yfinance_classes=["etf"])
    with connect(settings, "primary", db, autocommit=True) as c:
        etf = c.execute(
            "INSERT INTO assets (ticker, asset_class) VALUES ('SPY','etf') RETURNING asset_id"
        ).fetchone()[0]
        eq = c.execute(
            "INSERT INTO assets (ticker, asset_class) VALUES ('AAPL','equity') RETURNING asset_id"
        ).fetchone()[0]
        d1 = datetime(2026, 11, 19, 5, tzinfo=UTC)
        d2 = datetime(2026, 11, 20, 5, tzinfo=UTC)
        ins = (
            "INSERT INTO raw_prices (asset_id, ts, open, high, low, close, volume, source, "
            "released_at) VALUES (%s, %s, 1, 1, 1, %s, 1, %s, %s)"
        )
        for aid, t, close, src in [
            (etf, d1, 10.0, "alpaca"),
            (etf, d1, 11.0, "yfinance"),
            (etf, d2, 12.0, "yfinance"),
            (eq, d1, 20.0, "yfinance"),  # equity yfinance bar: not allowed
        ]:
            c.execute(ins, (aid, t, close, src, t + timedelta(hours=16)))
        out = load_bars(c, datetime(2026, 11, 21, tzinfo=UTC), cfg)
        got = {(r.asset_id, r.date): (r.close, r.source) for r in out.itertuples()}
        assert got == {
            (etf, date(2026, 11, 19)): (10.0, "alpaca"),
            (etf, date(2026, 11, 20)): (12.0, "yfinance"),
        }
        # Cut at a time before the second bar's release.
        early = load_bars(c, datetime(2026, 11, 20, 12, tzinfo=UTC), cfg)
        assert len(early) == 1


# ---------------------------------------------------------------- lint
def test_features_read_raw_tables_only_via_inputs():
    offenders = []
    for path in FEATURES_DIR.rglob("*.py"):
        if path.name.startswith("._") or path.name == "inputs.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                words = set(node.value.replace(",", " ").replace(".", " ").split())
                if words & RAW_TABLES:
                    offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module == "qt.db.pit":
                offenders.append(f"{path.name}:{node.lineno} imports qt.db.pit")
    assert offenders == [], offenders


def test_split_rejected_when_prices_disagree():
    # CNX 2017: Alpaca lists a "1:0.125 reverse split" that was really a spin-off (-16%).
    b = _bars([16.16, 13.50, 13.95])
    ex = b["date"][1]
    acts = pd.DataFrame([_action("ca_reverse_splits", ex, ts(ex, 5), old_rate=1.0, new_rate=0.125)])
    assert adjust(b, acts, ts(b["date"][2]), CFG)["adj_close"].tolist() == [16.16, 13.50, 13.95]
    # A split with no bar near its ex date (symbol now held by someone else) is rejected too.
    far = date(2027, 1, 1)
    acts = pd.DataFrame([_action("ca_reverse_splits", far, ts(far, 5), old_rate=10, new_rate=1)])
    assert adjust(b, acts, ts(far), CFG)["adj_close"].tolist() == [16.16, 13.50, 13.95]


def test_manual_dividend_skips_size_guard():
    b = _bars([123.66, 22.19])  # KDP 2018: $103.75 special = 84% of the price
    ex = b["date"][1]
    act = dict(_action("ca_cash_dividends", ex, ts(ex, 5), rate=103.75, special=True), manual=True)
    out = adjust(b, pd.DataFrame([act]), ts(ex), CFG)
    assert out["adj_close"].round(2).tolist() == [19.91, 22.19]


def test_name_changes_extend_ticker_history():
    tickers = pd.DataFrame(
        [(251, "EXE", None, None)], columns=["asset_id", "symbol", "valid_from", "valid_to"]
    )
    assets = pd.DataFrame({"asset_id": [251], "ticker": ["EXE"]})
    nc = pd.DataFrame(
        [
            {
                "kind": "ca_name_changes",
                "old_symbol": "CHK",
                "new_symbol": "EXE",
                "ex_date": date(2024, 10, 2),
            }
        ]
    )
    out = with_name_changes(tickers, nc, assets)
    assert (251, "CHK", None, date(2024, 10, 2)) in set(map(tuple, out.itertuples(index=False)))
    acts = pd.DataFrame([{"symbol": "CHK", "ex_date": date(2020, 4, 15)}])
    bars = pd.DataFrame({"asset_id": [251], "date": [date(2020, 4, 14)]})
    assert map_actions_to_assets(acts, out, assets, bars)["asset_id"].tolist() == [251]


def test_load_bars_drops_zero_volume_filler(settings, db):
    cfg = FeatureBarsCfg(
        source_priority=["alpaca"], yfinance_classes=[], drop_zero_volume_classes=["equity"]
    )
    with connect(settings, "primary", db, autocommit=True) as c:
        aid = c.execute(
            "INSERT INTO assets (ticker, asset_class) VALUES ('OLD','equity') RETURNING asset_id"
        ).fetchone()[0]
        for day, vol in ((19, 100), (20, 0)):
            t = datetime(2026, 11, day, 5, tzinfo=UTC)
            c.execute(
                "INSERT INTO raw_prices (asset_id, ts, close, volume, source, released_at) "
                "VALUES (%s, %s, 1, %s, 'alpaca', %s)",
                (aid, t, vol, t + timedelta(hours=16)),
            )
        out = load_bars(c, datetime(2026, 11, 21, tzinfo=UTC), cfg)
        assert out["date"].tolist() == [date(2026, 11, 19)]
