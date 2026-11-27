"""Phase 3 step 3: cross-sectional ranks and composites (component doc 03 sections 5.2 and 11)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.features import cross_section as xs
from qt.features import fundamentals, price
from qt.features.inputs import members_at

CFG = config.load().config.features
BASE = price.register(CFG.price) + fundamentals.register(CFG.fundamentals)
SPECS = xs.register(CFG.cross_section, BASE)
RANKED = sorted({s.name for s in BASE if s.group in CFG.cross_section.rank_groups})


def _panel(seed=1, n=40):
    rng = np.random.default_rng(seed)
    vals = pd.DataFrame(rng.normal(size=(n, len(RANKED))), columns=RANKED,
                        index=pd.Index(range(100, 100 + n), name="asset_id"))  # fmt: skip
    vals.iloc[::7, 0] = np.nan  # some NaN
    vals.iloc[1:6, 1] = 0.25  # ties
    sectors = pd.Series(["A"] * 20 + ["B"] * 17 + ["C"] * 3, index=vals.index)  # C is too small
    return xs.Panel(vals, sectors, CFG.cross_section)


P = _panel()


def naive_rank(x: dict) -> dict:
    """Average-rank percentile by counting, NaN excluded."""
    valid = {k: v for k, v in x.items() if not math.isnan(v)}
    n = len(valid)
    out = {}
    for k, v in x.items():
        if math.isnan(v):
            out[k] = math.nan
            continue
        below = sum(1 for w in valid.values() if w < v)
        equal = sum(1 for w in valid.values() if w == v)
        avg = below + (equal + 1) / 2  # 1-based average rank
        out[k] = 0.5 if n == 1 else (avg - 1) / (n - 1)
    return out


def hand(spec, panel):
    name = spec.name
    v = panel.values
    if spec.group == xs.GROUP_RANK:
        return naive_rank(v[name[: -len("_xs")]].to_dict())
    if spec.group == xs.GROUP_SECTOR:
        base = name[: -len("_xs_sec")]
        out = {}
        for sec in panel.sectors.unique():
            ids = panel.sectors.index[panel.sectors == sec]
            col = v.loc[ids, base]
            if col.notna().sum() >= CFG.cross_section.min_sector_size:
                out.update(naive_rank(col.to_dict()))
            else:
                out.update(dict.fromkeys(ids, math.nan))
        return out
    parts = CFG.cross_section.composites[name[len("comp_") :]]
    ranks = {c: naive_rank(v[c.lstrip("-")].to_dict()) for c in parts}
    out = {}
    for aid in v.index:
        got = []
        for c in parts:
            r = ranks[c][aid]
            if not math.isnan(r):
                got.append(1 - r if c.startswith("-") else r)
        out[aid] = (sum(got) / len(got) if len(got) >= CFG.cross_section.min_components
                    else math.nan)  # fmt: skip
    return out


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_xs_hand_calcs(spec):
    got = spec.fn(P)
    want = hand(spec, P)
    for aid, w in want.items():
        g = float(got.loc[aid])
        assert (math.isnan(g) and math.isnan(w)) or math.isclose(g, w, abs_tol=1e-12), (
            spec.name,
            aid,
            g,
            w,
        )


def test_ranks_nan_ties_and_small_sectors():
    col = RANKED[1]
    r = P.rank(col)
    assert r.min() == 0.0 and r.max() == 1.0
    tied = r.iloc[1:6]
    assert tied.nunique() == 1  # ties share the average rank
    r0 = P.rank(RANKED[0])
    assert r0.iloc[::7].isna().all() and r0.notna().sum() == P.values[RANKED[0]].notna().sum()
    sec = P.sector_rank(col)
    assert sec[P.sectors == "C"].isna().all()  # 3 members < min_sector_size
    a = P.sectors == "A"
    assert sec[a].min() == 0.0 and sec[a].max() == 1.0  # ranked only among sector A


def test_composite_needs_min_components():
    v = P.values.copy()
    v.loc[100, ["earnings_yield", "fcf_yield"]] = np.nan  # value keeps only ebitda_ev
    v.loc[101, ["earnings_yield"]] = np.nan
    p = xs.Panel(v, P.sectors, CFG.cross_section)
    comp = p.composite(CFG.cross_section.composites["value"])
    assert math.isnan(comp.loc[100]) and not math.isnan(comp.loc[101])


def test_negative_components_invert():
    v = pd.DataFrame({"rvol_252d": [0.1, 0.2, 0.3], "beta_252d": [0.5, 1.0, 1.5]},
                     index=[1, 2, 3])  # fmt: skip
    p = xs.Panel(v, pd.Series(["A"] * 3, index=v.index), CFG.cross_section)
    comp = p.composite(["-rvol_252d", "-beta_252d"])
    assert list(comp) == [1.0, 0.5, 0.0]  # least volatile ranks best


def test_every_ranked_feature_and_composite_registered():
    names = {s.name for s in SPECS}
    for b in RANKED:
        assert {f"{b}_xs", f"{b}_xs_sec"} <= names
    assert {f"comp_{c}" for c in CFG.cross_section.composites} <= names
    assert not any(s.name.startswith("earn_") for s in SPECS)  # calendar facts aren't ranked


def test_ranks_take_the_base_version_and_composites_count_bumps():
    v = {s.name: s.version for s in SPECS}
    assert v["earnings_yield_xs"] == v["earnings_yield_xs_sec"] == 2  # base v2 (D-045)
    assert v["roic_xs"] == 1
    assert v["comp_value"] == 4  # 1 + three components at v2
    assert v["comp_quality"] == 1


def test_members_as_of():
    def ts(d):
        return pd.Timestamp(datetime.fromisoformat(d).replace(tzinfo=UTC))

    m = pd.DataFrame({
        "snapshot_at": [ts("2024-01-07")] * 3 + [ts("2024-01-14")] * 2,
        "ticker": ["A", "B", "C", "A", "D"],
        "source": "wikipedia",
        "asset_id": [1, 2, 3, 1, 4],
        "released_at": [ts("2024-01-07")] * 3 + [ts("2024-01-14")] * 2,
    })  # fmt: skip
    assert members_at(m, ts("2024-01-10")) == {1, 2, 3}
    assert members_at(m, ts("2024-01-14")) == {1, 4}
    assert members_at(m, ts("2024-01-01")) == set()
