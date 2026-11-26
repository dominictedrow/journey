"""Phase 3 step 3: fundamentals timeline and ratios (component doc 03 sections 5.2 and 11).

A synthetic company files the way real ones do: 10-Qs with 3-month and year-to-date income
statements, cash flow and D&A as year-to-date only, a 10-K with fiscal-year figures only (Q4 is
derived), prior-year comparatives in every filing, a restatement, a revenue tag switch and a
2-for-1 split after the share-count date. Hand values come straight from the generator's true
quarterly numbers (as known at each time), not from the code under test.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import TradingCalendar
from qt.features import fundamentals as fu
from qt.features import price

CFG = config.load().config.features
FC = CFG.fundamentals
SPECS = fu.register(FC)
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")

QEND = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
QSTART = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
YEARS = (2022, 2023, 2024, 2025)
RESTATED = ((2024, 1), "revenue", 1.10)  # 2024 Q1 revenue restated +10% in the 2025 Q1 10-Q
SPLIT_EX = date(2025, 6, 2)


def qend(y, k):
    return date(y, *QEND[k])


def qstart(y, k):
    return date(y, *QSTART[k])


def _truth():
    rng = np.random.default_rng(3)
    t = {}
    i = 0
    for y in YEARS:
        for k in (1, 2, 3, 4):
            rev = 1000 * 1.03**i * (1 + rng.normal(0, 0.02))
            cost = rev * (0.6 + rng.normal(0, 0.01))
            op = rev * 0.2
            pretax = op * 0.95
            t[(y, k)] = {
                "revenue": rev,
                "cost": cost,
                "gp": rev - cost,
                "op": op,
                "ni": pretax * 0.79,
                "pretax": pretax,
                "tax": pretax * 0.21,
                "cfo": op * 1.1,
                "capex": rev * 0.05,
                "da": rev * 0.04,
                "eps": pretax * 0.79 / 100.0,
            }
            i += 1
    return t


TRUTH = _truth()
ITEMS = {  # truth key -> (tag before 2024, tag from 2024, reported as)
    "revenue": ("us-gaap:SalesRevenueNet",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "q"),
    "gp": ("us-gaap:GrossProfit", "us-gaap:GrossProfit", "q"),
    "op": ("us-gaap:OperatingIncomeLoss", "us-gaap:OperatingIncomeLoss", "q"),
    "ni": ("us-gaap:NetIncomeLoss", "us-gaap:NetIncomeLoss", "q"),
    "eps": ("us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareDiluted", "q"),
    "tax": ("us-gaap:IncomeTaxExpenseBenefit", "us-gaap:IncomeTaxExpenseBenefit", "q"),
    "pretax": (FC.concepts["pretax_income"][0], FC.concepts["pretax_income"][0], "q"),
    "cfo": ("us-gaap:NetCashProvidedByUsedInOperatingActivities",) * 2 + ("ytd",),
    "capex": ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",) * 2 + ("ytd",),
    "da": ("us-gaap:DepreciationDepletionAndAmortization",) * 2 + ("ytd",),
}  # fmt: skip


def filing_time(y, k) -> pd.Timestamp:
    lag = 60 if k == 4 else 35
    d = qend(y, k) + timedelta(days=lag)
    return pd.Timestamp(datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC))


def restate_time() -> pd.Timestamp:
    return filing_time(2025, 1)


def true_q(y, k, item, at: pd.Timestamp) -> float:
    v = TRUTH[(y, k)][item]
    (ry, rk), ritem, mult = RESTATED
    if (y, k) == (ry, rk) and item == ritem and at >= restate_time():
        v *= mult
    return v


def _facts():
    rows = []

    def add(metric, period, value, acc, rel, unit="USD"):
        rows.append({"asset_id": 1, "fiscal_period": period, "metric": metric, "value": value,
                     "unit": unit, "accession": acc, "released_at": rel})  # fmt: skip

    for y in YEARS:
        for k in (1, 2, 3, 4):
            rel = filing_time(y, k)
            acc = f"acc-{y}-{k}"
            for yy in (y, y - 1):  # current and prior-year comparative
                if yy not in YEARS:
                    continue
                for item, (tag_old, tag_new, how) in ITEMS.items():
                    tag = tag_new if y >= 2024 else tag_old
                    unit = "USD/shares" if item == "eps" else "USD"
                    qv = [true_q(yy, j, item, rel) for j in (1, 2, 3, 4)]
                    s = qstart(yy, 1)
                    if k < 4:
                        if how == "q":
                            add(tag, f"{qstart(yy, k)}/{qend(yy, k)}", qv[k - 1], acc, rel, unit)
                        if k > 1 or how == "ytd":
                            add(tag, f"{s}/{qend(yy, k)}", sum(qv[:k]), acc, rel, unit)
                    else:
                        add(tag, f"{s}/{qend(yy, 4)}", sum(qv), acc, rel, unit)
            e = qend(y, k)
            add("us-gaap:StockholdersEquity", str(e), 5000 + 50 * k + 200 * (y - 2022), acc, rel)
            add("us-gaap:CashAndCashEquivalentsAtCarryingValue", str(e), 800.0 + 10 * k, acc, rel)
            add("us-gaap:LongTermDebt", str(e), 2000.0, acc, rel)
            if k == 2:
                add("us-gaap:ShortTermBorrowings", str(e), 150.0, acc, rel)
            sd = rel.date() - timedelta(days=5)
            add("dei:EntityCommonStockSharesOutstanding", str(sd), 100.0, acc, rel, "shares")
    return pd.DataFrame(rows)


FACTS = _facts()


def _bars():
    days = [pd.Timestamp(x).date() for x in SESSIONS
            if np.datetime64("2023-01-03") <= x <= np.datetime64("2026-09-30")]  # fmt: skip
    n = len(days)
    rng = np.random.default_rng(5)
    close = 40 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    raw = close.copy()
    post = np.array([d >= SPLIT_EX for d in days])
    raw[post] /= 2
    rel = [pd.Timestamp(datetime(d.year, d.month, d.day, 21, 15, tzinfo=UTC)) for d in days]
    bars = pd.DataFrame({"asset_id": 1, "date": days, "open": raw, "high": raw * 1.01,
                         "low": raw * 0.99, "close": raw, "volume": 1e6, "source": "alpaca",
                         "released_at": rel, "asset_class": "equity"})  # fmt: skip
    split = pd.DataFrame([{
        "external_id": "s", "kind": "ca_forward_splits", "symbol": "X", "ex_date": SPLIT_EX,
        "released_at": rel[0], "old_rate": 1.0, "new_rate": 2.0, "rate": np.nan,
        "special": False, "asset_id": 1, "manual": False,
    }])  # fmt: skip
    return bars, split


BARS, SPLIT = _bars()
END = BARS["released_at"].max() + pd.Timedelta(hours=1)


def frame(asof=END, facts=FACTS, sector="Industrials", actions=SPLIT):
    f = price.prepare(BARS, asof, CFG.price, CFG.adjust, asset_class="equity", actions=actions,
                      sessions=SESSIONS, facts=facts, fund_cfg=FC, sector=sector)  # fmt: skip
    assert f is not None
    return f


FULL = frame()


# ---------------------------------------------------------------- hand truth at a time
def known_quarters(at: pd.Timestamp) -> list[tuple[int, int]]:
    return [(y, k) for y in YEARS for k in (1, 2, 3, 4) if filing_time(y, k) <= at]


def ttm_truth(item: str, at: pd.Timestamp, back: int = 0):
    qs = known_quarters(at)
    if len(qs) < 4 + back:
        return None, None
    last4 = qs[len(qs) - 4 - back : len(qs) - back]
    return qend(*last4[-1]), sum(true_q(y, k, item, at) for y, k in last4)


def hand(name: str, f, t: int) -> float:
    at = pd.Timestamp(int(f.ready_ns[t]), tz=UTC)
    d = pd.Timestamp(f.dates[t]).date()
    qs = known_quarters(at)
    if len(qs) < 4:
        return math.nan
    fresh = (d - qend(*qs[-1])).days <= FC.max_age_days
    # shares: the newest dei value (100), dated filing - 5 days, carried across the split
    last_filing = filing_time(*qs[-1]).date()
    shares = 100.0 * (2 if last_filing - timedelta(days=5) < SPLIT_EX <= d else 1)
    mcap = f.close_raw[t] * shares
    y, k = qs[-1]
    equity = 5000 + 50 * k + 200 * (y - 2022)
    cash = 800.0 + 10 * k
    debt = 2000.0 + (150.0 if k == 2 else 0.0)
    _, rev = ttm_truth("revenue", at)
    _, ni = ttm_truth("ni", at)
    _, op = ttm_truth("op", at)
    _, da = ttm_truth("da", at)
    _, cfo = ttm_truth("cfo", at)
    _, capex = ttm_truth("capex", at)
    _, tax = ttm_truth("tax", at)
    _, pretax = ttm_truth("pretax", at)
    _, eps = ttm_truth("eps", at)
    if not fresh:
        return math.nan
    if name == "earnings_yield":
        return ni / mcap
    if name == "fcf_yield":
        return (cfo - capex) / mcap
    if name == "ebitda_ev":
        return (op + da) / (mcap + debt - cash)
    if name == "roic":
        rate = min(max(tax / pretax, 0), FC.tax.max_rate)
        return op * (1 - rate) / (equity + debt - cash)
    if name == "net_debt_ebitda":
        return (debt - cash) / (op + da)
    if name == "gm_stability":
        if len(qs) < FC.gm_quarters:
            return math.nan
        m = [true_q(yy, kk, "gp", at) / true_q(yy, kk, "revenue", at)
             for yy, kk in qs[-FC.gm_quarters :]]  # fmt: skip
        return -float(np.std(m, ddof=1))
    if name in ("revenue_yoy", "eps_yoy"):
        if len(qs) < 8:
            return math.nan
        item = "revenue" if name == "revenue_yoy" else "eps"
        now, prev = (
            (rev, ttm_truth(item, at, back=4)[1])
            if item == "revenue"
            else (eps, ttm_truth("eps", at, back=4)[1])
        )
        if name == "eps_yoy":
            guard = timedelta(days=round(FC.split_guard_years * 365))
            if SPLIT_EX <= d and d - SPLIT_EX <= guard:
                return math.nan
            return (now - prev) / abs(prev)
        return now / prev - 1
    raise KeyError(name)


HAND_NAMES = {"earnings_yield", "fcf_yield", "ebitda_ev", "roic", "net_debt_ebitda",
              "gm_stability", "revenue_yoy", "eps_yoy"}  # fmt: skip


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# every 3rd bar: the restatement window (2025-11 .. 2026-03) once slipped past a sparse list
CHECK = list(range(0, len(BARS), 3)) + [len(BARS) - 1]


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_fundamental_hand_calcs(spec):
    assert spec.name in HAND_NAMES, f"{spec.name} has no hand calculation"
    got = np.asarray(spec.fn(FULL), dtype=float)
    for t in CHECK:
        assert _same(float(got[t]), hand(spec.name, FULL, t)), (
            spec.name,
            t,
            got[t],
            hand(spec.name, FULL, t),
        )
    assert np.isfinite(got).any()


def test_timeline_restatement_and_derived_quarters():
    tl = fu.timeline(FACTS, FC)
    before = tl[tl.index < int(restate_time().as_unit("ns").value)].iloc[-1]
    after = tl.loc[int(restate_time().as_unit("ns").value)]
    # revenue a year before 2025-03-31 = TTM at 2024-03-31, which holds the restated 2024 Q1
    at = restate_time()
    assert math.isclose(after["revenue_prev"], ttm_truth("revenue", at, back=4)[1])
    assert after["revenue_end"] == date(2025, 3, 31)
    assert before["revenue_end"] == date(2024, 12, 31)
    # cash flow is YTD-only: Q2..Q4 come from differences
    assert math.isclose(after["fcf_ttm"], ttm_truth("cfo", at)[1] - ttm_truth("capex", at)[1])


def test_staleness_and_sector_rules():
    late = np.flatnonzero(FULL.dates > np.datetime64("2026-07-20"))  # last period end 2025-12-31
    ey = np.asarray(fu.earnings_yield(FULL))
    assert np.isnan(ey[late]).all() and np.isfinite(ey[late[0] - 30])
    fin = frame(sector="Financials")
    assert np.isnan(np.asarray(fu.ebitda_ev(fin))).all()
    assert np.isnan(np.asarray(fu.net_debt_ebitda(fin))).all()
    assert np.isfinite(np.asarray(fu.roic(fin))).any()


def test_market_cap_carries_shares_across_the_split():
    i = int(np.searchsorted(FULL.dates, np.datetime64(SPLIT_EX)))
    m = fu.market_cap(FULL)
    # raw price halves at the split, the share count doubles: market cap is continuous
    assert abs(m[i] / m[i - 1] - FULL.close[i] / FULL.close[i - 1]) < 1e-12


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
@pytest.mark.parametrize("cut", [filing_time(2024, 2), restate_time(), filing_time(2025, 3)])
def test_no_lookahead_fundamentals(spec, cut):
    t = cut + pd.Timedelta(hours=2)
    clean = price.compute_asof(frame(t), [spec])[spec.key]
    late = FACTS["released_at"] > t
    dirty_facts = FACTS.copy()
    dirty_facts.loc[late, "value"] = dirty_facts.loc[late, "value"] * 3
    # a "restatement" of an old period released later
    extra = (
        FACTS[~late]
        .iloc[[0]]
        .assign(value=1e9, accession="late", released_at=t + pd.Timedelta(minutes=5))
    )
    dirty = price.compute_asof(frame(t, facts=pd.concat([dirty_facts, extra])), [spec])[spec.key]
    assert _same(clean, dirty), (spec.name, clean, dirty)
    full_vals, ready = price.compute_series(FULL, [spec])
    ok = np.flatnonzero(ready[spec.key].to_numpy() <= int(t.as_unit("ns").value))
    assert _same(float(full_vals[spec.key].iloc[ok[-1]]), clean)  # backfill = daily


def test_sum_option_and_tag_priority():
    k = fu.parse(FACTS)
    debt = fu._resolve(k, FC.concepts["debt"], "USD")
    assert debt[date(2024, 6, 30)][-1][1] == 2150.0 and debt[date(2024, 9, 30)][-1][1] == 2000.0
    rev = fu._resolve(k, FC.concepts["revenue"], "USD")
    assert (date(2023, 1, 1), date(2023, 3, 31)) in rev  # old tag (SalesRevenueNet)
    assert (date(2025, 1, 1), date(2025, 3, 31)) in rev  # new tag


def _mini(rows):
    rel = pd.Timestamp("2026-05-10 21:00", tz="UTC")
    return pd.DataFrame([{"asset_id": 2, "fiscal_period": fp, "metric": m, "value": v,
                          "unit": "USD", "accession": "a1", "released_at": rel}
                         for fp, m, v in rows])  # fmt: skip


QS = ["2025-04-01/2025-06-30", "2025-07-01/2025-09-30", "2025-10-01/2025-12-31",
      "2026-01-01/2026-03-31"]  # fmt: skip


def test_operating_income_fallback_and_debt_free_company():
    rows = [(q, "us-gaap:Revenues", 100.0 + i) for i, q in enumerate(QS)]
    rows += [(q, "us-gaap:CostsAndExpenses", 70.0) for q in QS]
    rows += [("2026-03-31", "us-gaap:StockholdersEquity", 500.0)]
    last = fu.timeline(_mini(rows), FC).iloc[-1]
    assert last["opinc_ttm"] == sum(100.0 + i for i in range(4)) - 4 * 70.0
    assert last["opinc_end"] == date(2026, 3, 31)
    assert last["debt"] == 0.0 and last["debt_end"] == date(2026, 3, 31)  # never tagged debt


def test_debt_tagged_once_is_never_assumed_zero():
    rows = [(q, "us-gaap:Revenues", 100.0) for q in QS]
    rows += [("2026-03-31", "us-gaap:StockholdersEquity", 500.0),
             ("2020-12-31", "us-gaap:LongTermDebt", 300.0)]  # fmt: skip
    last = fu.timeline(_mini(rows), FC).iloc[-1]
    assert last["debt"] == 300.0 and last["debt_end"] == date(2020, 12, 31)  # stale -> NaN later


def test_asset_without_facts_is_all_nan():
    f = frame(facts=FACTS.iloc[:0])
    for s in SPECS:
        assert np.isnan(np.asarray(s.fn(f), dtype=float)).all(), s.name


def test_parse_instants_only():
    # an asset whose facts are all balance-sheet instants (no duration to split)
    facts = pd.DataFrame({
        "asset_id": [1, 1], "fiscal_period": ["2023-12-31", "2024-03-31"],
        "metric": ["dei:EntityCommonStockSharesOutstanding"] * 2, "value": [1e6, 1.1e6],
        "unit": ["shares", "shares"], "accession": ["a", "b"],
        "released_at": pd.to_datetime(["2024-02-01", "2024-05-01"], utc=True),
    })  # fmt: skip
    p = fu.parse(facts)
    assert p["start"].isna().all() and list(p["end"]) == [date(2023, 12, 31), date(2024, 3, 31)]
    tl = fu.timeline(facts, CFG.fundamentals)
    assert len(tl) == 2 and tl["shares"].iloc[-1] == 1.1e6
