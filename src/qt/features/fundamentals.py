"""Fundamentals as known at a decision time, and the per-asset ratios built on them (doc 03 §5.2).

Facts are EDGAR XBRL values (``raw_fundamentals``), one row per ``(fiscal_period, metric,
accession)`` with ``released_at`` = the filing's acceptance time. At a time r only rows released by
r count, and per fiscal period the newest released value wins (a restatement replaces the old value
from its own release on, never before).

:func:`timeline` evaluates one asset's facts at every filing release time, giving a step function:
TTM flows, the value a year earlier, balance-sheet instants and gross-margin stability. A bar uses
the newest step released by the time the bar is ready (``PriceFrame.ready_ns``), so the backfill
pass and the daily run see the same filings.

TTM: quarters are 3-month durations as reported, plus differences of year-to-date durations that
share a start (6M - 3M = Q2, FY - 9M = Q4), so cash-flow statements (YTD only) work too. TTM at a
period end E = the fiscal-year duration ending at E if reported, else the sum of the four quarters
ending at E (each ~91 days apart). Ratios that need a price use raw close x shares outstanding, with
the shares carried forward across later splits.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from qt.common.config import FundamentalsCfg
from qt.features import registry

GROUP = "fundamentals"
SOURCES = ("fundamentals", "bars")

FLOWS = (
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "costs_and_expenses",
    "net_income",
    "eps_diluted", "da", "income_tax", "pretax_income", "cfo", "capex",
)  # fmt: skip
INSTANTS = ("equity", "cash", "debt", "shares")

# timeline columns: value columns, then the period end (or instant date) each one refers to
COLUMNS = (
    "revenue_ttm", "revenue_prev", "revenue_end",
    "ni_ttm", "ni_end",
    "fcf_ttm", "fcf_end",
    "opinc_ttm", "opinc_end", "ebitda_ttm", "ebitda_end",
    "tax_rate",
    "eps_ttm", "eps_prev", "eps_end",
    "gm_std", "gm_end",
    "equity", "equity_end", "cash", "cash_end", "debt", "debt_end",
    "shares", "shares_end",
)  # fmt: skip


def tags_used(cfg: FundamentalsCfg) -> list[str]:
    """Every XBRL metric any input reads (for the loader's filter)."""
    out: list[str] = []
    for opts in cfg.concepts.values():
        for opt in opts:
            out.extend(t for t in opt.split("+") if t not in out)
    return out


# ---------------------------------------------------------------- parsing and resolution
def parse(facts: pd.DataFrame) -> pd.DataFrame:
    """``fiscal_period`` -> ``start`` (None for instants) and ``end``; ``released_at`` -> ns."""
    if facts.empty:
        return facts.assign(start=[], end=[], rel=pd.Series(dtype="int64"))
    fp = facts["fiscal_period"].astype(str)
    has = fp.str.contains("/")
    start = fp.where(has).str.split("/").str[0]
    end = fp.where(has, fp).str.split("/").str[-1]
    out = facts.assign(
        start=[date.fromisoformat(x) if isinstance(x, str) else None for x in start],
        end=[date.fromisoformat(x) for x in end],
        rel=pd.to_datetime(facts["released_at"], utc=True).dt.as_unit("ns").astype("int64"),
    )
    return out.sort_values(["rel", "accession"], kind="stable").reset_index(drop=True)


def _histories(known: pd.DataFrame, options: list[str], unit: str) -> dict:
    """period key -> per option ``(first-term history, [later-term histories])``.

    Histories are ``[(released ns, value)]``, oldest first. Built once per asset; :func:`_cut`
    then evaluates them at any time.
    """
    k = known[known["unit"] == unit]
    by_metric: dict[str, dict] = {}
    for m, g in k.groupby("metric", sort=False):
        d: dict = {}
        for s, e, v, r in zip(g["start"], g["end"], g["value"], g["rel"], strict=True):
            d.setdefault((s, e) if s is not None else e, []).append((int(r), float(v)))
        for h in d.values():
            h.sort()
        by_metric[str(m)] = d
    out: dict = {}
    for opt in options:
        terms = opt.split("+")
        for key, hist in by_metric.get(terms[0], {}).items():
            rest = [h for h in (by_metric.get(t, {}).get(key) for t in terms[1:]) if h]
            out.setdefault(key, []).append((hist, rest))
    return out


def _cut(hists: dict, rel: int) -> dict:
    """period key -> version history known at ``rel``, from the first option present by then.

    A sum option ``A+B`` adds the later terms' values known at each version of the first term
    (absent ones count 0).
    """
    out: dict = {}
    for key, opts in hists.items():
        for first, rest in opts:
            h = [x for x in first if x[0] <= rel]
            if not h:
                continue
            if rest:
                rest_k = [[x for x in r if x[0] <= rel] for r in rest]
                h = [(r, v + sum(_value_at(x, r) for x in rest_k if x)) for r, v in h]
            out[key] = h
            break
    return out


def _resolve(known: pd.DataFrame, options: list[str], unit: str) -> dict:
    """period key -> version history known now (every row of ``known``)."""
    return _cut(_histories(known, options, unit), np.iinfo(np.int64).max)


def _days(a: date, b: date) -> int:
    return (b - a).days


def _value_at(hist: list[tuple[int, float]], rel: int) -> float:
    """The newest value released by ``rel`` (else the first one ever released)."""
    v = hist[0][1]
    for r, x in hist:
        if r > rel:
            break
        v = x
    return v


def quarters(durs: dict, cfg: FundamentalsCfg) -> dict[date, float]:
    """Quarter values by period end: direct 3-month durations, then YTD differences.

    ``durs`` maps ``(start, end)`` to that period's version history ``[(released ns, value)]``.
    A difference takes the longer duration's newest value and the shorter one as known when that
    value was filed, so both sides come from one vintage: Q4 = FY - 9M stays right while a later
    10-Q has restated the 9M figure but the FY figure hasn't been re-filed yet.
    """
    qlo, qhi = cfg.quarter_days
    q: dict[date, float] = {}
    by_start: dict[date, list[tuple[date, list]]] = {}
    for (s, e), hist in durs.items():
        if qlo <= _days(s, e) <= qhi:
            q[e] = hist[-1][1]
        by_start.setdefault(s, []).append((e, hist))
    for _s, items in by_start.items():
        items.sort(key=lambda x: x[0])
        for (e0, h0), (e1, h1) in zip(items, items[1:], strict=False):
            if qlo <= _days(e0, e1) <= qhi and e1 not in q:
                r1, v1 = h1[-1]
                q[e1] = v1 - _value_at(h0, r1)
    return q


def annuals(durs: dict, cfg: FundamentalsCfg) -> dict[date, float]:
    alo, ahi = cfg.annual_days
    return {e: h[-1][1] for (s, e), h in durs.items() if alo <= _days(s, e) <= ahi}


def _step_back(q: dict[date, float], cur: date, step: int, tol: int) -> date | None:
    """The quarter end closest to ``step`` days before ``cur`` (within ``tol``), if any."""
    best = None
    for e in q:
        off = abs(_days(e, cur) - step)
        if off <= tol and (best is None or off < abs(_days(best, cur) - step)):
            best = e
    return best


def ttm_at(q: dict, a: dict, end: date, cfg: FundamentalsCfg) -> float | None:
    if end in a:
        return a[end]
    if end not in q:
        return None
    total, cur = q[end], end
    for _ in range(3):
        prev = _step_back(q, cur, 91, cfg.step_tol_days)
        if prev is None:
            return None
        total += q[prev]
        cur = prev
    return total


def latest_ttm(q: dict, a: dict, cfg: FundamentalsCfg) -> tuple[date | None, float | None]:
    """TTM at the newest period end it can be built for (trying the two newest ends)."""
    for end in sorted(set(q) | set(a), reverse=True)[:2]:
        v = ttm_at(q, a, end, cfg)
        if v is not None:
            return end, v
    return None, None


def year_before(q: dict, a: dict, end: date, cfg: FundamentalsCfg) -> float | None:
    for e in sorted(set(q) | set(a), reverse=True):
        if abs(_days(e, end) - 365) <= cfg.step_tol_days:
            return ttm_at(q, a, e, cfg)
    return None


def _latest_instant(d: dict) -> tuple[date | None, float | None]:
    if not d:
        return None, None
    e = max(d)
    return e, d[e]


def _gm_std(gp_q: dict, rev_q: dict, cost_q: dict, cfg: FundamentalsCfg):
    """Stdev of the last ``gm_quarters`` consecutive quarterly gross margins, and its end."""
    gp = dict(gp_q)
    for e, r in rev_q.items():
        if e not in gp and e in cost_q:
            gp[e] = r - cost_q[e]
    margins = {e: gp[e] / rev_q[e] for e in gp if e in rev_q and rev_q[e] > 0}
    if not margins:
        return None, None
    end = max(margins)
    seq, cur = [margins[end]], end
    while len(seq) < cfg.gm_quarters:
        prev = _step_back(margins, cur, 91, cfg.step_tol_days)
        if prev is None:
            return None, end
        seq.append(margins[prev])
        cur = prev
    return float(np.std(seq, ddof=1)), end


def snapshot(resolved: dict[str, dict], cfg: FundamentalsCfg) -> dict[str, float | date | None]:
    """Every timeline column from each input's version histories known at one time."""
    flows, inst = {}, {}
    for item in FLOWS:
        durs = {key: h for key, h in resolved[item].items() if isinstance(key, tuple)}
        flows[item] = (quarters(durs, cfg), annuals(durs, cfg))
    for item in INSTANTS:
        inst[item] = _latest_instant(
            {key: h[-1][1] for key, h in resolved[item].items() if not isinstance(key, tuple)}
        )

    out: dict[str, float | date | None] = dict.fromkeys(COLUMNS)

    def at(item: str, end: date | None) -> float | None:
        return None if end is None else ttm_at(*flows[item], end, cfg)

    e, v = latest_ttm(*flows["revenue"], cfg)
    out.update(revenue_ttm=v, revenue_end=e,
               revenue_prev=year_before(*flows["revenue"], e, cfg) if e else None)  # fmt: skip
    e, v = latest_ttm(*flows["net_income"], cfg)
    out.update(ni_ttm=v, ni_end=e)
    e, cfo = latest_ttm(*flows["cfo"], cfg)
    capex = at("capex", e)
    if cfo is not None and capex is not None:
        out.update(fcf_ttm=cfo - capex, fcf_end=e)
    e, op = latest_ttm(*flows["operating_income"], cfg)
    if op is None:  # filers without an operating-income line: revenue - costs and expenses
        re_, rev = latest_ttm(*flows["revenue"], cfg)
        costs = at("costs_and_expenses", re_)
        if rev is not None and costs is not None:
            e, op = re_, rev - costs
    da = at("da", e)
    out.update(opinc_ttm=op, opinc_end=e)
    if op is not None and da is not None:
        out.update(ebitda_ttm=op + da, ebitda_end=e)
    pe, pretax = latest_ttm(*flows["pretax_income"], cfg)
    tax = at("income_tax", pe)
    rate = cfg.tax.default_rate
    if pretax is not None and tax is not None and pretax > 0:
        rate = min(max(tax / pretax, 0.0), cfg.tax.max_rate)
    out["tax_rate"] = rate
    e, v = latest_ttm(*flows["eps_diluted"], cfg)
    out.update(eps_ttm=v, eps_end=e,
               eps_prev=year_before(*flows["eps_diluted"], e, cfg) if e else None)  # fmt: skip
    gp_q, rev_q, cost_q = flows["gross_profit"][0], flows["revenue"][0], flows["cost_of_revenue"][0]
    out["gm_std"], out["gm_end"] = _gm_std(gp_q, rev_q, cost_q, cfg)
    for item in INSTANTS:
        e, v = inst[item]
        out[item], out[f"{item}_end"] = v, e
    if cfg.debt_absent_as_zero and not resolved["debt"] and out["equity_end"] is not None:
        out["debt"], out["debt_end"] = 0.0, out["equity_end"]  # never tagged any debt: none
    return out


def timeline(facts: pd.DataFrame, cfg: FundamentalsCfg) -> pd.DataFrame:
    """One row per distinct filing release time (index: UTC ns), columns :data:`COLUMNS`."""
    p = parse(facts)
    if p.empty:
        return pd.DataFrame(columns=list(COLUMNS), index=pd.Index([], dtype="int64", name="rel"))
    hists = {item: _histories(p, cfg.concepts[item], cfg.unit(item)) for item in FLOWS + INSTANTS}
    uniq = np.unique(p["rel"].to_numpy())
    rows = [snapshot({i: _cut(h, int(r)) for i, h in hists.items()}, cfg) for r in uniq]
    return pd.DataFrame(rows, index=pd.Index(uniq, name="rel"), columns=list(COLUMNS))


# ---------------------------------------------------------------- per-bar alignment
def aligned(f, col: str) -> np.ndarray:
    """Timeline column per bar: the newest step released by the time the bar is ready."""
    key = f"fund:{col}"
    if key in f._cache:
        return f._cache[key]
    tl = f.fund
    n = f.n
    if tl is None or tl.empty:
        out = (np.full(n, np.datetime64("NaT", "D")) if col.endswith("_end")
               else np.full(n, np.nan))  # fmt: skip
    else:
        pos = np.searchsorted(tl.index.to_numpy(), f.ready_ns, side="right") - 1
        vals = tl[col].to_numpy()
        if col.endswith("_end"):
            src = np.array([np.datetime64(x, "D") if x is not None else np.datetime64("NaT", "D")
                            for x in vals])  # fmt: skip
            out = np.where(pos >= 0, src[np.maximum(pos, 0)], np.datetime64("NaT", "D"))
        else:
            src = np.array([np.nan if x is None else float(x) for x in vals])
            out = np.where(pos >= 0, src[np.maximum(pos, 0)], np.nan)
    f._cache[key] = out
    return out


def _fresh(f, *ends: str) -> np.ndarray:
    """True where every named period end is known and at most ``max_age_days`` old."""
    ok = np.ones(f.n, dtype=bool)
    lim = np.timedelta64(f.fcfg.max_age_days, "D")
    for e in ends:
        d = aligned(f, e)
        ok &= ~np.isnat(d) & ((f.dates - d) <= lim)
    return ok


def _v(f, col: str) -> np.ndarray:
    return aligned(f, col).astype(float)


def market_cap(f) -> np.ndarray:
    """Raw close x shares outstanding, with the shares carried across splits since their date."""
    shares = _v(f, "shares")
    sd = aligned(f, "shares_end")
    idx = np.searchsorted(f.dates, sd, side="right") - 1  # the bar on/before the shares date
    ok = ~np.isnat(sd) & (idx >= 0)
    pf_then = np.where(ok, f.split_pf[np.maximum(idx, 0)], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcap = f.close_raw * shares * f.split_pf / pf_then
    return np.where(ok & (mcap > 0), mcap, np.nan)


def _no_ebitda(f) -> bool:
    return f.sector is not None and f.sector in f.fcfg.no_ebitda_sectors


def _div(a: np.ndarray, b: np.ndarray, pos_b: bool = True) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return np.where((b > 0) if pos_b else (b != 0), r, np.nan)


def earnings_yield(f) -> np.ndarray:
    return np.where(_fresh(f, "ni_end", "shares_end"), _div(_v(f, "ni_ttm"), market_cap(f)), np.nan)


def fcf_yield(f) -> np.ndarray:
    return np.where(_fresh(f, "fcf_end", "shares_end"), _div(_v(f, "fcf_ttm"), market_cap(f)),
                    np.nan)  # fmt: skip


def enterprise_value(f) -> np.ndarray:
    return market_cap(f) + _v(f, "debt") - _v(f, "cash")


def ebitda_ev(f) -> np.ndarray:
    if _no_ebitda(f):
        return np.full(f.n, np.nan)
    ok = _fresh(f, "ebitda_end", "shares_end", "debt_end", "cash_end")
    return np.where(ok, _div(_v(f, "ebitda_ttm"), enterprise_value(f)), np.nan)


def roic(f) -> np.ndarray:
    ic = _v(f, "equity") + _v(f, "debt") - _v(f, "cash")
    nopat = _v(f, "opinc_ttm") * (1.0 - _v(f, "tax_rate"))
    ok = _fresh(f, "opinc_end", "equity_end", "debt_end", "cash_end")
    return np.where(ok, _div(nopat, ic), np.nan)


def gm_stability(f) -> np.ndarray:
    return np.where(_fresh(f, "gm_end"), -_v(f, "gm_std"), np.nan)


def net_debt_ebitda(f) -> np.ndarray:
    if _no_ebitda(f):
        return np.full(f.n, np.nan)
    ok = _fresh(f, "ebitda_end", "debt_end", "cash_end")
    return np.where(ok, _div(_v(f, "debt") - _v(f, "cash"), _v(f, "ebitda_ttm")), np.nan)


def revenue_yoy(f) -> np.ndarray:
    return np.where(_fresh(f, "revenue_end"), _div(_v(f, "revenue_ttm"), _v(f, "revenue_prev")) - 1,
                    np.nan)  # fmt: skip


def eps_yoy(f) -> np.ndarray:
    prev = _v(f, "eps_prev")
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (_v(f, "eps_ttm") - prev) / np.abs(prev)
    ok = _fresh(f, "eps_end") & (np.abs(prev) >= f.fcfg.eps_yoy_min_base)
    if len(f.split_dates):
        guard = np.timedelta64(int(round(f.fcfg.split_guard_years * 365)), "D")
        near = np.zeros(f.n, dtype=bool)
        for sd in f.split_dates:
            near |= (sd <= f.dates) & ((f.dates - sd) <= guard)
        ok &= ~near
    return np.where(ok, g, np.nan)


def register(cfg: FundamentalsCfg) -> list[registry.FeatureSpec]:
    """Register the fundamentals ratios (idempotent). Returns the specs."""
    age = f"period end <= {cfg.max_age_days} days old, else NaN"
    mcap = "mcap = raw close x shares outstanding (carried across later splits)"
    defs = [
        ("earnings_yield", f"net income TTM / mcap; {mcap}; {age}", earnings_yield),
        ("fcf_yield", f"(CFO - capex) TTM / mcap; {age}", fcf_yield),
        ("ebitda_ev",
         f"(operating income + D&A) TTM / (mcap + debt - cash); NaN for {cfg.no_ebitda_sectors} "
         f"and EV <= 0; {age}", ebitda_ev),
        ("roic",
         f"operating income TTM x (1 - tax rate) / (equity + debt - cash); tax rate = tax / "
         f"pretax clipped to [0, {cfg.tax.max_rate}], else {cfg.tax.default_rate}; IC <= 0 -> NaN",
         roic),
        ("gm_stability",
         f"-stdev(quarterly gross margin) over {cfg.gm_quarters} consecutive quarters; {age}",
         gm_stability),
        ("net_debt_ebitda",
         f"(debt - cash) / EBITDA TTM; EBITDA <= 0 or {cfg.no_ebitda_sectors} -> NaN",
         net_debt_ebitda),
        ("revenue_yoy", f"revenue TTM / revenue TTM a year earlier - 1; {age}", revenue_yoy),
        ("eps_yoy",
         f"(diluted EPS TTM - a year earlier) / |a year earlier|; |base| < "
         f"{cfg.eps_yoy_min_base} or a split within {cfg.split_guard_years} y -> NaN",
         eps_yoy),
    ]  # fmt: skip
    keys = []
    for name, formula, fn in defs:
        registry.feature(name, 1, kind="asset", group=GROUP, formula=formula, sources=SOURCES)(fn)
        keys.append((name, 1))
    return [registry.get(*k) for k in keys]
