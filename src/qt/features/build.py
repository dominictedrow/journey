"""Feature build: the daily job and the backfill (component doc 03 section 10, D-042).

    uv run python -m qt.features.build --since 2016-01-01 --until 2026-11-26
    uv run python -m qt.features.build --since 2016-01-01 --until 2026-11-26 \\
        --only 'earnings_yield,fcf_yield,earnings_yield_xs*'   # write just these names (backfill)
    uv run python -m qt.features.build --daily           # today's decision time only
    uv run python -m qt.features.build --refit-hmm       # missing monthly HMM fits up to now

One code path for both. Every input is loaded once, cut at the run's last decision time
(``until``), and every earlier decision time T is cut in memory (``released_at <= T``):

1. **Assets.** Each asset's frame is prepared once (adjusted as of ``until``: scale-free features
   are unchanged by a later split, D-036) and its whole history computed with a ready time per
   value. The value at T is the newest row ready by T (``price.compute_asof`` for many Ts at
   once). An asset has rows at T while its newest bar known at T is at most ``max_bar_age_days``
   old. Equities: point-in-time index members only; ETFs and coins: always.
2. **Decision times.** T = ``run_et`` New York time on every calendar day. Sessions get
   everything; other days only crypto assets and the ``nonsession_market_groups``.
3. **Market vector** from one ``MacroState`` walked forward through T, so each T sees exactly the
   vintages released by then. The market ``_tsz`` features need the previous ``z_window``
   session values, so the walk starts that many sessions before ``since`` (warm-up, not written).
4. **Writes** are append-only (``ON CONFLICT DO NOTHING``), in time order; a backfill compresses
   each 30-day chunk once it is complete, so the uncompressed history never piles up on disk.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from qt.collect.calendars import CPI_KIND, CPI_SOURCE, FOMC_KIND, FOMC_SOURCE
from qt.common.calendar import TradingCalendar
from qt.common.config import FeaturesCfg, Settings
from qt.common.paths import REPO_ROOT, repo_path
from qt.features import catalog as catalog_mod
from qt.features import cross_section, fundamentals, inputs, market, normalize, price, registry
from qt.features import regime as regime_mod
from qt.features.crypto import CryptoInputs
from qt.features.events import EventInputs
from qt.features.inputs import ET
from qt.features.regime import hmm

log = logging.getLogger("qt.features.build")

NS_INF = np.iinfo(np.int64).max
SECTOR_NONE = None


# ---------------------------------------------------------------- inputs
@dataclass
class BuildInputs:
    until: pd.Timestamp
    assets: pd.DataFrame
    bars: pd.DataFrame
    actions: pd.DataFrame
    earnings: pd.DataFrame
    facts: pd.DataFrame
    membership: pd.DataFrame
    macro: pd.DataFrame
    fomc: pd.DataFrame
    cpi: pd.DataFrame
    crypto: CryptoInputs
    fits: list[hmm.HmmFit] = field(default_factory=list)


def macro_series(cat: catalog_mod.Catalog, cfg: FeaturesCfg) -> list[str]:
    """Every FRED series a market feature or a price context series reads."""
    groups = sorted({s.group for s in cat.market})
    out = set(normalize.macro_sources(cat.market, groups))
    out.update(v for v in cfg.price.corr.series.values() if _is_fred(v))
    return sorted(out)


def _is_fred(name: str) -> bool:
    return name.isupper() and "/" not in name and "-" not in name and "." not in name \
        and "^" not in name and "=" not in name  # fmt: skip


def load_inputs(
    conn: psycopg.Connection, settings: Settings, until: datetime, cat: catalog_mod.Catalog
) -> BuildInputs:
    f = settings.config.features
    t0 = time.monotonic()
    assets = inputs.load_assets(conn)
    excl = inputs.bar_exclusions(conn, settings.config.collect.identity_overrides)
    bars = inputs.load_bars(conn, until, f.bars, exclude=excl)
    actions = inputs.mapped_corp_actions(conn, until, bars, f.adjust.manual)
    earnings = inputs.load_earnings(conn, until, bars)
    eq = assets[(assets["asset_class"] == "equity") & assets["cik"].notna()]["asset_id"]
    facts = inputs.load_fundamentals(conn, until, fundamentals.tags_used(f.fundamentals),
                                     asset_ids=eq.astype(int).tolist())  # fmt: skip
    membership = inputs.load_membership(conn, until, f.cross_section.index_code)
    macro = inputs.load_macro(conn, macro_series(cat, f), until, f.market.macro_fixes)
    syms = f.market.crypto.symbols
    cx = CryptoInputs(
        dominance=inputs.load_metrics(conn, until, "coingecko", ["btc_dominance_pct"]),
        mcap=inputs.load_metrics(conn, until, "coingecko", ["market_cap_usd"]),
        stablecoins=inputs.load_metrics(conn, until, "defillama", ["stablecoin_supply_usd"]),
        etf=inputs.load_metrics(conn, until, "sosovalue", ["total_net_inflow"]),
        funding=inputs.load_derivs(conn, until, "funding", syms),
        oi=inputs.load_derivs(conn, until, "oi", syms),
    )
    out = BuildInputs(
        until=pd.Timestamp(until), assets=assets, bars=bars, actions=actions, earnings=earnings,
        facts=facts, membership=membership, macro=macro,
        fomc=inputs.load_schedules(conn, until, FOMC_SOURCE, FOMC_KIND),
        cpi=inputs.load_schedules(conn, until, CPI_SOURCE, CPI_KIND), crypto=cx,
    )  # fmt: skip
    log.info("build_inputs_loaded", extra={"bars": len(bars), "facts": len(facts),
                                           "macro": len(macro),
                                           "secs": round(time.monotonic() - t0, 1)})  # fmt: skip
    return out


# ---------------------------------------------------------------- per-asset preparation
def asset_for_ticker(inp: BuildInputs, ticker: str) -> int | None:
    """The asset currently (or most recently) trading as ``ticker``: the one with the newest bar."""
    ids = inp.assets.loc[inp.assets["ticker"] == ticker, "asset_id"].astype(int).tolist()
    if not ids:
        return None
    last = inp.bars[inp.bars["asset_id"].isin(ids)].groupby("asset_id")["date"].max()
    return int(last.idxmax()) if len(last) else None


class Prepared:
    """Per-asset bars, actions, earnings and facts by asset id (split once)."""

    def __init__(self, inp: BuildInputs) -> None:
        self.inp = inp
        self.bars = {int(a): g for a, g in inp.bars.groupby("asset_id", sort=False)}
        self.actions = {int(a): g for a, g in inp.actions.groupby("asset_id", sort=False)} \
            if not inp.actions.empty else {}  # fmt: skip
        self.earn = {int(a): g for a, g in inp.earnings.groupby("asset_id", sort=False)} \
            if not inp.earnings.empty else {}  # fmt: skip
        self.facts = {int(a): g for a, g in inp.facts.groupby("asset_id", sort=False)} \
            if not inp.facts.empty else {}  # fmt: skip
        a = inp.assets.set_index("asset_id")
        self.cls = a["asset_class"].to_dict()
        self.sector = a["gics_sector"].where(a["gics_sector"].notna(), None).to_dict()
        self.ticker = a["ticker"].to_dict()

    def frame(self, aid: int, cfg: FeaturesCfg, ctx=None, sessions=None,
              with_facts: bool = False) -> price.PriceFrame | None:  # fmt: skip
        b = self.bars.get(aid)
        if b is None or b.empty:
            return None
        cls = self.cls[aid]
        return price.prepare(
            b, self.inp.until.to_pydatetime(), cfg.price, cfg.adjust, asset_class=cls,
            actions=self.actions.get(aid), ctx=ctx,
            earnings=self.earn.get(aid) if cls == "equity" else None, sessions=sessions,
            facts=self.facts.get(aid, inputs_empty_facts()) if with_facts else None,
            fund_cfg=cfg.fundamentals if with_facts else None, sector=self.sector.get(aid),
        )  # fmt: skip

    def close_frame(self, aid: int, cfg: FeaturesCfg, sessions=None) -> pd.DataFrame:
        """``date, value (adjusted close), released_at`` for a context / market series."""
        f = self.frame(aid, cfg, sessions=sessions)
        if f is None:
            return pd.DataFrame(columns=["date", "value", "released_at"])
        return pd.DataFrame({
            "date": pd.Index(f.dates).astype("datetime64[s]").date,
            "value": f.close,
            "released_at": pd.to_datetime(f.ready_ns, utc=True),
        })  # fmt: skip


def inputs_empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=["asset_id", "fiscal_period", "metric", "value", "unit",
                                 "accession", "released_at"])  # fmt: skip


def context_series(prep: Prepared, cfg: FeaturesCfg, sessions) -> dict[str, dict]:
    """Price context per asset class: ``{"bench": ..., <corr keys>: ...}`` of ``CtxSeries``."""
    cache: dict[str, price.CtxSeries] = {}

    tickers = set(prep.inp.assets["ticker"])

    def series(name: str) -> price.CtxSeries:
        if name not in cache:
            # an asset ticker is a price series ("SPY" looks like a FRED id too)
            if name not in tickers and _is_fred(name):
                m = prep.inp.macro[prep.inp.macro["series_id"] == name]
                fr = pd.DataFrame({"date": m["period"].tolist(), "value": m["value"].astype(float),
                                   "released_at": m["released_at"]})  # fmt: skip
                cache[name] = price.CtxSeries(fr.dropna(subset=["value"]), kind="diff")
            else:
                aid = asset_for_ticker(prep.inp, name)
                fr = prep.close_frame(aid, cfg, sessions) if aid is not None else \
                    pd.DataFrame(columns=["date", "value", "released_at"])  # fmt: skip
                cache[name] = price.CtxSeries(fr)
        return cache[name]

    out = {}
    for cls in cfg.build.asset_classes:
        ctx = {"bench": series(cfg.price.bench_for(cls))}
        ctx.update({k: series(v) for k, v in cfg.price.corr.series.items()})
        out[cls] = ctx
    return out


# ---------------------------------------------------------------- decision-time grid
def decision_time(d: date, hhmm: str) -> pd.Timestamp:
    h, m = (int(x) for x in hhmm.split(":")[:2])
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, m, tzinfo=ET)).tz_convert(UTC)


@dataclass(frozen=True)
class Grid:
    days: list[date]
    t: np.ndarray  # UTC ns per day
    session: np.ndarray  # bool per day

    @classmethod
    def build(cls, start: date, end: date, hhmm: str, cal: TradingCalendar) -> Grid:
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        t = np.array([decision_time(d, hhmm).value for d in days], dtype="int64")
        return cls(days, t, np.array([cal.is_session(d) for d in days], dtype=bool))


def _suffix_min(a: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(a[::-1], axis=0)[::-1]


def resolve(values: pd.DataFrame, ready: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """Per decision time and column: the newest row whose ready time is <= T (NaN if none)."""
    v = values.to_numpy(dtype=float)
    r = _suffix_min(ready.to_numpy(dtype="int64"))
    out = np.full((len(t), v.shape[1]), np.nan)
    for j in range(v.shape[1]):
        i = np.searchsorted(r[:, j], t, side="right") - 1
        ok = i >= 0
        out[ok, j] = v[i[ok], j]
    return out


def active_mask(frame: price.PriceFrame, grid: Grid, max_age_days: int) -> np.ndarray:
    """True where the asset's newest bar known at T is at most ``max_age_days`` old."""
    r = _suffix_min(frame.ready_ns)
    i = np.searchsorted(r, grid.t, side="right") - 1
    ok = i >= 0
    days = np.array(grid.days, dtype="datetime64[D]")
    age = np.where(ok, (days - frame.dates[np.maximum(i, 0)]).astype(int), 10**9)
    return ok & (age <= max_age_days)


@dataclass
class AssetGrid:
    """One asset's values at the grid times where it is active: ``row(t)`` for ``keys``."""

    asset_id: int
    asset_class: str
    keys: list[tuple[str, int]]
    vals: np.ndarray  # float64 [n_active, n_keys]
    active: np.ndarray  # bool [n_t]
    pos: np.ndarray  # int [n_t]: row of vals for an active grid time

    def row(self, i: int) -> np.ndarray:
        return self.vals[self.pos[i]]


def asset_specs(cat: catalog_mod.Catalog, cls: str) -> list[registry.FeatureSpec]:
    if cls == "equity":
        return cat.price + cat.fund + cat.masks
    return [s for s in cat.price if s.group != price.GROUP_EARN] + cat.masks


def asset_grid(frame: price.PriceFrame, aid: int, cls: str, cat: catalog_mod.Catalog,
               grid: Grid, cfg: FeaturesCfg) -> AssetGrid:  # fmt: skip
    specs = asset_specs(cat, cls)
    values, ready = price.compute_series(frame, specs)
    if cls in cfg.normalize.asset_classes and cat.asset_tsz:
        zv, zr = normalize.asset_ts_z(values, ready, cat.asset_tsz)
        values, ready = pd.concat([values, zv], axis=1), pd.concat([ready, zr], axis=1)
    keys = list(values.columns)
    act = active_mask(frame, grid, cfg.build.max_bar_age_days)
    if cls != "crypto":
        act &= grid.session
    vals = resolve(values, ready, grid.t[act])
    return AssetGrid(aid, cls, keys, vals, act, np.cumsum(act) - 1)


# ---------------------------------------------------------------- breadth
def breadth_inputs(
    closes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    membership: pd.DataFrame, sessions: np.ndarray, cfg: FeaturesCfg,
) -> pd.DataFrame:  # fmt: skip
    """``market.breadth_table`` over the index members (membership as of each session's T)."""
    if not closes:
        return pd.DataFrame(columns=["above_sma", "nh_nl", "adv_dec", "ready_ns"])
    c = pd.DataFrame({a: pd.Series(v, index=pd.Index(d)) for a, (d, v, _) in closes.items()})
    r = pd.DataFrame({a: pd.Series(x, index=pd.Index(d)) for a, (d, _, x) in closes.items()})
    c = c.sort_index()
    idx = c.index.to_numpy(dtype="datetime64[D]")
    idx = idx[np.isin(idx, sessions)]
    c, r = c.loc[idx], r.reindex(idx)
    mem = pd.DataFrame(False, index=c.index, columns=c.columns)
    snaps = membership.sort_values("released_at")
    for d in c.index:
        t = decision_time(pd.Timestamp(d).date(), cfg.build.run_et)
        ids = inputs.members_at(snaps, t.to_pydatetime())
        cols = [a for a in ids if a in mem.columns]
        if cols:
            mem.loc[d, cols] = True
    b = cfg.build.breadth
    return market.breadth_table(c, r, mem, sma=b.sma, hl=b.hl)


# ---------------------------------------------------------------- HMM fits
def load_fits(conn: psycopg.Connection, version: int, asof: datetime) -> list[hmm.HmmFit]:
    """Stored fits of ``version`` whose fit date is on or before ``asof`` (files under the repo)."""
    rows = conn.execute(
        "SELECT fit_date, model_path FROM hmm_fits WHERE regime_version = %s AND fit_date <= %s "
        "ORDER BY fit_date", (version, pd.Timestamp(asof).tz_convert(ET).date()),
    ).fetchall()  # fmt: skip
    out = []
    for _, p in rows:
        path = REPO_ROOT / Path(p)
        if not path.exists():
            raise FileNotFoundError(f"HMM model file missing: {path}")
        out.append(hmm.load(path))
    return out


def ensure_fits(
    conn: psycopg.Connection, settings: Settings, inp: BuildInputs, until: datetime,
    run_id: str | None = None, alert: Callable[..., bool] | None = None,
) -> tuple[list[hmm.HmmFit], list[dict]]:  # fmt: skip
    """Fit every monthly HMM missing up to ``until`` (in order); store files + ``hmm_fits`` rows.

    Returns all fits and one record per new fit (for the stability report).
    """
    f = settings.config.features
    h, version = f.regime.hmm, f.regime.version
    have = {x.fit_date: x for x in load_fits(conn, version, until)}
    sessions = _sessions()
    ms = inputs.MacroState(inp.macro)
    prep = Prepared(inp)
    prices = market_prices(prep, f, sessions)
    end = pd.Timestamp(until).tz_convert(ET).date()
    new: list[dict] = []
    prev: hmm.HmmFit | None = None
    model_dir = repo_path(h.model_dir)
    for fd in hmm.fit_dates(h.first_obs, end, h.refit_weekday):
        t = hmm.available_at(fd, h)
        if t > pd.Timestamp(until):
            break
        if fd in have:
            prev = have[fd]
            continue
        ms.advance(t)
        m = market.MarketState(asof=t, macro=ms, prices=prices, sessions=sessions, cfg=f.market)
        w = hmm.weekly_obs(m, f.market, h)
        x = hmm.training_set(w)
        if len(x) < h.min_weeks:
            continue
        fit = hmm.fit(x, h, fd, regime_version=version)
        path = hmm.model_path(model_dir, fit)
        hmm.save(fit, path)
        change = hmm.relabel_change(prev, fit, w, h.stability_weeks) if prev else None
        conn.execute(
            "INSERT INTO hmm_fits (fit_date, regime_version, train_start, train_end, n_weeks, "
            "best_seed, log_likelihood, state_order, relabel_change, model_path, run_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (fd, version, fit.train_start, fit.train_end, fit.n_weeks, fit.seed,
             fit.log_likelihood, list(fit.state_order), change,
             _stored_path(path), run_id),
        )  # fmt: skip
        flagged = change is not None and change > h.stability_threshold
        rec = {"fit_date": fd, "n_weeks": fit.n_weeks, "seed": fit.seed,
               "ll": fit.log_likelihood, "relabel_change": change, "flagged": flagged}  # fmt: skip
        new.append(rec)
        log.info("hmm_fit", extra={k: str(v) for k, v in rec.items()})
        if flagged and alert is not None:
            alert("warn", "hmm_relabel", f"HMM refit {fd} relabels {change:.0%} of 52 weeks",
                  f"fit {fd} vs {prev.fit_date}: {change:.0%} of the last {h.stability_weeks} "
                  f"weekly states changed (threshold {h.stability_threshold:.0%})")  # fmt: skip
        have[fd] = fit
        prev = fit
    return sorted(have.values(), key=lambda x: x.fit_date), new


def _stored_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo (portable to Windows), else absolute."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def stability_report(conn: psycopg.Connection, settings: Settings, today: date) -> Path:
    """``reports/hmm_stability_YYYYMMDD.md``: every stored fit with its relabel share."""
    f = settings.config.features.regime
    rows = conn.execute(
        "SELECT fit_date, train_start, train_end, n_weeks, best_seed, log_likelihood, "
        "relabel_change FROM hmm_fits WHERE regime_version = %s ORDER BY fit_date",
        (f.version,),
    ).fetchall()
    thr = f.hmm.stability_threshold
    lines = [f"# HMM stability report ({today.isoformat()})", "",
             f"Regime version {f.version}. Monthly expanding-window fits; relabel = share of the "
             f"prior {f.hmm.stability_weeks} weekly argmax states that changed vs the previous "
             f"fit (alert above {thr:.0%}).", "",
             "| fit date | train window | weeks | seed | log-lik | relabel | |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    flagged = 0
    for fd, a, b, n, seed, ll, ch in rows:
        bad = ch is not None and ch > thr
        flagged += bad
        lines.append(f"| {fd} | {a} → {b} | {n} | {seed} | {ll:.1f} | "
                     f"{'' if ch is None else f'{ch:.0%}'} | {'**ALERT**' if bad else ''} |")
    lines += ["", f"{len(rows)} fits, {flagged} above the threshold."]
    out = repo_path(settings.config.app.reports.daily_dir).parent / \
        f"hmm_stability_{today:%Y%m%d}.md"  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- market inputs
def market_prices(prep: Prepared, cfg: FeaturesCfg, sessions) -> dict[str, pd.DataFrame]:
    tks = set(cfg.market.tickers.values()) | {t for pair in cfg.market.ratios for t in pair}
    out = {}
    for tk in sorted(tks):
        aid = asset_for_ticker(prep.inp, tk)
        if aid is None:
            log.warning("market_ticker_missing", extra={"ticker": tk})
            continue
        out[tk] = prep.close_frame(aid, cfg, sessions)
    return out


_SESS: np.ndarray | None = None


def _sessions() -> np.ndarray:
    global _SESS
    if _SESS is None:
        _SESS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
    return _SESS


# ---------------------------------------------------------------- selection
def select_features(cat: catalog_mod.Catalog, patterns) -> set[tuple[str, int]]:
    """Keys ``(name, version)`` of the catalog specs whose name matches any pattern (``fnmatch``).

    ``patterns`` is a comma-separated string or an iterable of names/globs. A pattern that
    matches nothing raises ``ValueError`` (a typo must not silently write nothing).
    """
    pats = [p.strip() for p in patterns.split(",")] if isinstance(patterns, str) \
        else [p.strip() for p in patterns]  # fmt: skip
    pats = [p for p in pats if p]
    if not pats:
        raise ValueError("--only: no feature names given")
    specs = cat.all()
    unknown = [p for p in pats if not any(fnmatch.fnmatchcase(s.name, p) for s in specs)]
    if unknown:
        raise ValueError(f"--only: no feature in the catalog matches {', '.join(unknown)}")
    return {s.key for s in specs if any(fnmatch.fnmatchcase(s.name, p) for p in pats)}


# ---------------------------------------------------------------- writing
class Writer:
    """Buffers rows and inserts them append-only (COPY into a temp table, then INSERT ... ON
    CONFLICT DO NOTHING), and compresses finished chunks during a backfill."""

    COLS = ("asof", "asset_id", "feature_name", "feature_version", "value", "run_id")

    def __init__(self, conn: psycopg.Connection, run_id: str | None, batch: int,
                 compress: bool, groups: dict | None = None,
                 only: set[tuple[str, int]] | None = None) -> None:  # fmt: skip
        self.conn, self.run_id, self.batch, self.compress = conn, run_id, batch, compress
        self.only = only  # write only these (name, version) keys; None writes everything
        self.groups = groups or {}
        self.group_names: list[str] = sorted(set(self.groups.values()) | {"?"})
        gi = {g: i for i, g in enumerate(self.group_names)}
        self._gi = gi
        self.n_total = np.zeros(len(self.group_names), dtype=np.int64)
        self.n_nan = np.zeros(len(self.group_names), dtype=np.int64)
        self._meta: dict[tuple, tuple[list[str], np.ndarray, np.ndarray | None]] = {}
        self.buf = io.StringIO()
        self.n_buf = 0
        self.written = 0
        self.offered = 0
        self.regime: list[tuple] = []
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _features_stage (asof timestamptz, asset_id bigint, "
            "feature_name text, feature_version int, value double precision, run_id text) "
            "ON COMMIT PRESERVE ROWS"
        )

    def _key_meta(self, keys) -> tuple[list[str], np.ndarray, np.ndarray | None]:
        """Per key tuple: COPY text pieces, group indexes and (when ``only`` is set and drops
        some of the keys) the keep mask. Pieces and indexes are already filtered."""
        k = tuple(keys)
        m = self._meta.get(k)
        if m is None:
            keep = None
            if self.only is not None:
                keep = np.array([x in self.only for x in k], dtype=bool)
                if keep.all():
                    keep = None
            kept = k if keep is None else tuple(x for x, ok in zip(k, keep, strict=True) if ok)
            m = ([f"\t{n}\t{v}\t" for n, v in kept],
                 np.array([self._gi[self.groups.get(x, "?")] for x in kept], dtype=np.int64), keep)
            self._meta[k] = m
        return m

    def add(self, asof_iso: str, asset_id: int | None, keys, values) -> None:
        mid, gidx, keep = self._key_meta(keys)
        if not mid:
            return
        v = np.asarray(values, dtype=float)
        if keep is not None:
            v = v[keep]
        fin = np.isfinite(v)
        self.n_total += np.bincount(gidx, minlength=len(self.group_names))
        self.n_nan += np.bincount(gidx[~fin], minlength=len(self.group_names))
        head = f"{asof_iso}\t{'\\N' if asset_id is None else asset_id}"
        tail = f"\t{self.run_id or '\\N'}\n"
        self.buf.write("".join(
            f"{head}{p}{repr(float(x)) if ok else '\\N'}{tail}"
            for p, x, ok in zip(mid, v.tolist(), fin.tolist(), strict=True)
        ))  # fmt: skip
        self.n_buf += len(mid)
        if self.n_buf >= self.batch:
            self.flush()

    def nan_rate(self) -> dict[str, float]:
        return {g: float(self.n_nan[i] / self.n_total[i])
                for i, g in enumerate(self.group_names) if self.n_total[i]}  # fmt: skip

    def add_regime(self, row: dict) -> None:
        self.regime.append((row["asof"], row["regime_version"], row["quadrant"], row["hmm_calm"],
                            row["hmm_normal"], row["hmm_stress"], row["stress_score"],
                            self.run_id))  # fmt: skip

    def flush(self) -> None:
        if self.n_buf:
            with self.conn.cursor() as cur:
                cur.execute("TRUNCATE _features_stage")
                with cur.copy(f"COPY _features_stage ({', '.join(self.COLS)}) FROM STDIN") as cp:
                    cp.write(self.buf.getvalue())
                cur.execute(
                    f"INSERT INTO features ({', '.join(self.COLS)}) SELECT {', '.join(self.COLS)} "
                    "FROM _features_stage ON CONFLICT DO NOTHING"
                )
                self.written += cur.rowcount
            self.offered += self.n_buf
            self.buf = io.StringIO()
            self.n_buf = 0
        if self.regime:
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO regime (asof, regime_version, quadrant, hmm_calm, hmm_normal, "
                    "hmm_stress, stress_score, run_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING", self.regime,
                )  # fmt: skip
            self.regime = []
        self.conn.commit()

    def compress_before(self, t: pd.Timestamp) -> int:
        """Compress every features chunk that ends before ``t`` (nothing more will land there)."""
        if not self.compress:
            return 0
        self.flush()
        n = self.conn.execute(
            "SELECT count(compress_chunk(c, if_not_compressed => true)) "
            "FROM show_chunks('features', older_than => %s::timestamptz) c", (t,),
        ).fetchone()[0]  # fmt: skip
        self.conn.commit()
        return int(n)


# ---------------------------------------------------------------- the build
@dataclass
class BuildResult:
    since: date
    until: date
    rows_offered: int = 0
    rows_written: int = 0
    regime_rows: int = 0
    secs: float = 0.0
    nan_rate: dict[str, float] = field(default_factory=dict)


def build(
    conn: psycopg.Connection, settings: Settings, since: date, until: date, *,
    run_id: str | None = None, compress: bool | None = None, inp: BuildInputs | None = None,
    cat: catalog_mod.Catalog | None = None, cal: TradingCalendar | None = None,
    only=None,
) -> BuildResult:  # fmt: skip
    """Compute and write every feature and regime row for decision times on ``since..until``.

    ``only`` (comma-separated names or an iterable, globs allowed) limits what is WRITTEN to the
    matching features; everything is still computed (ranks and composites read the whole panel).
    Regime rows are skipped then. Unknown names raise before any work.
    """
    t_start = time.monotonic()
    f = settings.config.features
    cal = cal or TradingCalendar(settings.config.calendar)
    cat = cat or catalog_mod.register_all(f)
    selected = select_features(cat, only) if only is not None else None
    if selected is not None:
        log.info("build_selection", extra={"keys": len(selected), "of": len(cat.keys()),
                                           "regime_rows": "skipped"})  # fmt: skip
    t_until = decision_time(until, f.build.run_et)
    if inp is None:
        inp = load_inputs(conn, settings, t_until.to_pydatetime(), cat)
        inp.fits = load_fits(conn, f.regime.version, t_until.to_pydatetime())
    sessions = _sessions()
    res = BuildResult(since, until)

    # warm-up for the market z features: z_window sessions before since
    s_before = sessions[sessions < np.datetime64(since, "D")]
    warm = pd.Timestamp(s_before[-f.normalize.z_window]).date() if len(s_before) >= \
        f.normalize.z_window else since  # fmt: skip
    grid = Grid.build(since, until, f.build.run_et, cal)
    wgrid = Grid.build(warm, until, f.build.run_et, cal)

    # -- assets
    prep = Prepared(inp)
    ctx = context_series(prep, f, sessions)
    grids: list[AssetGrid] = []
    closes: dict[int, tuple] = {}
    t0 = time.monotonic()
    spx = set(inp.membership["asset_id"].dropna().astype(int))
    for aid, cls in prep.cls.items():
        aid = int(aid)
        if cls not in f.build.asset_classes or aid not in prep.bars:
            continue
        frame = prep.frame(aid, f, ctx=ctx[cls], sessions=sessions,
                           with_facts=(cls == "equity"))  # fmt: skip
        if frame is None:
            continue
        if cls == "equity" and aid in spx:
            closes[aid] = (frame.dates, frame.close, frame.ready_ns)
        g = asset_grid(frame, aid, cls, cat, grid, f)
        if g.active.any():
            grids.append(g)
    log.info("build_assets_done", extra={"assets": len(grids),
                                         "secs": round(time.monotonic() - t0, 1)})  # fmt: skip

    # -- market inputs
    prices = market_prices(prep, f, sessions)
    breadth = breadth_inputs(closes, inp.membership, sessions, f)
    events = EventInputs(fomc=inp.fomc, cpi=inp.cpi, earnings=inp.earnings,
                         membership=inp.membership)  # fmt: skip
    hin = hmm.HmmInputs(inp.fits, f.market, f.regime.hmm)
    ms = inputs.MacroState(inp.macro)
    ns_groups = set(f.build.nonsession_market_groups)
    mk_session = cat.market
    mk_nonsession = [s for s in cat.market if s.group in ns_groups]
    tsz_base = {normalize.base_key(s): s for s in cat.market_tsz}
    hist: dict[tuple[str, int], list[float]] = {k: [] for k in tsz_base}
    tsz_keys = [s.key for s in cat.market_tsz]
    by_t = {int(t): i for i, t in enumerate(grid.t)}
    sectors = pd.Series(prep.sector)
    groups = {s.key: s.group for s in cat.all()}
    w = Writer(conn, run_id, f.build.write_batch_rows,
               f.build.compress if compress is None else compress, groups, selected)  # fmt: skip
    last_month = None
    for wi, d in enumerate(wgrid.days):
        t = pd.Timestamp(int(wgrid.t[wi]), tz=UTC)
        session = bool(wgrid.session[wi])
        ms.advance(t)
        m = market.MarketState(asof=t, macro=ms, prices=prices, sessions=sessions, cfg=f.market,
                               breadth=breadth, events=events, crypto=inp.crypto,
                               regime=hin)  # fmt: skip
        specs = mk_session if session else mk_nonsession
        mv = market.compute(m, specs)
        if session:
            for k in hist:
                hist[k].append(mv.get(k, math.nan))
        gi = by_t.get(int(wgrid.t[wi]))
        if gi is None:
            continue  # warm-up
        iso = t.isoformat()
        month = (d.year, d.month)
        if last_month is not None and month != last_month:
            w.compress_before(t - pd.Timedelta(days=1))
        last_month = month
        keys = list(mv)
        w.add(iso, None, keys, [mv[k] for k in keys])
        if session:
            zv = [tsz_base_fn(tsz_base[k], hist[k]) for k in tsz_base]
            w.add(iso, None, tsz_keys, zv)
            if selected is None:
                w.add_regime(regime_mod.regime_row(m, f.market, f.regime))
                res.regime_rows += 1
        # -- assets at T
        members = inputs.members_at(inp.membership, t.to_pydatetime()) if session else set()
        panel_rows = {}
        for g in grids:
            if not g.active[gi]:
                continue
            if g.asset_class == "equity" and g.asset_id not in members:
                continue
            row = g.row(gi)
            w.add(iso, g.asset_id, g.keys, row)
            if g.asset_class == "equity":
                panel_rows[g.asset_id] = dict(zip((k[0] for k in g.keys), row, strict=True))
        if session and panel_rows and cat.xs:
            vals = pd.DataFrame.from_dict(panel_rows, orient="index").astype(float)
            panel = cross_section.Panel(vals, sectors, f.cross_section)
            xs = cross_section.compute(panel, cat.xs)
            xkeys = list(xs.columns)
            for aid, r in zip(xs.index, xs.to_numpy(), strict=True):
                w.add(iso, int(aid), xkeys, r)
    w.flush()
    res.rows_offered, res.rows_written = w.offered, w.written
    res.nan_rate = w.nan_rate()
    res.secs = time.monotonic() - t_start
    log.info("build_done", extra={"since": str(since), "until": str(until),
                                  "offered": res.rows_offered, "written": res.rows_written,
                                  "secs": round(res.secs, 1)})  # fmt: skip
    return res


def tsz_base_fn(spec: registry.FeatureSpec, history: list[float]) -> float:
    return float(spec.fn(np.asarray(history, dtype=float)))


# ---------------------------------------------------------------- entry points
def run_daily(settings: Settings, connect, now: datetime | None = None,
              cal: TradingCalendar | None = None) -> BuildResult | None:  # fmt: skip
    """Today's decision time (the scheduler calls this at ``run_et``)."""
    from qt.common.runs import run_context

    now = now or datetime.now(UTC)
    d = pd.Timestamp(now).tz_convert(ET).date()
    with run_context("features", settings, connect) as run, connect() as conn:
        registry.sync(conn)
        res = build(conn, settings, d, d, run_id=run.run_id, compress=False, cal=cal)
        conn.commit()
        return res


def run_backfill(settings: Settings, connect, since: date, until: date,
                 only=None) -> BuildResult:  # fmt: skip
    from qt.common.runs import run_context

    cat = catalog_mod.register_all(settings.config.features)
    if only is not None:
        select_features(cat, only)  # a bad pattern fails here, before a run row or any load
    with run_context("features_backfill", settings, connect) as run, connect() as conn:
        registry.sync(conn)
        conn.commit()
        res = build(conn, settings, since, until, run_id=run.run_id, cat=cat, only=only)
        conn.commit()
        return res


def run_refit(settings: Settings, connect, now: datetime | None = None) -> list[dict]:
    """Fit every missing monthly HMM up to ``now``; writes the stability report if any is new."""
    from qt.common import alerts
    from qt.common.runs import run_context

    now = now or datetime.now(UTC)
    f = settings.config.features
    with run_context("hmm_refit", settings, connect) as run, connect() as conn:
        cat = catalog_mod.register_all(f)
        inp = load_inputs(conn, settings, now, cat)
        _, new = ensure_fits(conn, settings, inp, now, run_id=run.run_id, alert=alerts.alert)
        conn.commit()
        if new:
            p = stability_report(conn, settings, pd.Timestamp(now).tz_convert(ET).date())
            log.info("hmm_stability_report", extra={"path": str(p), "new_fits": len(new)})
        return new


def main(argv: list[str] | None = None) -> int:
    from qt.common import config
    from qt.common import logging as qtlog
    from qt.db import conn as dbconn

    ap = argparse.ArgumentParser(prog="python -m qt.features.build")
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--until", type=date.fromisoformat)
    ap.add_argument("--daily", action="store_true", help="today's decision time only")
    ap.add_argument("--refit-hmm", action="store_true", help="fit missing monthly HMMs first")
    ap.add_argument("--only", help="backfill only: write just these feature names "
                    "(comma-separated, fnmatch globs allowed); regime rows are skipped")
    a = ap.parse_args(argv)
    if a.only is not None and a.daily:
        ap.error("--only is backfill-only and cannot be combined with --daily")
    if a.only is not None and not a.since:
        ap.error("--only needs --since (backfill)")
    settings = config.load()
    connect = lambda: dbconn.connect(settings)  # noqa: E731
    qtlog.setup(settings, connect=connect)
    if a.refit_hmm:
        new = run_refit(settings, connect)
        print(f"hmm: {len(new)} new fits")
    if a.daily:
        res = run_daily(settings, connect)
    elif a.since:
        until = a.until or pd.Timestamp.now(tz=ET).date()
        if a.only is not None:
            try:
                select_features(catalog_mod.register_all(settings.config.features), a.only)
            except ValueError as e:
                print(e, file=sys.stderr)
                return 2
        res = run_backfill(settings, connect, a.since, until, only=a.only)
    else:
        return 0 if a.refit_hmm else ap.print_usage() or 2
    if res is not None:
        print(f"features {res.since}..{res.until}: {res.rows_written:,} rows written "
              f"({res.rows_offered:,} offered), {res.regime_rows} regime rows, "
              f"{res.secs:.0f}s")  # fmt: skip
        for g, r in res.nan_rate.items():
            print(f"  {g:28s} NaN {r:6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
