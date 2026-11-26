"""Crypto structure (component doc 03 section 5.3; Phase 3 step 4b, D-039).

Functions of a :class:`qt.features.market.MarketState` whose ``crypto`` field holds a
:class:`CryptoInputs`. Every frame is cut at T inside the feature (``released_at <= T``).
Windows are calendar days; "n days ago" is the newest observation dated between ``n`` and
``n + match_tolerance_days`` days before the newest one.

History is short for some inputs, so these are NaN (and masked in step 5) until it builds up:
BTC dominance and OI start with our first CoinGecko ``/global`` and OI snapshots (2026-11-26),
ETF flows wait for a SoSoValue key. Stablecoin supply is used from 2020 (``valid_from``: DefiLlama's
earlier totals miss most of the market), Hyperliquid funding from 2023-06 and OKX funding from
2026-08 (a venue counts once its settlements cover the window).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from qt.common.config import MarketCfg
from qt.features import registry
from qt.features.inputs import ET
from qt.features.market import MarketState

G_CRYPTO_STRUCT = "crypto_structure"
EMPTY = pd.DataFrame(columns=["subject", "venue", "symbol", "ts", "value", "funding_rate",
                              "funding_interval_h", "oi_usd", "released_at"])  # fmt: skip


@dataclass
class CryptoInputs:
    dominance: pd.DataFrame  # coingecko btc_dominance_pct: ts, value, released_at
    mcap: pd.DataFrame  # coingecko market_cap_usd: subject, ts, value, released_at
    stablecoins: pd.DataFrame  # defillama stablecoin_supply_usd
    etf: pd.DataFrame  # sosovalue total_net_inflow: subject, ts, value, released_at
    funding: pd.DataFrame  # inputs.load_derivs(kind='funding')
    oi: pd.DataFrame  # inputs.load_derivs(kind='oi')


def _known(m: MarketState, name: str, subject: str | None = None) -> pd.DataFrame:
    key = ("cx", name, subject)
    if key not in m._cache:
        df = getattr(m.crypto, name) if m.crypto is not None else None
        if df is None:
            df = EMPTY
        df = df[df["released_at"] <= m.asof]
        start = m.cfg.crypto.valid_from.get(name)
        if start is not None:
            df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
        if subject is not None:
            col = "symbol" if "symbol" in df.columns else "subject"
            df = df[df[col] == subject]
        m._cache[key] = df.sort_values("ts", kind="stable")
    return m._cache[key]


def _latest(m: MarketState, df: pd.DataFrame) -> pd.Series | None:
    """The newest observation, if it was released within ``max_age_days`` of T."""
    if df.empty:
        return None
    r = df.iloc[-1]
    if m.asof - r["released_at"] > pd.Timedelta(days=m.cfg.crypto.max_age_days):
        return None
    return r


def _ago(m: MarketState, df: pd.DataFrame, ts: pd.Timestamp, days: int) -> float:
    lo = ts - pd.Timedelta(days=days + m.cfg.crypto.match_tolerance_days)
    hit = df[(df["ts"] <= ts - pd.Timedelta(days=days)) & (df["ts"] >= lo)]
    return float(hit["value"].iloc[-1]) if len(hit) else math.nan


def level(m: MarketState, name: str, subject: str | None = None) -> float:
    r = _latest(m, _known(m, name, subject))
    return math.nan if r is None else float(r["value"])


def change(m: MarketState, name: str, days: int, pct: bool) -> float:
    df = _known(m, name)
    r = _latest(m, df)
    if r is None:
        return math.nan
    past = _ago(m, df, r["ts"], days)
    if pct:
        return float(r["value"]) / past - 1 if past > 0 else math.nan
    return float(r["value"]) - past


def funding_ann(m: MarketState, sym: str) -> float:
    """Mean over venues of the time-averaged annualized funding over ``funding_days``.

    Per venue: sum of settled rates in (T - n days, T] x 8760 / hours covered, if the settlements
    cover at least ``funding_min_coverage`` of the window.
    """
    c = m.cfg.crypto
    df = _known(m, "funding", sym)
    df = df[df["ts"] > m.asof - pd.Timedelta(days=c.funding_days)]
    vals = []
    for v in c.funding_venues:
        x = df[df["venue"] == v]
        hours = float(x["funding_interval_h"].sum())
        if hours >= c.funding_min_coverage * 24 * c.funding_days:
            vals.append(float(x["funding_rate"].sum()) * 8760.0 / hours)
    return float(sum(vals) / len(vals)) if vals else math.nan


def oi_mcap(m: MarketState, sym: str) -> float:
    """USD open interest summed over every configured venue / the coin's market cap."""
    c = m.cfg.crypto
    df = _known(m, "oi", sym)
    df = df[df["ts"] >= m.asof - pd.Timedelta(hours=c.oi_max_age_h)]
    tot = 0.0
    for v in c.oi_venues:
        x = df[df["venue"] == v]
        if x.empty or not math.isfinite(float(x["oi_usd"].iloc[-1])):
            return math.nan
        tot += float(x["oi_usd"].iloc[-1])
    cap = level(m, "mcap", sym)
    return tot / cap if cap > 0 else math.nan


def etf_flow(m: MarketState, sym: str) -> float:
    """Net ETF flows over the last ``etf_days`` trade dates, USD bn."""
    c = m.cfg.crypto
    df = _known(m, "etf", c.etf_subject.format(symbol=sym))
    df = df.drop_duplicates("ts", keep="last")
    if len(df) < c.etf_days:
        return math.nan
    last = df.iloc[-c.etf_days :]
    today = pd.Timestamp(m.asof.tz_convert(ET).date())
    oldest = pd.Timestamp(last["ts"].iloc[0].tz_convert("UTC").date())
    if (today - oldest).days > c.etf_window_days:
        return math.nan
    return float(last["value"].sum()) / 1e9


def register(cfg: MarketCfg) -> list[registry.FeatureSpec]:
    """Register the crypto-structure features (idempotent). Returns the specs."""
    c = cfg.crypto
    keys: list[tuple[str, int]] = []

    def reg(name, formula, sources, fn, window=None):
        registry.feature(name, 1, kind="market", group=G_CRYPTO_STRUCT, formula=formula,
                         window=window, sources=tuple(sources))(fn)  # fmt: skip
        keys.append((name, 1))

    n_d, n_s, n_f = c.dominance_chg_days, c.stable_chg_days, c.funding_days
    reg("btc_dominance", "CoinGecko BTC market-cap dominance, %", ["coingecko:btc_dominance_pct"],
        lambda m: level(m, "dominance"))  # fmt: skip
    reg(f"btc_dominance_chg{n_d}", f"BTC dominance - its value {n_d} days earlier (pp)",
        ["coingecko:btc_dominance_pct"], lambda m: change(m, "dominance", n_d, pct=False),
        n_d)  # fmt: skip
    reg(f"stablecoin_supply_chg{n_s}", f"total stablecoin supply / {n_s} days earlier - 1",
        ["defillama:stablecoin_supply_usd"], lambda m: change(m, "stablecoins", n_s, pct=True),
        n_s)  # fmt: skip
    for sym in c.symbols:
        s = sym.lower()
        reg(f"funding_{n_f}d_ann_{s}",
            f"{sym} perp funding: mean over {'/'.join(c.funding_venues)} of sum of rates over "
            f"{n_f} days x 8760 / hours covered (>= {c.funding_min_coverage:.0%})",
            [f"derivs:funding:{sym}"], lambda m, sym=sym: funding_ann(m, sym), n_f)  # fmt: skip
        reg(f"oi_mcap_{s}",
            f"{sym} perp OI in USD summed over {'/'.join(c.oi_venues)} / market cap",
            [f"derivs:oi:{sym}", "coingecko:market_cap_usd"],
            lambda m, sym=sym: oi_mcap(m, sym))  # fmt: skip
        reg(f"etf_flow_{c.etf_days}d_{s}",
            f"{sym} spot ETF net inflow over the last {c.etf_days} trade dates, USD bn",
            [f"sosovalue:{c.etf_subject.format(symbol=sym)}"],
            lambda m, sym=sym: etf_flow(m, sym), c.etf_days)  # fmt: skip
    return [registry.get(*k) for k in keys]
