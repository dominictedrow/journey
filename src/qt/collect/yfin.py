"""yfinance daily series -> raw_prices, source 'yfinance' (component doc 6.2, Q3).

The spec tickers (SPY, NQ=F, GC=F, DX-Y.NYB, ^VIX, ^VIX3M) daily, plus QQQ and the SPDR sector
ETFs from 2000 for the HMM. Never mixed with Alpaca rows: same ``asset_id`` for an ETF, but a
different ``source``, and Phase 3 picks sources explicitly.

- ``ts`` = the trade date at 00:00 ET (same convention as Alpaca equity bars).
- ``released_at``: XNYS-listed = session close + 15 min (VIX settles 16:15 ET); futures/FX =
  17:00 ET on the trade date. A row not yet released (today's bar in progress; futures show
  the current trade date all day) is skipped and fetched on a later run.
- ``close`` is yfinance's ``Close`` (split-adjusted, not dividend-adjusted). ``vwap`` and
  ``trade_count`` are null.
- Fragile source: every response is cached raw under ``yfinance.cache_dir`` and tickers are
  spaced by ``sleep_s``. If it fails, FRED ``VIXCLS`` still covers VIX (non-critical).
"""

from __future__ import annotations

import gzip
import logging
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg

from qt.collect import pit
from qt.collect.bars import PRICE_COLS
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.common.calendar import ET
from qt.common.config import YfTicker
from qt.common.paths import repo_path

log = logging.getLogger("qt.collect.yfinance")

Fetch = Callable[[str, date], Any]  # (ticker, start) -> pandas DataFrame indexed by date


def _now() -> datetime:
    return datetime.now(UTC)


def _yf_fetch(ticker: str, start: date) -> Any:
    import yfinance as yf

    return yf.Ticker(ticker).history(
        start=start.isoformat(), interval="1d", auto_adjust=False, actions=False
    )


def asset_id(conn: psycopg.Connection, ticker: str, t: YfTicker) -> int:
    """The asset row for a yfinance ticker: an existing ETF (Alpaca's row) or a series row."""
    row = conn.execute(
        "SELECT asset_id FROM assets WHERE ticker = %s AND asset_class = %s "
        "AND security_key IS NULL ORDER BY alpaca_id NULLS LAST, asset_id LIMIT 1",
        (ticker, t.asset_class),
    ).fetchone()
    if row:
        return row[0]
    return conn.execute(
        "INSERT INTO assets (ticker, asset_class, name) VALUES (%s, %s, %s) RETURNING asset_id",
        (ticker, t.asset_class, f"{ticker} (yfinance)"),
    ).fetchone()[0]


def _f(v: Any) -> float | None:
    if v is None:
        return None
    x = float(v)
    return None if math.isnan(x) else x


def yf_rows(aid: int, df: Any, t: YfTicker, ctx: RunCtx, now: datetime) -> tuple[list[tuple], int]:
    pc = ctx.settings.config.collect.pit
    out, bad = [], 0
    for idx, r in df.iterrows():
        d = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
        if t.rule == "xnys":
            if not ctx.cal.is_session(d):
                bad += 1
                continue
            rel = pit.yfinance_xnys(ctx.cal, pc, d)
        else:
            rel = pit.futures_fx(pc, d)
        if rel > now:
            continue
        close = _f(r.get("Close"))
        if close is None:
            bad += 1
            continue
        ts = datetime.combine(d, datetime.min.time(), ET).astimezone(UTC)
        out.append((aid, "1d", ts, _f(r.get("Open")), _f(r.get("High")), _f(r.get("Low")),
                    close, _f(r.get("Volume")), None, None, "yfinance", rel))  # fmt: skip
    return out, bad


def cache_raw(cache_dir: str, ticker: str, df: Any, now: datetime) -> None:
    d = repo_path(cache_dir) / re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"{now:%Y%m%dT%H%M%SZ}.csv.gz", "wt", encoding="utf-8") as fh:
        df.to_csv(fh)


class YfinanceCollector:
    name = "yfinance"
    required_secrets: tuple[str, ...] = ()

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        fetch: Fetch = _yf_fetch,
        sleep: Callable[[float], None] = time.sleep,
        cache: bool = True,
    ) -> None:
        self.clock, self.fetch, self.sleep, self.cache = clock, fetch, sleep, cache

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.yfinance
        now = self.clock()
        res = CollectResult(cursor=dict(ctx.cursor))
        failed: dict[str, str] = {}
        for i, (ticker, t) in enumerate(cfg.tickers.items()):
            if i:
                self.sleep(cfg.sleep_s)
            try:
                aid = asset_id(ctx.conn, ticker, t)
                last = ctx.conn.execute(
                    "SELECT max(ts) FROM raw_prices WHERE asset_id = %s AND source = 'yfinance' "
                    "AND timeframe = '1d'", (aid,),
                ).fetchone()[0]  # fmt: skip
                start = (
                    last.astimezone(ET).date() - timedelta(days=cfg.refetch_days)
                    if last else date.fromisoformat(cfg.start)
                )  # fmt: skip
                df = self.fetch(ticker, start)
                if df is None or len(df) == 0:
                    raise RuntimeError("empty response")
                if self.cache:
                    cache_raw(cfg.cache_dir, ticker, df, now)
                rows, bad = yf_rows(aid, df, t, ctx, now)
                res.add("raw_prices", insert_rows(ctx.conn, "raw_prices", PRICE_COLS, rows))
                res.rejected += bad
                ctx.conn.commit()
            except Exception as e:  # noqa: BLE001  (one ticker failing never stops the rest)
                ctx.conn.rollback()
                failed[ticker] = f"{type(e).__name__}: {e}"
                log.warning("yfinance_failed", extra={"ticker": ticker, "error": failed[ticker]})
        if failed:
            raise RuntimeError(f"{len(failed)} yfinance tickers failed: {failed}")
        return res
