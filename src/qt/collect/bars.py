"""Alpaca daily bars -> raw_prices (component doc 6.1, D-024, D-030).

``bars_equity`` covers every equity/ETF asset: S&P 500 members since 2016 (delisted ones too, up
to ``listed_until``, so there's no survivorship bias) and the configured ETFs. ``bars_crypto``
covers the crypto pool. Backfill and the daily run are one code path: each asset resumes from
its newest stored bar.

- Equity bars: ``released_at`` = that session's XNYS close; crypto bars: bar start + 1 day.
- A bar whose ``released_at`` is still in the future is today's bar in progress. It's skipped
  (the table is append-only, so a partial bar could never be corrected) and fetched next run.
- Per-symbol requests for history (renamed securities resolve correctly one symbol at a time);
  daily increments are batched, since the assets then are distinct, live securities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg

from qt.collect import pit
from qt.collect.alpaca import Alpaca, Bar
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.common.calendar import ET, TradingCalendar

log = logging.getLogger("qt.collect.bars")

PRICE_COLS = [
    "asset_id", "timeframe", "ts", "open", "high", "low", "close", "volume", "vwap",
    "trade_count", "source", "released_at",
]  # fmt: skip
BATCH_MAX_GAP_DAYS = 30  # resume points closer than this to today are batched


def _now() -> datetime:
    return datetime.now(UTC)


def _default_alpaca(ctx: RunCtx) -> Alpaca:
    return Alpaca.from_settings(ctx.settings, ctx.http)


def last_bars(conn: psycopg.Connection, source: str, asset_class: list[str]) -> dict[int, datetime]:
    rows = conn.execute(
        "SELECT p.asset_id, max(p.ts) FROM raw_prices p JOIN assets a USING (asset_id) "
        "WHERE p.source = %s AND p.timeframe = '1d' AND a.asset_class = ANY(%s) "
        "GROUP BY p.asset_id",
        (source, asset_class),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def equity_rows(
    asset_id: int,
    bars: list[Bar],
    source: str,
    cal: TradingCalendar,
    now: datetime,
    until: date | None,
) -> tuple[list[tuple], int]:
    """Rows for raw_prices; also returns how many bars were dropped as untimeable."""
    out, bad = [], 0
    for b in bars:
        d = b.session
        if until is not None and d > until:
            continue  # after delisting: filler bars or a reused symbol
        if not cal.is_session(d):
            bad += 1
            continue
        rel = pit.stock_bar(cal, d)
        if rel > now:
            continue  # in progress
        out.append((asset_id, "1d", b.t, b.o, b.h, b.low, b.c, b.v, b.vw, b.n, source, rel))
    return out, bad


def crypto_rows(asset_id: int, bars: list[Bar], source: str, now: datetime) -> tuple[list, int]:
    out, bad = [], 0
    for b in bars:
        try:
            rel = pit.crypto_bar(b.t)
        except pit.UntimeableRow:
            bad += 1
            continue
        if rel > now:
            continue
        out.append((asset_id, "1d", b.t, b.o, b.h, b.low, b.c, b.v, b.vw, b.n, source, rel))
    return out, bad


class BarsEquityCollector:
    name = "bars_equity"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        alpaca: Callable[[RunCtx], Any] = _default_alpaca,
        tickers: list[str] | None = None,
    ) -> None:
        self.clock, self.make_alpaca, self.tickers = clock, alpaca, tickers

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.alpaca
        api = self.make_alpaca(ctx)
        now = self.clock()
        today = now.astimezone(ET).date()
        start0 = date.fromisoformat(cfg.prices_start)
        assets = ctx.conn.execute(
            "SELECT asset_id, ticker, listed_until FROM assets "
            "WHERE asset_class IN ('equity', 'etf') ORDER BY ticker, asset_id"
        ).fetchall()
        if self.tickers is not None:
            assets = [a for a in assets if a[1] in self.tickers]
        res = CollectResult(cursor=dict(ctx.cursor))
        failed: dict[str, str] = {}
        for adjustment, source in cfg.adjustments.items():
            last = last_bars(ctx.conn, source, ["equity", "etf"])
            todo = []  # (asset_id, ticker, start, end)
            for aid, ticker, listed_until in assets:
                start = (
                    (last[aid].astimezone(ET).date() + timedelta(days=1)) if aid in last else start0
                )
                end = min(listed_until, today) if listed_until else today
                if start <= end:
                    todo.append((aid, ticker, start, end, listed_until))
            # batch only increments of live securities that already have history
            near = [t for t in todo if t[0] in last and t[4] is None
                    and (today - t[2]).days <= BATCH_MAX_GAP_DAYS]  # fmt: skip
            far = [t for t in todo if t not in near]
            if len({t[1] for t in near}) != len(near):  # same ticker twice: no batching
                far, near = far + near, []

            def store(aid: int, bars: list[Bar], until_: date | None, src: str = source) -> None:
                rows, bad = equity_rows(aid, bars, src, ctx.cal, now, until_)
                res.add("raw_prices", insert_rows(ctx.conn, "raw_prices", PRICE_COLS, rows))
                res.rejected += bad

            for i in range(0, len(near), cfg.multi_symbols):
                chunk = near[i : i + cfg.multi_symbols]
                first = min(t[2] for t in chunk)
                try:
                    got = api.stock_bars_multi([t[1] for t in chunk], first, today, adjustment)
                except Exception as e:  # noqa: BLE001
                    ctx.conn.rollback()
                    failed[f"batch@{chunk[0][1]}"] = f"{type(e).__name__}: {e}"
                    continue
                for aid, ticker, start, _, lu in chunk:
                    store(aid, [b for b in got.get(ticker, []) if b.session >= start], lu)
                ctx.conn.commit()
            for aid, ticker, start, end, lu in far:
                try:
                    store(aid, api.stock_bars(ticker, start, end, adjustment), lu)
                except Exception as e:  # noqa: BLE001  (one symbol failing never stops the rest)
                    ctx.conn.rollback()
                    failed[ticker] = f"{type(e).__name__}: {e}"
                    log.warning(
                        "bars_symbol_failed", extra={"ticker": ticker, "error": failed[ticker]}
                    )
                    continue
                ctx.conn.commit()  # resumable: each symbol is durable
            res.cursor[f"{source}_requested"] = len(todo)
        if failed:
            raise RuntimeError(
                f"{len(failed)} bar requests failed: {dict(list(failed.items())[:10])}"
            )
        return res


class BarsCryptoCollector:
    name = "bars_crypto"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        alpaca: Callable[[RunCtx], Any] = _default_alpaca,
    ) -> None:
        self.clock, self.make_alpaca = clock, alpaca

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.alpaca
        api = self.make_alpaca(ctx)
        now = self.clock()
        start0 = date.fromisoformat(cfg.crypto_start)
        pool = ctx.conn.execute(
            "SELECT asset_id, ticker FROM assets WHERE asset_class = 'crypto' AND alpaca_tradable "
            "ORDER BY ticker"
        ).fetchall()
        if not pool:
            raise RuntimeError("no tradable crypto assets: run crypto_assets first")
        last = last_bars(ctx.conn, "alpaca", ["crypto"])
        res = CollectResult(cursor=dict(ctx.cursor))
        by_start: dict[date, list[tuple[int, str]]] = {}
        for aid, sym in pool:
            s = (last[aid] + timedelta(days=1)).date() if aid in last else start0
            by_start.setdefault(s, []).append((aid, sym))
        for start, group in sorted(by_start.items()):
            for i in range(0, len(group), cfg.multi_symbols):
                chunk = group[i : i + cfg.multi_symbols]
                got = api.crypto_bars([s for _, s in chunk], start)
                for aid, sym in chunk:
                    rows, bad = crypto_rows(aid, got.get(sym, []), "alpaca", now)
                    res.add("raw_prices", insert_rows(ctx.conn, "raw_prices", PRICE_COLS, rows))
                    res.rejected += bad
                ctx.conn.commit()
        return res


class SuccessorBarsCollector:
    """Real history for a reused-symbol security, fetched under its successor symbol (D-035).

    For each ``collect.identity_overrides`` entry with a ``successor``, request
    ``successor.symbol`` over ``[start, end]`` and store the bars under the overridden security's
    asset as source ``alpaca_succ`` (same release rule as any Alpaca stock bar). A finished entry
    is recorded in the cursor and skipped afterwards; re-running is a no-op (ON CONFLICT).
    """

    name = "succ_bars"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")
    source = "alpaca_succ"

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        alpaca: Callable[[RunCtx], Any] = _default_alpaca,
    ) -> None:
        self.clock, self.make_alpaca = clock, alpaca

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        c = ctx.settings.config.collect
        now = self.clock()
        cursor = dict(ctx.cursor)
        done = dict(cursor.get("done") or {})
        res = CollectResult(cursor=cursor)
        todo = [o for o in c.identity_overrides if o.successor is not None]
        api = None
        for o in todo:
            s = o.successor
            sig = f"{s.symbol}:{s.start}:{s.end}"
            if done.get(o.security_key) == sig:
                continue
            row = ctx.conn.execute(
                "SELECT asset_id FROM assets WHERE security_key = %s", (o.security_key,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"identity override for unknown security {o.security_key}")
            api = api or self.make_alpaca(ctx)
            bars = api.stock_bars(s.symbol, s.start, s.end)
            rows, bad = equity_rows(row[0], bars, self.source, ctx.cal, now, s.end)
            res.add("raw_prices", insert_rows(ctx.conn, "raw_prices", PRICE_COLS, rows))
            res.rejected += bad
            ctx.conn.commit()
            done[o.security_key] = sig
            log.info(
                "successor_bars_done",
                extra={"security_key": o.security_key, "symbol": s.symbol, "bars": len(rows)},
            )
        cursor["done"] = done
        return res
