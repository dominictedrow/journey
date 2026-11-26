"""Crypto perpetuals: funding history and OI/mark snapshots -> raw_crypto_derivs (6.10, Q9).

Raw per venue; cross-venue aggregates are Phase 3 features. ``symbol`` is the base coin (BTC),
so venues line up; the venue instrument names live in config.

**Funding (``funding``, critical).** Settled rates, ``ts`` = ``released_at`` = settlement time.

- OKX ``funding-rate-history`` (8 h, ~3 months served): paged backwards with ``after`` until we
  reach what's stored. ``realizedRate`` (what was actually paid) when present, else
  ``fundingRate``. ``funding_interval_h`` = hours since the previous settlement (OKX shortens
  the interval in volatile markets), else the configured default.
- Hyperliquid ``fundingHistory`` (1 h): paged forwards from the newest stored settlement, or
  ``derivs.funding_start``.

**OI (``crypto_oi``, non-critical per venue).** One snapshot per run from OKX, Hyperliquid,
Deribit and Kraken Futures: ``ts`` = ``released_at`` = fetch time. ``oi_contracts`` holds the
venue's native OI figure (OKX: contracts; Hyperliquid, Kraken, Deribit linear: coins; Deribit
inverse: USD); ``oi_usd`` is OKX's own USD figure, else native x mark (:func:`to_usd`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows

log = logging.getLogger("qt.collect.derivs")

DERIV_COLS = [
    "venue", "symbol", "ts", "kind", "funding_rate", "funding_interval_h", "oi_contracts",
    "oi_usd", "mark_price", "released_at",
]  # fmt: skip
HL_PAGE = 500  # Hyperliquid returns at most 500 funding rows per request


def _now() -> datetime:
    return datetime.now(UTC)


def _ms(v: int | float | str) -> datetime:
    return datetime.fromtimestamp(int(v) / 1000, UTC)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def annualize(rate: float, interval_h: float) -> float:
    """A per-interval funding rate as a simple annual rate (8 h: x 1095; 1 h: x 8760)."""
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    return rate * (24.0 / interval_h) * 365.0


def to_usd(native: float, unit: str, mark: float) -> float:
    """Native OI in ``unit`` ('usd' or 'base' coin) to USD at the mark price."""
    if unit == "usd":
        return native
    if unit == "base":
        return native * mark
    raise ValueError(f"unknown OI unit {unit!r}")


def stored_last(ctx: RunCtx, venue: str, kind: str) -> dict[str, datetime]:
    rows = ctx.conn.execute(
        "SELECT symbol, max(ts) FROM raw_crypto_derivs WHERE venue = %s AND kind = %s "
        "GROUP BY symbol",
        (venue, kind),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def funding_rows(
    venue: str,
    symbol: str,
    settlements: list[tuple[datetime, float]],
    prev: datetime | None,
    default_h: float,
    now: datetime,
) -> list[tuple]:
    """Rows for settled funding newer than ``prev``; interval = hours since the previous
    settlement, or ``default_h`` for the first one and across data gaps (> 2x the default)."""
    out = []
    for ts, rate in sorted(settlements):
        if ts > now or (prev is not None and ts <= prev):
            continue  # not settled yet, or already stored
        h = (ts - prev).total_seconds() / 3600 if prev is not None else default_h
        if h > 2 * default_h:
            h = default_h
        out.append((venue, symbol, ts, "funding", rate, h, None, None, None,
                    pit.funding_settlement(ts)))  # fmt: skip
        prev = ts
    return out


# ---------------------------------------------------------------- funding
class FundingCollector:
    name = "funding"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.derivs
        now = self.clock()
        res = CollectResult(cursor=dict(ctx.cursor))
        failed: dict[str, str] = {}
        for venue, fn in (("okx", self._okx), ("hyperliquid", self._hl)):
            last = stored_last(ctx, venue, "funding")
            for sym in cfg.symbols:
                try:
                    rows = fn(ctx, sym, last.get(sym), now)
                    stats = insert_rows(ctx.conn, "raw_crypto_derivs", DERIV_COLS, rows)
                    res.add("raw_crypto_derivs", stats)
                    ctx.conn.commit()
                except Exception as e:  # noqa: BLE001
                    ctx.conn.rollback()
                    failed[f"{venue}:{sym}"] = f"{type(e).__name__}: {e}"
                    log.warning(
                        "funding_failed",
                        extra={"venue": venue, "symbol": sym, "error": failed[f"{venue}:{sym}"]},
                    )
        if failed:
            raise RuntimeError(f"{len(failed)} funding feeds failed: {failed}")
        return res

    @staticmethod
    def _okx(ctx: RunCtx, sym: str, last: datetime | None, now: datetime) -> list[tuple]:
        c = ctx.settings.config.collect.derivs.okx
        got: dict[datetime, float] = {}
        after: str | None = None
        while True:
            q: dict[str, Any] = {"instId": c.inst[sym], "limit": c.page_limit}
            if after:
                q["after"] = after
            j = ctx.http.get(f"{c.base_url}/funding-rate-history", params=q).json()
            if j.get("code") not in ("0", 0):
                raise RuntimeError(f"OKX error {j.get('code')}: {j.get('msg')}")
            data = j.get("data") or []
            for d in data:
                rate = d.get("realizedRate") or d.get("fundingRate")
                got[_ms(d["fundingTime"])] = float(rate)
            if not data:
                break
            oldest = min(int(d["fundingTime"]) for d in data)
            if last is not None and _ms(oldest) <= last:
                break
            after = str(oldest)
        # the stored row before the first new one gives the first interval
        return funding_rows("okx", sym, list(got.items()), last, c.default_interval_h, now)

    @staticmethod
    def _hl(ctx: RunCtx, sym: str, last: datetime | None, now: datetime) -> list[tuple]:
        cfg = ctx.settings.config.collect.derivs
        c = cfg.hyperliquid
        start = last + timedelta(milliseconds=1) if last else datetime.fromisoformat(
            cfg.funding_start).replace(tzinfo=UTC)  # fmt: skip
        got: list[tuple[datetime, float]] = []
        while start < now:
            data = ctx.http.post(c.url, json={"type": "fundingHistory", "coin": sym,
                                              "startTime": _to_ms(start)}).json()  # fmt: skip
            if not data:
                break
            got += [(_ms(d["time"]), float(d["fundingRate"])) for d in data]
            newest = max(int(d["time"]) for d in data)
            if len(data) < HL_PAGE:
                break
            start = _ms(newest + 1)
        # Hyperliquid settles on the hour but stamps a few ms after it: floor to the second
        clean = [(t.replace(microsecond=0), r) for t, r in got]
        return funding_rows("hyperliquid", sym, clean, last, c.interval_h, now)


# ---------------------------------------------------------------- open interest
def oi_row(venue: str, sym: str, now: datetime, native: float, usd: float,
           mark: float) -> tuple:  # fmt: skip
    return (venue, sym, now, "oi", None, None, native, usd, mark, pit.first_seen(now))


class CryptoOiCollector:
    name = "crypto_oi"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.derivs
        now = self.clock().replace(microsecond=0)
        res = CollectResult(cursor=dict(ctx.cursor))
        failed: dict[str, str] = {}
        venues: dict[str, Callable[[RunCtx, str, datetime], tuple]] = {
            "okx": self._okx,
            "deribit": self._deribit,
            "kraken": self._kraken,
        }
        rows: list[tuple] = []
        for venue, fn in venues.items():
            for sym in cfg.symbols:
                try:
                    rows.append(fn(ctx, sym, now))
                except Exception as e:  # noqa: BLE001
                    failed[f"{venue}:{sym}"] = f"{type(e).__name__}: {e}"
        try:
            rows += self._hl(ctx, cfg.symbols, now)
        except Exception as e:  # noqa: BLE001
            failed["hyperliquid"] = f"{type(e).__name__}: {e}"
        res.add("raw_crypto_derivs", insert_rows(ctx.conn, "raw_crypto_derivs", DERIV_COLS, rows))
        ctx.conn.commit()
        if failed:
            log.warning("oi_venues_failed", extra={"failed": failed})
            raise RuntimeError(f"{len(failed)} OI snapshots failed: {failed}")
        return res

    @staticmethod
    def _okx(ctx: RunCtx, sym: str, now: datetime) -> tuple:
        c = ctx.settings.config.collect.derivs.okx
        inst = c.inst[sym]
        oi = ctx.http.get(
            f"{c.base_url}/open-interest", params={"instType": "SWAP", "instId": inst}
        ).json()["data"][0]
        mk = ctx.http.get(
            f"{c.base_url}/mark-price", params={"instType": "SWAP", "instId": inst}
        ).json()["data"][0]
        return oi_row("okx", sym, now, float(oi["oi"]), float(oi["oiUsd"]), float(mk["markPx"]))

    @staticmethod
    def _deribit(ctx: RunCtx, sym: str, now: datetime) -> tuple:
        c = ctx.settings.config.collect.derivs.deribit
        inst = c.inst[sym]
        r = ctx.http.get(c.url, params={"instrument_name": inst.name}).json()["result"][0]
        native, mark = float(r["open_interest"]), float(r["mark_price"])
        return oi_row("deribit", sym, now, native, to_usd(native, inst.oi_unit, mark), mark)

    @staticmethod
    def _kraken(ctx: RunCtx, sym: str, now: datetime) -> tuple:
        c = ctx.settings.config.collect.derivs.kraken
        t = ctx.http.get(c.url.format(symbol=c.inst[sym])).json()["ticker"]
        native, mark = float(t["openInterest"]), float(t["markPrice"])
        return oi_row("kraken", sym, now, native, to_usd(native, "base", mark), mark)

    @staticmethod
    def _hl(ctx: RunCtx, symbols: list[str], now: datetime) -> list[tuple]:
        c = ctx.settings.config.collect.derivs.hyperliquid
        meta, ctxs = ctx.http.post(c.url, json={"type": "metaAndAssetCtxs"}).json()
        idx = {u["name"]: i for i, u in enumerate(meta["universe"])}
        out = []
        for sym in symbols:
            a = ctxs[idx[sym]]
            native, mark = float(a["openInterest"]), float(a["markPx"])
            out.append(oi_row("hyperliquid", sym, now, native, to_usd(native, "base", mark), mark))
        return out
