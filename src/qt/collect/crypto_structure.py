"""Crypto market structure -> raw_metrics (component doc 6.10, D-021, Q2).

**CoinGecko (``coingecko``, Demo key).** Per pool coin, daily ``volume_usd``, ``market_cap_usd``
and ``price_usd`` from ``/coins/{id}/market_chart`` (the Demo plan serves 365 days, so that's the
backfill), plus a market-wide ``/global`` snapshot (BTC/ETH dominance, total cap and volume).

- Daily points are stamped 00:00 UTC. ``released_at`` = point + 1 day (§3 "day + 1 at 00:00
  UTC"). That's conservative: the 00:00 point is CoinGecko's trailing 24 h, so it describes the
  day before, but it may still be revised minutes after midnight. A 1-day lag is noise for the
  30-day ADV filter.
- The last point of a ``market_chart`` response is "now" (not midnight): dropped. Points whose
  ``released_at`` is in the future are dropped too (append-only: they'd never be corrected).
- ``/global`` has no history on the Demo plan, so dominance exists only from our first run:
  ``ts`` = CoinGecko's ``updated_at``, ``released_at`` = fetch time.

**DefiLlama (``defillama``).** Total USD stablecoin supply (``stablecoin_supply_usd``), full
daily history each run (one request, ~3k rows; stored rows conflict and are skipped). Today's
point is partial and changes during the day, so it's only stored once released (day + 1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows

log = logging.getLogger("qt.collect.crypto")

METRIC_COLS = ["source", "metric", "subject", "ts", "value", "unit", "released_at"]
CHART_SERIES = {"total_volumes": "volume_usd", "market_caps": "market_cap_usd",
                "prices": "price_usd"}  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def _ms(v: int | float) -> datetime:
    return datetime.fromtimestamp(v / 1000, UTC)


def daily_rows(
    source: str,
    metric: str,
    subject: str,
    points: list[tuple[datetime, float | None]],
    unit: str,
    now: datetime,
) -> tuple[list[tuple], int]:
    """raw_metrics rows for 00:00 UTC daily points; returns (rows, dropped-as-untimeable).

    Non-midnight points are "now" values (dropped, not counted as rejects); points whose
    release is after ``now`` are dropped until a later run.
    """
    out, bad = [], 0
    for ts, v in points:
        if (ts.hour, ts.minute, ts.second, ts.microsecond) != (0, 0, 0, 0):
            continue
        rel = pit.daily_next_utc(ts.date())
        if rel > now:
            continue
        if v is None:
            bad += 1
            continue
        out.append((source, metric, subject, ts, float(v), unit, rel))
    return out, bad


def stored_last(ctx: RunCtx, source: str, metric: str) -> dict[str, datetime]:
    rows = ctx.conn.execute(
        "SELECT subject, max(ts) FROM raw_metrics WHERE source = %s AND metric = %s "
        "GROUP BY subject",
        (source, metric),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def global_rows(j: dict[str, Any], fetched: datetime) -> list[tuple]:
    d = j["data"]
    ts = datetime.fromtimestamp(int(d["updated_at"]), UTC)
    rel = pit.first_seen(fetched)
    pct = d.get("market_cap_percentage") or {}
    vals = {
        "btc_dominance_pct": (pct.get("btc"), "pct"),
        "eth_dominance_pct": (pct.get("eth"), "pct"),
        "total_market_cap_usd": ((d.get("total_market_cap") or {}).get("usd"), "usd"),
        "total_volume_usd": ((d.get("total_volume") or {}).get("usd"), "usd"),
    }
    return [("coingecko", m, "", ts, float(v), u, rel) for m, (v, u) in vals.items()
            if v is not None]  # fmt: skip


class CoingeckoCollector:
    name = "coingecko"
    required_secrets: tuple[str, ...] = ("coingecko_demo_api_key",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.coingecko
        key = ctx.settings.secrets.coingecko_demo_api_key
        assert key is not None
        hdr = {"x-cg-demo-api-key": key.get_secret_value()}
        res = CollectResult(cursor=dict(ctx.cursor))
        now = self.clock()

        pool = [r[0].split("/")[0] for r in ctx.conn.execute(
            "SELECT ticker FROM assets WHERE asset_class = 'crypto' AND alpaca_tradable "
            "ORDER BY ticker").fetchall()]  # fmt: skip
        unmapped = [s for s in pool if s not in cfg.ids]
        if unmapped:
            log.warning("coingecko_unmapped", extra={"symbols": unmapped})
        last = stored_last(ctx, "coingecko", "volume_usd")
        failed: dict[str, str] = {}

        j = ctx.http.get(f"{cfg.base_url}/global", headers=hdr).json()
        res.add("raw_metrics", insert_rows(ctx.conn, "raw_metrics", METRIC_COLS,
                                           global_rows(j, now)))  # fmt: skip
        ctx.conn.commit()

        for sym in pool:
            if sym not in cfg.ids:
                continue
            have = last.get(sym)
            days = cfg.history_days
            if have is not None:
                days = min(cfg.history_days, max(cfg.refetch_days, (now - have).days + 2))
            try:
                chart = ctx.http.get(
                    f"{cfg.base_url}/coins/{cfg.ids[sym]}/market_chart",
                    params={"vs_currency": "usd", "days": days, "interval": "daily"},
                    headers=hdr,
                ).json()
                for series, metric in CHART_SERIES.items():
                    pts = [(_ms(t), v) for t, v in chart.get(series) or []]
                    rows, bad = daily_rows("coingecko", metric, sym, pts, "usd", now)
                    res.add("raw_metrics", insert_rows(ctx.conn, "raw_metrics", METRIC_COLS, rows))
                    res.rejected += bad
                ctx.conn.commit()
            except Exception as e:  # noqa: BLE001  (one coin failing never stops the rest)
                ctx.conn.rollback()
                failed[sym] = f"{type(e).__name__}: {e}"
                log.warning("coingecko_coin_failed", extra={"symbol": sym, "error": failed[sym]})
        res.cursor["coins"] = len(pool) - len(unmapped)
        if failed:
            raise RuntimeError(f"{len(failed)} CoinGecko coins failed: {failed}")
        return res


def stablecoin_points(j: list[dict[str, Any]]) -> list[tuple[datetime, float | None]]:
    out = []
    for p in j:
        ts = datetime.fromtimestamp(int(p["date"]), UTC)
        out.append((ts, (p.get("totalCirculatingUSD") or {}).get("peggedUSD")))
    return out


class DefillamaCollector:
    name = "defillama"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.defillama
        now = self.clock()
        j = ctx.http.get(cfg.stablecoins_url).json()
        if not j:
            raise RuntimeError("DefiLlama returned no stablecoin history")
        rows, bad = daily_rows(
            "defillama", "stablecoin_supply_usd", "", stablecoin_points(j), "usd", now
        )
        res = CollectResult(cursor=dict(ctx.cursor))
        res.add("raw_metrics", insert_rows(ctx.conn, "raw_metrics", METRIC_COLS, rows))
        res.rejected += bad
        res.cursor["last_day"] = max(r[3] for r in rows).date().isoformat() if rows else None
        return res
