"""FRED / ALFRED macro series and the HY OAS archive -> raw_macro (component doc 6.3, D-001, D-023).

**ALFRED (``fred``).** Every vintage of every configured series (``source="alfred"``),
``vintage`` = the row's ``realtime_start``, ``released_at`` from ``pit.fred_vintage``.
Backfill and the daily run are one code path: start from the newest stored vintage (or the
ALFRED origin when nothing is stored) and append what's new.

Two API facts shape the code (live probes 2026-11-25):

- A realtime range may hold at most 2000 vintage dates (HTTP 400 above that; DGS10 has ~5200), so
  the range is split into windows that each start on a real vintage date.
- FRED **clips** ``realtime_start`` to the requested start: a value in force since 2026-09-11,
  requested from 2026-10-01, comes back stamped 2026-10-01. So a returned row is a new vintage
  only if its value differs from the latest value we already hold for that period. Clipped
  continuations are dropped, real revisions kept.

**HY archive (``hy_archive``).** FRED serves only the last ~3 years of ICE BofA series. Each run
snapshots the current values and stores each ``(period, value)`` that differs from our latest
archived value, ``source='fred_archive'``, ``vintage`` = fetch date (ET), ``released_at`` =
fetch time.

Not recorded: an observation FRED later *deletes* (its vintage ends with no successor). Rare,
and ``macro_asof`` keeps showing the last value, which is what was known.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.http import HttpClient, HttpError
from qt.common.calendar import ET
from qt.common.config import FredCfg

log = logging.getLogger("qt.collect.fred")

MACRO_COLS = ["series_id", "period", "value", "vintage", "released_at", "source"]
REALTIME_END = "9999-12-31"


@dataclass(frozen=True)
class Obs:
    period: date
    realtime_start: date
    value: float | None


def _value(v: str) -> float | None:
    return None if v in (".", "") else float(v)


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------- API client
class FredApi:
    def __init__(self, http: HttpClient, cfg: FredCfg, api_key: str) -> None:
        self.http, self.cfg, self._key = http, cfg, api_key

    def _get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        q = {"api_key": self._key, "file_type": "json", **params}
        return self.http.get(f"{self.cfg.base_url}/{endpoint}", params=q).json()

    def vintage_dates(self, series_id: str, start: str) -> list[date]:
        out: list[date] = []
        offset = 0
        while True:
            j = self._get(
                "series/vintagedates", series_id=series_id, realtime_start=start,
                realtime_end=REALTIME_END, limit=10000, offset=offset,
            )  # fmt: skip
            batch = j.get("vintage_dates", [])
            out += [date.fromisoformat(v) for v in batch]
            offset += len(batch)
            if not batch or offset >= int(j.get("count", 0)):
                return out

    def observations(self, series_id: str, **params: Any) -> Iterator[Obs]:
        offset = 0
        while True:
            j = self._get(
                "series/observations", series_id=series_id, limit=self.cfg.page_limit,
                offset=offset, **params,
            )  # fmt: skip
            batch = j.get("observations", [])
            for o in batch:
                yield Obs(
                    date.fromisoformat(o["date"]),
                    date.fromisoformat(o["realtime_start"]),
                    _value(o["value"]),
                )
            offset += len(batch)
            if not batch or offset >= int(j.get("count", 0)):
                return


def windows(vintages: list[date], size: int) -> list[tuple[str, str]]:
    """Split sorted vintage dates into realtime ranges of at most ``size`` vintages each.

    Each range starts on a vintage date and ends the day before the next range starts; the last
    one is open-ended.
    """
    out = []
    for i in range(0, len(vintages), size):
        start = vintages[i]
        nxt = vintages[i + size] if i + size < len(vintages) else None
        end = (nxt - timedelta(days=1)).isoformat() if nxt else REALTIME_END
        out.append((start.isoformat(), end))
    return out


def fetch_windows(api: FredApi, sid: str, vintages: list[date], size: int) -> list[Obs]:
    """All observations over the realtime span of ``vintages``, one request per window."""
    out: list[Obs] = []
    for i, (_, end) in enumerate(windows(vintages, size)):
        out += _fetch_range(api, sid, vintages[i * size : (i + 1) * size], end)
    return out


def _fetch_range(api: FredApi, sid: str, vints: list[date], end: str) -> list[Obs]:
    """Observations for the realtime range ``vints[0] .. end``.

    FRED's own count of vintage dates in a range can exceed what ``vintagedates`` lists (2001 vs
    2000 for DGS10, probe 2026-11-25), so a range that still gets HTTP 400 is split in half.
    """
    try:
        return list(api.observations(sid, realtime_start=vints[0].isoformat(), realtime_end=end))
    except HttpError as e:
        if e.status != 400 or len(vints) < 2:
            raise
        h = len(vints) // 2
        log.info("fred_window_split", extra={"series": sid, "start": vints[0].isoformat()})
        mid_end = (vints[h] - timedelta(days=1)).isoformat()
        return _fetch_range(api, sid, vints[:h], mid_end) + _fetch_range(api, sid, vints[h:], end)


def new_vintages(obs: Iterable[Obs], latest: dict[date, tuple[date, float | None]]) -> list[Obs]:
    """Rows that are genuinely new vintages, given ``latest[period] = (vintage, value)`` held.

    Processes in vintage order and updates ``latest`` as it goes. A row is dropped if we already
    hold that period at the same or a later vintage, or if its value equals the latest held
    value (a clipped continuation, or a no-op).
    """
    out = []
    for o in sorted(obs, key=lambda o: (o.realtime_start, o.period)):
        held = latest.get(o.period)
        if held is not None and (o.realtime_start <= held[0] or o.value == held[1]):
            continue
        latest[o.period] = (o.realtime_start, o.value)
        out.append(o)
    return out


def latest_held(
    conn: psycopg.Connection, series_id: str, source: str
) -> dict[date, tuple[date, float | None]]:
    rows = conn.execute(
        "SELECT DISTINCT ON (period) period, vintage, value FROM raw_macro "
        "WHERE series_id = %s AND source = %s ORDER BY period, vintage DESC",
        (series_id, source),
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _api(ctx: RunCtx) -> FredApi:
    key = ctx.settings.secrets.fred_api_key
    assert key is not None  # run_collector skips the collector when the key is missing
    return FredApi(ctx.http, ctx.settings.config.collect.fred, key.get_secret_value())


def _fail_if_any(failed: dict[str, str], what: str, total: int) -> None:
    if failed:
        raise RuntimeError(f"{len(failed)}/{total} {what} failed: {failed}")


# ---------------------------------------------------------------- collectors
class FredCollector:
    name = "fred"
    required_secrets: tuple[str, ...] = ("fred_api_key",)

    def __init__(self, series: list[str] | None = None) -> None:
        self.series = series  # None = every configured series

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.fred
        api = _api(ctx)
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        state: dict[str, Any] = cursor.setdefault("series", {})
        failed: dict[str, str] = {}
        series = self.series or cfg.series
        for sid in series:
            try:
                n, last = self._series(ctx, api, cfg, sid)
            except Exception as e:  # noqa: BLE001  (one series failing never stops the others)
                ctx.conn.rollback()
                failed[sid] = f"{type(e).__name__}: {e}"
                log.warning("fred_series_failed", extra={"series": sid, "error": failed[sid]})
                continue
            res.rows["raw_macro"] = res.rows.get("raw_macro", 0) + n
            state[sid] = {"last_vintage": last.isoformat() if last else None, "rows_last_run": n}
            ctx.conn.commit()  # each series is durable before the next
        _fail_if_any(failed, "FRED series", len(series))
        return res

    @staticmethod
    def _series(ctx: RunCtx, api: FredApi, cfg: FredCfg, sid: str) -> tuple[int, date | None]:
        latest = latest_held(ctx.conn, sid, "alfred")
        last = max((v for v, _ in latest.values()), default=None)
        start = last.isoformat() if last else cfg.realtime_origin
        # >= last, not > last: FRED sometimes adds rows to an existing vintage date later the same
        # day (e.g. DFF weekend observations). Rows we already hold are dropped by new_vintages.
        vintages = [v for v in api.vintage_dates(sid, start) if last is None or v >= last]
        if not vintages:
            return 0, last
        obs = fetch_windows(api, sid, vintages, cfg.max_vintages_per_request)
        fresh = new_vintages(obs, latest)
        pit_cfg = ctx.settings.config.collect.pit
        rows = [
            (sid, o.period, o.value, o.realtime_start,
             pit.fred_vintage(pit_cfg, sid, o.realtime_start, ctx.cal), "alfred")
            for o in fresh
        ]  # fmt: skip
        stats = insert_rows(ctx.conn, "raw_macro", MACRO_COLS, rows)
        log.info(
            "fred_series_done",
            extra={"series": sid, "vintages": len(vintages), "fetched": len(obs),
                   "inserted": stats.inserted},
        )  # fmt: skip
        return stats.inserted, max(vintages)


class HyArchiveCollector:
    name = "hy_archive"
    required_secrets: tuple[str, ...] = ("fred_api_key",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.fred
        api = _api(ctx)
        res = CollectResult(cursor=dict(ctx.cursor))
        failed: dict[str, str] = {}
        for sid in cfg.archive_series:
            try:
                fetched = pit.first_seen(self.clock())
                current = [o for o in api.observations(sid) if o.value is not None]
                latest = latest_held(ctx.conn, sid, "fred_archive")
                vintage = fetched.astimezone(ET).date()
                rows = [
                    (sid, o.period, o.value, vintage, fetched, "fred_archive")
                    for o in current
                    if o.period not in latest or latest[o.period][1] != o.value
                ]
                res.add("raw_macro", insert_rows(ctx.conn, "raw_macro", MACRO_COLS, rows))
                ctx.conn.commit()
            except Exception as e:  # noqa: BLE001
                ctx.conn.rollback()
                failed[sid] = f"{type(e).__name__}: {e}"
                log.warning("hy_archive_failed", extra={"series": sid, "error": failed[sid]})
        _fail_if_any(failed, "archive series", len(cfg.archive_series))
        return res
