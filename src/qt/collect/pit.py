"""Point-in-time ``released_at`` rules, one pure function per source (component doc section 3).

``released_at`` is the earliest instant the system could have known a row. Phase 3 features for
decision time T read only rows with ``released_at <= T``, so these functions are the contract.
Backfills and live collects call the same function (one code path per source).

Every function returns a tz-aware UTC datetime. Naive datetimes in raise; nothing here ever
falls back to ``now()``. A row that can't be timed raises :class:`UntimeableRow`, and the
collector rejects and logs it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from qt.common.calendar import TradingCalendar
from qt.common.config import PitCfg

ET = ZoneInfo("America/New_York")


class UntimeableRow(ValueError):
    """The collector can't determine when this row became knowable. Reject it, never guess."""


# ---------------------------------------------------------------- helpers
def _aware(dt: datetime, what: str) -> datetime:
    if not isinstance(dt, datetime):
        raise TypeError(f"{what}: expected datetime, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{what}: naive datetime; pass a timezone-aware one")
    return dt.astimezone(UTC)


def _day(d: date, what: str) -> date:
    if isinstance(d, datetime):  # a datetime is a date subclass; refuse the ambiguity
        raise TypeError(f"{what}: expected a date, got a datetime")
    if not isinstance(d, date):
        raise TypeError(f"{what}: expected date, got {type(d).__name__}")
    return d


def parse_hhmm(s: str) -> time:
    parts = [int(p) for p in s.split(":")]
    return time(*parts)


def at_et(d: date, hhmm: str) -> datetime:
    """``d`` at wall-clock ``hhmm`` in New York, as UTC (DST-correct)."""
    return datetime.combine(_day(d, "at_et"), parse_hhmm(hhmm), tzinfo=ET).astimezone(UTC)


def midnight_utc(d: date) -> datetime:
    return datetime.combine(_day(d, "midnight_utc"), time(0), tzinfo=UTC)


# ---------------------------------------------------------------- prices
def stock_bar(cal: TradingCalendar, session: date) -> datetime:
    """Alpaca stock/ETF daily bar: XNYS session close (early closes handled by the calendar)."""
    return cal.session_close(_day(session, "stock_bar"))


def crypto_bar(bar_start: datetime) -> datetime:
    """Alpaca crypto daily bar starting 00:00 UTC: known at the next 00:00 UTC (D-024)."""
    ts = _aware(bar_start, "crypto_bar")
    if (ts.hour, ts.minute, ts.second, ts.microsecond) != (0, 0, 0, 0):
        raise UntimeableRow(f"crypto daily bar must start at 00:00 UTC, got {ts.isoformat()}")
    return ts + timedelta(days=1)


def yfinance_xnys(cal: TradingCalendar, cfg: PitCfg, session: date) -> datetime:
    """yfinance XNYS-listed series (SPY, sector ETFs, ^VIX, ^VIX3M): close + offset."""
    return stock_bar(cal, session) + timedelta(minutes=cfg.yfinance_xnys_offset_min)


def futures_fx(cfg: PitCfg, trade_date: date) -> datetime:
    """yfinance futures/FX (NQ=F, GC=F, DX-Y.NYB): 17:00 ET on the trade date."""
    return at_et(trade_date, cfg.futures_release_et)


# ---------------------------------------------------------------- macro
def fred_release_time(cfg: PitCfg, series_id: str) -> str:
    table = cfg.fred_release_time_et
    return table.get(series_id, table["default"])


def fred_vintage(
    cfg: PitCfg, series_id: str, vintage: date, cal: TradingCalendar | None = None
) -> datetime:
    """ALFRED vintage (realtime_start): the vintage date at the series' release time ET.

    Some series are stamped with the observation date but posted by FRED the next business
    morning (BAMLH0A0HYM2, VIXCLS; D-029). For those, ``fred_release_lag_sessions`` moves the
    release to the N-th XNYS session after the vintage date.
    """
    d = _day(vintage, "fred_vintage")
    lag = cfg.fred_release_lag_sessions.get(series_id, 0)
    if lag:
        if cal is None:
            raise ValueError(f"{series_id} has a release lag; fred_vintage needs the calendar")
        for _ in range(lag):
            d = cal.next_session(d)
    return at_et(d, fred_release_time(cfg, series_id))


def first_seen(fetched_at: datetime) -> datetime:
    """Fetch-time rule: news, live Fed/Congress items, HY archive, live earnings, OI snapshots.

    The caller passes the instant the fetch happened; this never reads the clock itself.
    """
    return _aware(fetched_at, "first_seen")


# ---------------------------------------------------------------- filings
def edgar_fact(
    cfg: PitCfg, acceptance: datetime | None, filed: date | None
) -> tuple[datetime, bool]:
    """EDGAR fact: the filing's acceptanceDateTime, else ``filed + N days`` at 00:00 ET.

    Returns ``(released_at, approx)``; ``approx`` is True when the fallback was used.
    """
    if acceptance is not None:
        return _aware(acceptance, "edgar_fact.acceptance"), False
    if filed is None:
        raise UntimeableRow("EDGAR fact has neither acceptanceDateTime nor a filed date")
    d = _day(filed, "edgar_fact.filed") + timedelta(days=cfg.edgar_fallback_days)
    return at_et(d, "00:00"), True


# ---------------------------------------------------------------- events
def fomc_schedule(cfg: PitCfg, meeting: date) -> datetime:
    """Scheduled FOMC meeting dates: known from Dec 1 of the preceding year, 00:00 ET."""
    k = cfg.fomc_schedule_known
    return at_et(date(_day(meeting, "fomc_schedule").year - 1, k.month, k.day), "00:00")


def fed_item_backfill(published: datetime | None) -> datetime:
    """Backfilled Fed statement/minutes/speech: the item's official published time."""
    if published is None:
        raise UntimeableRow("backfilled Fed item has no published time")
    return _aware(published, "fed_item_backfill")


def congress_backfill(cfg: PitCfg, action_date: date) -> datetime:
    """Backfilled Congress action: end of the action date in ET."""
    return at_et(_day(action_date, "congress_backfill"), cfg.congress_backfill_et)


def earnings_backfill(cfg: PitCfg, report_date: date) -> tuple[datetime, bool]:
    """Backfilled earnings date: report date - N days at 00:00 ET. Always approximate."""
    d = _day(report_date, "earnings_backfill") - timedelta(days=cfg.earnings_backfill_lead_days)
    return at_et(d, "00:00"), True


# ---------------------------------------------------------------- crypto structure
def funding_settlement(settled_at: datetime) -> datetime:
    """Funding rate: known at its settlement time."""
    return _aware(settled_at, "funding_settlement")


def daily_next_utc(day: date) -> datetime:
    """CoinGecko volume/dominance and DefiLlama supply history: ``day + 1`` at 00:00 UTC."""
    return midnight_utc(_day(day, "daily_next_utc") + timedelta(days=1))


def sosovalue_history(cfg: PitCfg, trade_date: date) -> datetime:
    """Backfilled ETF flow: trade date + 1 day at 09:00 ET (flows publish the next morning)."""
    d = _day(trade_date, "sosovalue_history") + timedelta(days=1)
    return at_et(d, cfg.sosovalue_history_et)


# ---------------------------------------------------------------- membership
def membership_fja(snapshot: date) -> datetime:
    """fja05680 membership snapshot: the effective date at 00:00 ET (conservative)."""
    return at_et(_day(snapshot, "membership_fja"), "00:00")
