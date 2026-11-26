"""XNYS calendar helpers on top of ``exchange_calendars``.

Session-level functions take a ``date``. Instant-level functions take tz-aware datetimes and
return UTC datetimes; naive datetimes raise (CLAUDE.md timestamp rule).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from qt.common.config import CalendarCfg

ET = ZoneInfo("America/New_York")
_DEFAULT = CalendarCfg(
    exchange="XNYS", tz="America/New_York", start="2000-01-01", decision_offset_min=30
)


@lru_cache(maxsize=4)
def _cal(exchange: str, start: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(exchange, start=start)


class TradingCalendar:
    def __init__(self, cfg: CalendarCfg = _DEFAULT) -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.tz)
        self.cal = _cal(cfg.exchange, cfg.start)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _require_aware(dt: datetime) -> None:
        if not isinstance(dt, datetime):
            raise TypeError(f"expected datetime, got {type(dt).__name__}")
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("naive datetime: pass a timezone-aware datetime")

    def _label(self, d: date) -> pd.Timestamp:
        if isinstance(d, datetime):
            self._require_aware(d)
            d = d.astimezone(self.tz).date()
        return pd.Timestamp(d)

    @staticmethod
    def _utc(ts: pd.Timestamp) -> datetime:
        return ts.tz_convert("UTC").to_pydatetime()

    # ------------------------------------------------------------ sessions
    def is_session(self, d: date) -> bool:
        return bool(self.cal.is_session(self._label(d)))

    def sessions(self, start: date, end: date) -> list[date]:
        rng = self.cal.sessions_in_range(self._label(start), self._label(end))
        return [ts.date() for ts in rng]

    def next_session(self, d: date) -> date:
        label = self._label(d)
        if self.cal.is_session(label):
            return self.cal.next_session(label).date()
        return self.cal.date_to_session(label, direction="next").date()

    def previous_session(self, d: date) -> date:
        label = self._label(d)
        if self.cal.is_session(label):
            return self.cal.previous_session(label).date()
        return self.cal.date_to_session(label, direction="previous").date()

    def _session(self, d: date) -> pd.Timestamp:
        label = self._label(d)
        if not self.cal.is_session(label):
            raise ValueError(f"{label.date()} is not an {self.cfg.exchange} session")
        return label

    def session_open(self, d: date) -> datetime:
        return self._utc(self.cal.session_open(self._session(d)))

    def session_close(self, d: date) -> datetime:
        return self._utc(self.cal.session_close(self._session(d)))

    def is_early_close(self, d: date) -> bool:
        label = self._session(d)
        return bool(label in self.cal.early_closes)

    def decision_time(self, d: date) -> datetime:
        """UTC instant of the post-close decision cycle for session ``d`` (handles early closes)."""
        return self.session_close(d) + timedelta(minutes=self.cfg.decision_offset_min)

    # ------------------------------------------------------------ instants
    def is_open_at(self, dt: datetime) -> bool:
        self._require_aware(dt)
        return bool(self.cal.is_open_on_minute(pd.Timestamp(dt).tz_convert("UTC")))

    def to_et(self, dt: datetime) -> datetime:
        self._require_aware(dt)
        return dt.astimezone(self.tz)

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        TradingCalendar._require_aware(dt)
        return dt.astimezone(UTC)
