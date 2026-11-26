from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from qt.common.calendar import ET, TradingCalendar


@pytest.fixture(scope="module")
def cal(settings):
    return TradingCalendar(settings.config.calendar)


def test_holiday_and_sessions(cal):
    assert not cal.is_session(date(2026, 11, 26))  # Thanksgiving
    assert cal.is_session(date(2026, 11, 25))
    assert not cal.is_session(date(2026, 11, 28))  # Saturday
    assert cal.sessions(date(2026, 11, 23), date(2026, 11, 30)) == [
        date(2026, 11, 23),
        date(2026, 11, 24),
        date(2026, 11, 25),
        date(2026, 11, 27),
        date(2026, 11, 30),
    ]
    assert cal.next_session(date(2026, 11, 25)) == date(2026, 11, 27)
    assert cal.next_session(date(2026, 11, 26)) == date(2026, 11, 27)
    assert cal.previous_session(date(2026, 11, 27)) == date(2026, 11, 25)


def test_early_close(cal):
    d = date(2026, 11, 27)  # day after Thanksgiving: 13:00 ET close
    assert cal.is_early_close(d)
    assert cal.session_close(d).astimezone(ET).hour == 13
    assert cal.decision_time(d) == cal.session_close(d) + timedelta(minutes=30)
    assert not cal.is_early_close(date(2026, 11, 25))


def test_dst_boundary(cal):
    # 2026-03-06 (EST, UTC-5) vs 2026-03-09 (EDT, UTC-4): 09:30 ET both days
    assert cal.session_open(date(2026, 3, 6)) == datetime(2026, 3, 6, 14, 30, tzinfo=UTC)
    assert cal.session_open(date(2026, 3, 9)) == datetime(2026, 3, 9, 13, 30, tzinfo=UTC)
    assert cal.session_close(date(2026, 3, 9)).tzinfo is not None


def test_open_at_and_conversions(cal):
    noon_et = datetime(2026, 11, 25, 12, 0, tzinfo=ET)
    assert cal.is_open_at(noon_et)
    assert not cal.is_open_at(datetime(2026, 11, 26, 12, 0, tzinfo=ET))
    assert cal.to_utc(noon_et) == datetime(2026, 11, 25, 17, 0, tzinfo=UTC)
    assert cal.to_et(datetime(2026, 11, 25, 17, 0, tzinfo=UTC)).hour == 12
    # aware datetime passed as a date uses its ET calendar day
    late_utc = datetime(2026, 11, 26, 2, 0, tzinfo=UTC)  # still 25th in ET
    assert cal.is_session(late_utc)


@pytest.mark.parametrize("fn", ["is_open_at", "to_et", "to_utc"])
def test_naive_rejected(cal, fn):
    with pytest.raises(ValueError, match="naive"):
        getattr(cal, fn)(datetime(2026, 11, 25, 12, 0))


def test_non_session_raises(cal):
    with pytest.raises(ValueError, match="not an XNYS session"):
        cal.session_open(date(2026, 11, 26))


def test_other_offset_aware_ok(cal):
    tokyo = datetime(2026, 11, 26, 2, 0, tzinfo=timezone(timedelta(hours=9)))  # 12:00 ET 25th
    assert cal.is_open_at(tokyo)
