"""Section 3 released_at rules, one assertion block per source (pure, no DB)."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from qt.collect import pit
from qt.common.calendar import TradingCalendar


@pytest.fixture(scope="module")
def cal(settings):
    return TradingCalendar(settings.config.calendar)


@pytest.fixture(scope="module")
def cfg(settings):
    return settings.config.collect.pit


def utc(*a):
    return datetime(*a, tzinfo=UTC)


# ---------------------------------------------------------------- prices
def test_stock_bar_is_session_close(cal):
    assert pit.stock_bar(cal, date(2026, 11, 24)) == utc(2026, 11, 24, 21, 0)  # 16:00 EST
    assert pit.stock_bar(cal, date(2026, 7, 15)) == utc(2026, 7, 15, 20, 0)  # 16:00 EDT


def test_stock_bar_early_close(cal):
    # Day after Thanksgiving 2026: 13:00 ET close
    assert pit.stock_bar(cal, date(2026, 11, 27)) == utc(2026, 11, 27, 18, 0)


def test_stock_bar_non_session_raises(cal):
    with pytest.raises(ValueError):
        pit.stock_bar(cal, date(2026, 11, 26))  # Thanksgiving


def test_crypto_bar_next_midnight():
    assert pit.crypto_bar(utc(2026, 11, 24)) == utc(2026, 11, 25)
    with pytest.raises(pit.UntimeableRow):
        pit.crypto_bar(utc(2026, 11, 24, 5))
    with pytest.raises(ValueError):
        pit.crypto_bar(datetime(2026, 11, 24))  # naive


def test_yfinance_xnys_close_plus_15(cal, cfg):
    assert pit.yfinance_xnys(cal, cfg, date(2026, 11, 24)) == utc(2026, 11, 24, 21, 15)
    assert pit.yfinance_xnys(cal, cfg, date(2026, 11, 27)) == utc(2026, 11, 27, 18, 15)


def test_futures_1700_et(cfg):
    assert pit.futures_fx(cfg, date(2026, 11, 24)) == utc(2026, 11, 24, 22, 0)  # EST
    assert pit.futures_fx(cfg, date(2026, 7, 15)) == utc(2026, 7, 15, 21, 0)  # EDT


def test_futures_day_d_invisible_at_day_d_decision(cal, cfg):
    d = date(2026, 11, 24)
    assert pit.futures_fx(cfg, d) > cal.decision_time(d)  # 16:30 ET sees D-1 only
    assert pit.futures_fx(cfg, date(2026, 11, 23)) <= cal.decision_time(d)


# ---------------------------------------------------------------- macro
def test_fred_vintage_release_times(cfg):
    assert pit.fred_vintage(cfg, "CPIAUCSL", date(2026, 11, 13)) == utc(2026, 11, 13, 13, 30)
    assert pit.fred_vintage(cfg, "CPIAUCSL", date(2026, 7, 15)) == utc(2026, 7, 15, 12, 30)
    assert pit.fred_vintage(cfg, "DGS10", date(2026, 11, 13)) == utc(2026, 11, 13, 22, 0)


def test_first_seen_requires_aware():
    t = datetime(2026, 11, 24, 10, tzinfo=timezone(timedelta(hours=-5)))
    assert pit.first_seen(t) == utc(2026, 11, 24, 15)
    assert pit.first_seen(t).tzinfo is UTC
    with pytest.raises(ValueError):
        pit.first_seen(datetime(2026, 11, 24, 10))


# ---------------------------------------------------------------- filings
def test_edgar_acceptance_join(cfg):
    acc = datetime(2026, 10, 30, 16, 5, 12, tzinfo=timezone(timedelta(hours=-4)))
    assert pit.edgar_fact(cfg, acc, date(2026, 10, 30)) == (utc(2026, 10, 30, 20, 5, 12), False)


def test_edgar_fallback(cfg):
    # filed 2026-10-30 (EDT) -> 2026-10-31 00:00 ET = 04:00 UTC, flagged approximate
    assert pit.edgar_fact(cfg, None, date(2026, 10, 30)) == (utc(2026, 10, 31, 4, 0), True)
    with pytest.raises(pit.UntimeableRow):
        pit.edgar_fact(cfg, None, None)


# ---------------------------------------------------------------- events
def test_fomc_schedule_dec1_prior_year(cfg):
    assert pit.fomc_schedule(cfg, date(2027, 3, 17)) == utc(2026, 12, 1, 5, 0)


def test_fed_backfill_needs_published():
    assert pit.fed_item_backfill(utc(2026, 11, 5, 19)) == utc(2026, 11, 5, 19)
    with pytest.raises(pit.UntimeableRow):
        pit.fed_item_backfill(None)


def test_congress_backfill_end_of_day(cfg):
    assert pit.congress_backfill(cfg, date(2026, 11, 24)) == utc(2026, 11, 25, 4, 59, 59)


def test_earnings_backfill_approx(cfg):
    assert pit.earnings_backfill(cfg, date(2026, 11, 24)) == (utc(2026, 11, 17, 5, 0), True)


# ---------------------------------------------------------------- crypto + membership
def test_daily_next_utc_and_funding():
    assert pit.daily_next_utc(date(2026, 11, 24)) == utc(2026, 11, 25)
    assert pit.funding_settlement(utc(2026, 11, 24, 8)) == utc(2026, 11, 24, 8)


def test_sosovalue_history(cfg):
    assert pit.sosovalue_history(cfg, date(2026, 11, 24)) == utc(2026, 11, 25, 14, 0)


def test_membership_fja(cfg):
    assert pit.membership_fja(date(2026, 11, 24)) == utc(2026, 11, 24, 5, 0)


def test_datetime_where_date_expected_rejected(cfg):
    with pytest.raises(TypeError):
        pit.fred_vintage(cfg, "CPIAUCSL", utc(2026, 11, 13))
