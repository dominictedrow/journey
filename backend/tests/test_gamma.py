from datetime import UTC, datetime
from decimal import Decimal

from app.ingest.gamma import parse_market

RAW = {
    "conditionId": "0x1234abcd",
    "question": "Will it rain tomorrow?",
    "slug": "will-it-rain-tomorrow",
    "category": "Weather",
    "outcomes": '["Yes", "No"]',
    "clobTokenIds": '["111", "222"]',
    "active": True,
    "closed": False,
    "endDate": "2026-12-31T00:00:00Z",
    "liquidity": "12345.67",
    "volume": 890.12,
    "createdAt": "2026-01-15T10:30:00Z",
    "negRisk": True,
}


def test_parse_market_maps_gamma_fields() -> None:
    row = parse_market(RAW)
    assert row is not None
    assert row["id"] == "0x1234abcd"
    assert row["outcomes"] == ["Yes", "No"]
    assert row["token_ids"] == ["111", "222"]
    assert row["end_date"] == datetime(2026, 12, 31, tzinfo=UTC)
    assert row["liquidity"] == Decimal("12345.67")
    assert row["volume"] == Decimal("890.12")
    assert row["active"] is True
    assert row["closed"] is False
    assert row["neg_risk"] is True


def test_parse_market_without_condition_id_is_skipped() -> None:
    assert parse_market({"question": "orphan"}) is None


def test_parse_market_tolerates_missing_fields() -> None:
    row = parse_market({"conditionId": "0xff"})
    assert row is not None
    assert row["question"] == ""
    assert row["end_date"] is None
    assert row["liquidity"] is None
    assert row["neg_risk"] is False
