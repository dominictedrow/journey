"""/rankings endpoints against in-memory SQLite: rebuild writes per-category generations,
GET serves only the latest one with score breakdowns attached."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.database.models import Wallet, WalletMetrics
from app.deps import get_db_session
from app.main import app

STRONG = "0x" + "aa" * 20  # politics-dominant, hft, early-info, best risk profile
STEADY = "0x" + "bb" * 20  # crypto-dominant, long_term holder
WEAK = "0x" + "cc" * 20  # no dominant category, worst metrics
NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _metrics(address: str, **values) -> WalletMetrics:
    return WalletMetrics(wallet_address=address, computed_at=NOW, sample_size=50, **values)


@pytest.fixture
async def client(db_engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    maker = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override() -> AsyncIterator:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    async with maker() as session:
        session.add_all([Wallet(address=a) for a in (STRONG, STEADY, WEAK)])
        session.add_all(
            [
                _metrics(
                    STRONG,
                    roi_lifetime=Decimal("0.5"),
                    win_rate=Decimal("0.7"),
                    sharpe=Decimal("2"),
                    roi_30d=Decimal("0.2"),
                    monthly_consistency=Decimal("1"),
                    max_drawdown=Decimal("0.05"),
                    volatility=Decimal("0.1"),
                    losing_streak_max=1,
                    category_mix={"politics": 0.9, "crypto": 0.1},
                    trade_frequency=Decimal("20"),
                    avg_hold_hours=Decimal("2"),
                    early_entry_score=Decimal("0.9"),
                ),
                _metrics(
                    STEADY,
                    roi_lifetime=Decimal("0.1"),
                    win_rate=Decimal("0.5"),
                    sharpe=Decimal("1"),
                    roi_30d=Decimal("0.05"),
                    monthly_consistency=Decimal("0.5"),
                    max_drawdown=Decimal("0.2"),
                    volatility=Decimal("0.3"),
                    losing_streak_max=2,
                    category_mix={"crypto": 0.8, "sports": 0.2},
                    trade_frequency=Decimal("0.5"),
                    avg_hold_hours=Decimal("300"),
                    early_entry_score=Decimal("0.5"),
                ),
                _metrics(
                    WEAK,
                    roi_lifetime=Decimal("-0.2"),
                    win_rate=Decimal("0.2"),
                    sharpe=Decimal("-1"),
                    roi_30d=Decimal("-0.1"),
                    monthly_consistency=Decimal("0"),
                    max_drawdown=Decimal("0.6"),
                    volatility=Decimal("0.8"),
                    losing_streak_max=6,
                    category_mix={"sports": 0.3, "unknown": 0.7},
                    trade_frequency=Decimal("1"),
                    avg_hold_hours=Decimal("24"),
                    early_entry_score=Decimal("0.2"),
                ),
            ]
        )
        await session.commit()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_rankings_empty_before_first_rebuild(client: AsyncClient) -> None:
    body = (await client.get("/rankings")).json()
    assert body == {"computed_at": None, "category": "overall", "categories": [], "items": []}


async def test_rebuild_assigns_categories(client: AsyncClient) -> None:
    body = (await client.post("/rankings/rebuild")).json()
    assert body["wallets_scored"] == 3
    assert body["categories"] == {
        "overall": 3,
        "politics": 1,  # STRONG: 0.9 volume share
        "crypto": 1,  # STEADY: 0.8 volume share
        "sports": 0,  # nobody crosses the 0.5 dominance threshold
        "economics": 0,
        "hft": 1,  # STRONG: 20 fills/day
        "long_term": 1,  # STEADY: 300h average hold
        "early_info": 1,  # STRONG: 0.9 early-entry score
        "low_risk": 1,  # STRONG: best-of-cohort risk component (0.833333)
    }


async def test_rankings_ordered_with_score_breakdown(client: AsyncClient) -> None:
    await client.post("/rankings/rebuild")
    body = (await client.get("/rankings")).json()
    assert body["computed_at"] is not None
    assert [i["rank"] for i in body["items"]] == [1, 2, 3]
    assert [i["wallet"]["address"] for i in body["items"]] == [STRONG, STEADY, WEAK]
    top = body["items"][0]
    detail = top["score_detail"]
    assert detail["composite"] == top["score"]
    # 0.40*0.833333 + 0.30*0.833333 + 0.20*0.5 + 0.10*0.833333 (best of 3 on everything).
    assert float(top["score"]) == pytest.approx(0.766666)
    # No copy orders yet -> neutral prior.
    assert float(detail["copy_quality_component"]) == pytest.approx(0.5)
    components = detail["explanation"]["components"]
    assert components["historical"]["weight"] == 0.4
    assert components["historical"]["inputs"]["roi_lifetime"]["value"] == 0.5
    counts = {c["category"]: c["count"] for c in body["categories"]}
    assert counts["overall"] == 3 and "sports" not in counts


async def test_rankings_filter_by_category(client: AsyncClient) -> None:
    await client.post("/rankings/rebuild")
    politics = (await client.get("/rankings", params={"category": "politics"})).json()
    assert [i["wallet"]["address"] for i in politics["items"]] == [STRONG]
    long_term = (await client.get("/rankings", params={"category": "long_term"})).json()
    assert [i["wallet"]["address"] for i in long_term["items"]] == [STEADY]
    economics = (await client.get("/rankings", params={"category": "economics"})).json()
    assert economics["items"] == []


async def test_rebuild_twice_serves_only_latest_generation(client: AsyncClient) -> None:
    first = (await client.post("/rankings/rebuild")).json()
    second = (await client.post("/rankings/rebuild")).json()
    assert second["computed_at"] > first["computed_at"]
    body = (await client.get("/rankings")).json()
    assert body["computed_at"] is not None
    assert [i["rank"] for i in body["items"]] == [1, 2, 3]  # one generation, not two stacked
    counts = {c["category"]: c["count"] for c in body["categories"]}
    assert counts["overall"] == 3
