"""/wallets + /discovery endpoints against in-memory SQLite (dependency-overridden session)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.database.models import Trade, Wallet
from app.deps import get_db_session
from app.main import app

TRACKED = "0x" + "aa" * 20
CANDIDATE = "0x" + "bb" * 20


@pytest.fixture
async def client(db_engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    maker = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override() -> AsyncIterator:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    async with maker() as session:
        session.add_all(
            [
                Wallet(
                    address=TRACKED,
                    is_tracked=True,
                    total_volume=Decimal("500"),
                    discovery_source="leaderboard:PNL:ALL",
                    discovered_at=datetime(2026, 6, 1, tzinfo=UTC),
                ),
                Wallet(
                    address=CANDIDATE,
                    is_tracked=False,
                    total_volume=Decimal("100"),
                    discovery_source="market_holders",
                    discovered_at=datetime(2026, 6, 2, tzinfo=UTC),
                ),
            ]
        )
        session.add(
            Trade(
                tx_hash="0x" + "cd" * 32,
                log_index=0,
                wallet_address=TRACKED,
                token_id="t1",
                side="BUY",
                price=Decimal("0.5"),
                size=Decimal("100"),
                usdc_amount=Decimal("50"),
                block_number=1,
                timestamp=datetime(2026, 6, 10, tzinfo=UTC),
            )
        )
        await session.commit()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_list_wallets_sorted_and_paged(client: AsyncClient) -> None:
    body = (await client.get("/wallets", params={"sort": "total_volume"})).json()
    assert body["total"] == 2
    assert [w["address"] for w in body["items"]] == [TRACKED, CANDIDATE]
    page = (await client.get("/wallets", params={"limit": 1, "offset": 1})).json()
    assert page["total"] == 2
    assert [w["address"] for w in page["items"]] == [CANDIDATE]


async def test_list_wallets_filters(client: AsyncClient) -> None:
    tracked = (await client.get("/wallets", params={"tracked": "true"})).json()
    assert [w["address"] for w in tracked["items"]] == [TRACKED]
    by_source = (await client.get("/wallets", params={"source": "market_holders"})).json()
    assert [w["address"] for w in by_source["items"]] == [CANDIDATE]
    searched = (await client.get("/wallets", params={"search": "BB" * 6})).json()
    assert [w["address"] for w in searched["items"]] == [CANDIDATE]


async def test_wallet_detail_and_404(client: AsyncClient) -> None:
    body = (await client.get(f"/wallets/{TRACKED.upper()}")).json()
    assert body["address"] == TRACKED
    assert body["metrics"] is None  # not computed yet
    assert len(body["recent_trades"]) == 1
    assert body["recent_trades"][0]["side"] == "BUY"
    assert (await client.get("/wallets/0x" + "ff" * 20)).status_code == 404


async def test_patch_wallet_toggles_tracking(client: AsyncClient) -> None:
    body = (await client.patch(f"/wallets/{CANDIDATE}", json={"is_tracked": True})).json()
    assert body["is_tracked"] is True
    body = (await client.get(f"/wallets/{CANDIDATE}")).json()
    assert body["is_tracked"] is True


async def test_refresh_metrics_computes_row(client: AsyncClient) -> None:
    body = (await client.post(f"/wallets/{TRACKED}/refresh-metrics")).json()
    assert body["metrics"]["sample_size"] == 1
    assert body["metrics"]["realized_pnl"] == "0.000000"
    # The wallet rollups refresh from indexed fills too.
    assert body["total_trades"] == 1
    detail = (await client.get(f"/wallets/{TRACKED}")).json()
    assert detail["metrics"]["sample_size"] == 1


async def test_discovery_status(client: AsyncClient) -> None:
    body = (await client.get("/discovery")).json()
    assert body["total_wallets"] == 2
    assert body["tracked_wallets"] == 1
    assert body["last_discovered_at"].startswith("2026-06-02")
    counts = {row["source"]: row["count"] for row in body["by_source"]}
    assert counts == {"leaderboard:PNL:ALL": 1, "market_holders": 1}
