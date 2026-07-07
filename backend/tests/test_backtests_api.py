"""/backtests endpoints against in-memory SQLite: create queues a run (Celery send
monkeypatched), `execute_backtest` completes it, GET serves status/result."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.backtesting.engine import execute_backtest
from app.database.models import Trade, Wallet
from app.deps import get_db_session
from app.main import app

WALLET = "0x" + "aa" * 20
BASE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def client(maker: async_sessionmaker) -> AsyncIterator[AsyncClient]:
    async def override() -> AsyncIterator:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    async with maker() as session:
        session.add(Wallet(address=WALLET, is_tracked=True))
        session.add_all(
            [
                Trade(
                    tx_hash="0x" + "01" * 32,
                    log_index=0,
                    wallet_address=WALLET,
                    token_id="tokA",
                    side="BUY",
                    price=Decimal("0.40"),
                    size=Decimal("100"),
                    usdc_amount=Decimal("40"),
                    block_number=1,
                    timestamp=BASE,
                ),
                Trade(
                    tx_hash="0x" + "02" * 32,
                    log_index=0,
                    wallet_address=WALLET,
                    token_id="tokA",
                    side="SELL",
                    price=Decimal("0.60"),
                    size=Decimal("100"),
                    usdc_amount=Decimal("60"),
                    block_number=2,
                    timestamp=BASE + timedelta(hours=1),
                ),
            ]
        )
        await session.commit()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sent_tasks(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture Celery dispatches instead of publishing to a (nonexistent) broker."""
    from app.workers.celery_app import celery_app

    sent: list = []

    class _FakeResult:
        id = "test-task"

    def fake_send_task(name: str, args: list | None = None, **_: object) -> _FakeResult:
        sent.append((name, args))
        return _FakeResult()

    monkeypatch.setattr(celery_app, "send_task", fake_send_task)
    return sent


async def test_create_queues_run(client: AsyncClient, sent_tasks: list) -> None:
    body = {"wallets": [WALLET.upper()], "delay_seconds": 0, "slippage_pct": "0"}
    response = await client.post("/backtests", json=body)
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["params"]["wallets"] == [WALLET]  # addresses normalized to lowercase
    assert sent_tasks == [("backtests.run", [created["id"]])]


async def test_create_rejects_empty_wallets(client: AsyncClient) -> None:
    assert (await client.post("/backtests", json={"wallets": []})).status_code == 422


async def test_execute_and_fetch_result(
    client: AsyncClient, maker: async_sessionmaker, sent_tasks: list
) -> None:
    body = {
        "wallets": [WALLET],
        "initial_capital": "1000",
        "sizing_mode": "fixed",
        "sizing_value": "50",
        "delay_seconds": 0,
        "slippage_pct": "0",
        "fee_pct": "0",
    }
    backtest_id = (await client.post("/backtests", json=body)).json()["id"]

    async with maker() as session:
        row = await execute_backtest(session, backtest_id)
        assert row.status == "completed"

    detail = (await client.get(f"/backtests/{backtest_id}")).json()
    assert detail["status"] == "completed"
    # $50 at 0.40 -> 125 shares, sold at 0.60 -> +25 on 1000.
    assert detail["metrics"]["final_equity"] == 1025.0
    assert detail["metrics"]["trades_executed"] == 2
    assert len(detail["equity_curve"]) == 3
    assert detail["trade_log"][1]["realized_pnl"] == 25.0
    assert detail["started_at"] is not None
    assert detail["finished_at"] is not None


async def test_list_orders_newest_first(
    client: AsyncClient, maker: async_sessionmaker, sent_tasks: list
) -> None:
    first = (await client.post("/backtests", json={"wallets": [WALLET]})).json()["id"]
    second = (await client.post("/backtests", json={"wallets": [WALLET]})).json()["id"]
    body = (await client.get("/backtests")).json()
    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == [second, first]
    # The list view stays light: no equity curve / trade log payloads.
    assert "equity_curve" not in body["items"][0]


async def test_get_unknown_backtest_404s(client: AsyncClient) -> None:
    assert (await client.get("/backtests/999")).status_code == 404


async def test_execute_failure_marks_row_failed(
    client: AsyncClient, maker: async_sessionmaker, sent_tasks: list
) -> None:
    created = (await client.post("/backtests", json={"wallets": [WALLET]})).json()
    # Corrupt the stored params so config validation fails inside execute.
    async with maker() as session:
        from app.database.models import Backtest

        stored = await session.get(Backtest, created["id"])
        assert stored is not None
        stored.params = {"wallets": []}
        await session.commit()
    async with maker() as session:
        row = await execute_backtest(session, created["id"])
        assert row.status == "failed"
        assert "error" in (row.metrics or {})
