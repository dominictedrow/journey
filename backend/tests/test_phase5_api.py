"""/risk, /copy, /portfolio endpoints against in-memory SQLite. The paper executor's book
fetch is stubbed to None (offline fallback fill) and Celery dispatch is captured."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.deps import get_db_session
from app.main import app

WALLET = "0x" + "cc" * 20


@pytest.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
def offline_book(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_book(token_id: str) -> None:
        return None

    monkeypatch.setattr("app.copy_trading.executors.paper.get_book", no_book)


@pytest.fixture
async def client(maker: async_sessionmaker) -> AsyncIterator[AsyncClient]:
    async def override() -> AsyncIterator:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def _sim_body(side: str = "BUY", size: str = "100", price: str = "0.40") -> dict:
    return {"source_wallet": WALLET, "token_id": "tok", "side": side, "size": size, "price": price}


async def test_simulate_buy_flows_to_portfolio(client: AsyncClient) -> None:
    resp = await client.post("/copy/simulate", json=_sim_body())
    assert resp.status_code == 201
    order = resp.json()["order"]
    assert order["status"] == "filled"
    assert order["filled_size"] == "100.000000"

    portfolio = (await client.get("/portfolio")).json()
    assert portfolio["open_positions"] == 1
    assert portfolio["positions"][0]["token_id"] == "tok"
    # 10_000 - 100 * (0.40 * 1.005).
    assert portfolio["cash"] == "9959.800000"


async def test_kill_switch_endpoint_halts_simulate(client: AsyncClient) -> None:
    engaged = (
        await client.post("/risk/kill-switch", json={"enabled": True, "reason": "halt"})
    ).json()
    assert engaged["engaged"] is True

    order = (await client.post("/copy/simulate", json=_sim_body())).json()["order"]
    assert order["status"] == "rejected"
    assert order["error"] == "kill switch engaged"

    await client.post("/risk/kill-switch", json={"enabled": False})
    order = (await client.post("/copy/simulate", json=_sim_body())).json()["order"]
    assert order["status"] == "filled"


async def test_risk_status_exposes_limits_and_switch(client: AsyncClient) -> None:
    body = (await client.get("/risk")).json()
    assert body["kill_switch"]["engaged"] is False
    assert body["limits"]["max_open_positions"] == 10
    assert body["execution_mode"] == "paper"
    assert body["live_armed"] is False


async def test_copy_orders_and_status(client: AsyncClient) -> None:
    await client.post("/copy/simulate", json=_sim_body())
    orders = (await client.get("/copy/orders")).json()
    assert orders["total"] == 1
    status = (await client.get("/copy")).json()
    assert status["by_status"] == {"filled": 1}
    assert status["total_orders"] == 1


async def test_risk_event_recorded_on_rejection(client: AsyncClient) -> None:
    await client.post("/risk/kill-switch", json={"enabled": True, "reason": "halt"})
    await client.post("/copy/simulate", json=_sim_body())
    events = (await client.get("/risk/events?type=kill_switch")).json()
    assert events["total"] >= 1


async def test_revalue_and_history(client: AsyncClient) -> None:
    await client.post("/copy/simulate", json=_sim_body())
    snap = await client.post("/portfolio/revalue")
    assert snap.status_code == 201
    assert snap.json()["open_positions"] == 1

    history = (await client.get("/portfolio/history")).json()
    assert len(history["items"]) >= 1


async def test_copy_run_enqueues_task(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers.celery_app import celery_app

    sent: list = []

    class _Result:
        id = "task-1"

    monkeypatch.setattr(celery_app, "send_task", lambda name, **_: (sent.append(name) or _Result()))
    resp = await client.post("/copy/run")
    assert resp.status_code == 202
    assert resp.json()["task_id"] == "task-1"
    assert sent == ["copy.detect_and_copy"]
