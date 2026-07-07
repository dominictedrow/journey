"""/strategies endpoints against in-memory SQLite: history list, active lookup, detail with
rationale, and the queue-an-optimizer-run trigger (Celery send monkeypatched)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.database.models import Strategy
from app.deps import get_db_session
from app.main import app

NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


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
        session.add_all(
            [
                Strategy(
                    name="Top-10 overall · equal-weighted · proportional sizing",
                    params={"category": "overall", "top_n": 10, "weighting": "equal"},
                    is_active=False,
                    created_at=NOW,
                    rationale={"decision": "kept_active", "summary": "older run"},
                ),
                Strategy(
                    name="Top-5 overall · score-weighted · fixed sizing",
                    params={"category": "overall", "top_n": 5, "weighting": "score"},
                    is_active=True,
                    created_at=NOW,
                    adopted_at=NOW,
                    rationale={"decision": "adopted", "summary": "Adopted Top-5 ..."},
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
    from app.workers.celery_app import celery_app

    sent: list = []

    class _FakeResult:
        id = "opt-task"

    def fake_send_task(name: str, args: list | None = None, **_: object) -> _FakeResult:
        sent.append((name, args))
        return _FakeResult()

    monkeypatch.setattr(celery_app, "send_task", fake_send_task)
    return sent


async def test_list_newest_first(client: AsyncClient) -> None:
    body = (await client.get("/strategies")).json()
    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == [2, 1]
    assert "rationale" not in body["items"][0]  # list view stays light


async def test_active_returns_current(client: AsyncClient) -> None:
    body = (await client.get("/strategies/active")).json()
    assert body["is_active"] is True
    assert body["params"]["top_n"] == 5
    assert body["rationale"]["decision"] == "adopted"


async def test_detail_includes_rationale(client: AsyncClient) -> None:
    body = (await client.get("/strategies/1")).json()
    assert body["id"] == 1
    assert body["rationale"]["summary"] == "older run"


async def test_unknown_strategy_404s(client: AsyncClient) -> None:
    assert (await client.get("/strategies/999")).status_code == 404


async def test_optimize_queues_task(client: AsyncClient, sent_tasks: list) -> None:
    response = await client.post("/strategies/optimize")
    assert response.status_code == 202
    assert response.json()["task_id"] == "opt-task"
    assert sent_tasks == [("strategies.optimize", None)]


async def test_active_404_when_none(client: AsyncClient, maker: async_sessionmaker) -> None:
    async with maker() as session:
        for row in (await session.scalars(select(Strategy))).all():
            row.is_active = False
        await session.commit()
    assert (await client.get("/strategies/active")).status_code == 404
