from httpx import ASGITransport, AsyncClient

from app import __version__
from app.main import app


async def test_health_returns_ok() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["execution_mode"] == "paper"
    assert body["live_armed"] is False
