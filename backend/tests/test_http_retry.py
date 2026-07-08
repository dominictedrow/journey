"""Shared outbound-HTTP helper: retry on 429/5xx, pass through non-retryable 4xx."""

import httpx
import pytest

from app.ingest.http import Throttle, throttled_get


def _transport(script: list[httpx.Response]) -> tuple[httpx.MockTransport, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return script[min(len(calls) - 1, len(script) - 1)]

    return httpx.MockTransport(handler), calls


async def test_retries_429_then_succeeds() -> None:
    transport, calls = _transport(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
        response = await throttled_get(client, "/thing", upstream="test-retry")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(calls) == 2


async def test_400_is_returned_without_retry() -> None:
    transport, calls = _transport([httpx.Response(400, json={"error": "bad"})])
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
        response = await throttled_get(client, "/thing", upstream="test-retry")
    assert response.status_code == 400
    assert len(calls) == 1  # non-retryable: the caller decides what a 400 means


async def test_exhausted_retries_return_last_response() -> None:
    transport, calls = _transport([httpx.Response(503, headers={"Retry-After": "0"})])
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
        response = await throttled_get(client, "/thing", upstream="test-retry", retries=2)
    assert response.status_code == 503
    assert len(calls) == 3  # initial + 2 retries


async def test_transport_error_exhaustion_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://x"
    ) as client:
        with pytest.raises(httpx.ConnectError):
            await throttled_get(client, "/thing", upstream="test-retry", retries=1)


async def test_throttle_enforces_min_interval() -> None:
    import time

    throttle = Throttle(0.05)
    start = time.monotonic()
    await throttle.wait()
    await throttle.wait()
    assert time.monotonic() - start >= 0.05
