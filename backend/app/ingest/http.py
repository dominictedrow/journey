"""Shared outbound-HTTP plumbing for every Polymarket API read (Data-API, Gamma, CLOB,
leaderboard).

All of these endpoints are free and unauthenticated, so *we* are the rate limiter:
every GET goes through a per-upstream `Throttle` (a minimum interval between requests)
plus retry with exponential backoff on 429/5xx/transport errors, honouring `Retry-After`
when the server sends one. Non-retryable 4xx responses are returned to the caller
un-raised so call sites can distinguish expected statuses (e.g. the Data-API's
deep-pagination 400 cap) from real errors before calling `raise_for_status()`.

Throttles are process-local (module-level registry). The Celery worker runs with
--concurrency=2 (docker/worker.Dockerfile), so the true outbound ceiling per upstream is
2x the per-process rate — the intervals below are chosen with that multiplier in mind,
comfortably inside Polymarket's public-API courtesy limits.
"""

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Minimum seconds between requests per upstream, per worker process (x2 processes =>
# effective floor is half this interval's rate). Data-API gets the widest gap because
# unattended history backfills hammer it hardest.
_MIN_INTERVAL = {
    "data_api": 0.35,  # ~3 req/s per process
    "gamma": 0.25,
    "clob": 0.20,
}
_DEFAULT_INTERVAL = 0.30


class Throttle:
    """Async minimum-interval gate: `wait()` returns no sooner than `interval` seconds
    after the previous `wait()` returned (per instance)."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_at = now + self.interval


_throttles: dict[str, Throttle] = {}


def get_throttle(upstream: str) -> Throttle:
    if upstream not in _throttles:
        _throttles[upstream] = Throttle(_MIN_INTERVAL.get(upstream, _DEFAULT_INTERVAL))
    return _throttles[upstream]


def _retry_delay(response: httpx.Response | None, attempt: int, backoff: float) -> float:
    """Server-directed Retry-After when present, else exponential backoff with jitter."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass  # HTTP-date form — fall through to backoff
    return backoff * (2**attempt) + random.uniform(0, backoff)


async def throttled_get(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any] | None = None,
    *,
    upstream: str,
    retries: int | None = None,
) -> httpx.Response:
    """GET through the upstream's throttle, retrying 429/5xx/transport errors.

    Returns the final response *without* raising for status: callers that expect a
    specific 4xx (Data-API pagination cap) inspect it first; everyone else calls
    `raise_for_status()` on the result.
    """
    settings = get_settings()
    retries = settings.http_retries if retries is None else retries
    backoff = float(settings.http_backoff_seconds)
    throttle = get_throttle(upstream)

    last_response: httpx.Response | None = None
    for attempt in range(retries + 1):
        await throttle.wait()
        try:
            response = await client.get(path, params=params)
        except httpx.TransportError as exc:
            if attempt >= retries:
                raise
            delay = _retry_delay(None, attempt, backoff)
            logger.warning(
                "%s GET %s transport error (%s); retry %d/%d in %.1fs",
                upstream,
                path,
                exc,
                attempt + 1,
                retries,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        if response.status_code in RETRYABLE_STATUS and attempt < retries:
            delay = _retry_delay(response, attempt, backoff)
            logger.warning(
                "%s GET %s -> %d; retry %d/%d in %.1fs",
                upstream,
                path,
                response.status_code,
                attempt + 1,
                retries,
                delay,
            )
            await asyncio.sleep(delay)
            last_response = response
            continue
        return response
    assert last_response is not None  # loop always returns or sets last_response first
    return last_response
