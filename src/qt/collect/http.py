"""Shared HTTP client for every collector (component doc section 5).

- one ``httpx.Client`` with a generic User-Agent (never Dom's details; EDGAR passes its own)
- per-host rate limiting from ``collect.http.per_host_rps``
- retries on transport errors, 429 and 5xx with exponential backoff + jitter, honouring
  ``Retry-After``; other 4xx fail fast
- a success that needed a retry logs a warning; running out of retries raises :class:`HttpError`

URLs are logged as host + path only: some APIs (FRED, api.data.gov) take the key as a query
parameter, and logs must never contain secrets.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from qt.common.config import HttpCfg

log = logging.getLogger("qt.collect.http")

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class HttpError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None) -> None:
        super().__init__(msg)
        self.status = status


def safe_url(url: httpx.URL | str) -> str:
    """``host/path`` with the query string and credentials removed, for logs and errors."""
    u = httpx.URL(str(url))
    return f"{u.host}{u.path}"


class HostRateLimiter:
    """Minimum spacing between requests to the same host. Thread-safe."""

    def __init__(
        self,
        per_host_rps: Mapping[str, float],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rps = dict(per_host_rps)
        self._clock, self._sleep = clock, sleep
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def interval(self, host: str) -> float:
        return 1.0 / self._rps.get(host, self._rps["default"])

    def wait(self, host: str) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(host, now))
            self._next[host] = slot + self.interval(host)
        delay = slot - now
        if delay > 0:
            self._sleep(delay)


def _retry_after(resp: httpx.Response, now: datetime | None = None) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


class HttpClient:
    def __init__(
        self,
        cfg: HttpCfg,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.cfg = cfg
        self._sleep, self._rng = sleep, rng
        self.limiter = HostRateLimiter(cfg.per_host_rps, clock=clock, sleep=sleep)
        self._client = httpx.Client(
            timeout=cfg.timeout_s,
            headers={"User-Agent": cfg.user_agent},
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ requests
    def backoff(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (1-based): base * 2^(attempt-1) + jitter, capped."""
        d = self.cfg.backoff_base_s * (2 ** (attempt - 1)) + self.cfg.jitter_s * self._rng()
        return min(d, self.cfg.backoff_max_s)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        req = self._client.build_request(method, url, **kwargs)
        where = safe_url(req.url)
        attempts = self.cfg.retries + 1
        last: str = ""
        for attempt in range(1, attempts + 1):
            self.limiter.wait(req.url.host)
            delay: float | None = None
            try:
                resp = self._client.send(req)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}"
            else:
                if resp.status_code < 400:
                    if attempt > 1:
                        log.warning(
                            "http_retry_succeeded",
                            extra={"url": where, "attempt": attempt, "after": last},
                        )
                    return resp
                if resp.status_code not in RETRY_STATUSES:
                    raise HttpError(f"{method} {where} -> HTTP {resp.status_code}",
                                    resp.status_code)
                last = f"HTTP {resp.status_code}"
                ra = _retry_after(resp)
                if ra is not None:
                    delay = min(ra, self.cfg.backoff_max_s)
            if attempt == attempts:
                break
            self._sleep(delay if delay is not None else self.backoff(attempt))
        status = int(last.split()[1]) if last.startswith("HTTP ") else None
        raise HttpError(f"{method} {where} failed after {attempts} attempts ({last})", status)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)
