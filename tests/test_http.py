"""Shared HTTP client: retries, Retry-After, fail-fast 4xx, rate limit, no secrets in logs."""

import logging

import httpx
import pytest

from qt.collect.http import HostRateLimiter, HttpClient, HttpError, safe_url


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(settings, rec, sleeps):
    return HttpClient(
        settings.config.collect.http,
        transport=httpx.MockTransport(rec),
        sleep=sleeps.append,
        rng=lambda: 0.0,
    )


def test_retry_then_success_warns(settings, caplog):
    rec = Recorder([httpx.Response(503), httpx.ConnectError("x"), httpx.Response(200, text="ok")])
    sleeps = []
    with caplog.at_level(logging.WARNING, logger="qt.collect.http"):
        r = client(settings, rec, sleeps).get("https://api.example.com/v1?api_key=SECRET123")
    assert r.text == "ok" and len(rec.calls) == 3
    base = settings.config.collect.http.backoff_base_s
    assert [s for s in sleeps if s >= base] == [base, base * 2]  # backoff 2, 4 (jitter 0)
    assert "http_retry_succeeded" in caplog.text
    assert "SECRET123" not in caplog.text


def test_retry_after_honoured(settings):
    rec = Recorder([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    sleeps = []
    client(settings, rec, sleeps).get("https://api.example.com/x")
    assert 7.0 in sleeps


def test_exhaustion_raises_without_secret(settings):
    n = settings.config.collect.http.retries + 1
    rec = Recorder([httpx.Response(500)] * n)
    with pytest.raises(HttpError) as ei:
        client(settings, rec, []).get("https://api.example.com/x?api_key=SECRET123")
    assert ei.value.status == 500 and len(rec.calls) == n
    assert "SECRET123" not in str(ei.value)


def test_4xx_fails_fast(settings):
    rec = Recorder([httpx.Response(404)])
    with pytest.raises(HttpError) as ei:
        client(settings, rec, []).get("https://api.example.com/x")
    assert ei.value.status == 404 and len(rec.calls) == 1


def test_generic_user_agent(settings):
    rec = Recorder([httpx.Response(200)])
    client(settings, rec, []).get("https://api.example.com/x")
    assert rec.calls[0].headers["User-Agent"] == settings.config.collect.http.user_agent


def test_rate_limiter_spaces_same_host():
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    rl = HostRateLimiter({"default": 2, "fast.example": 10}, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        rl.wait("a.example")
    assert sleeps == [0.5, 0.5]
    rl.wait("fast.example")  # other host: no wait
    assert sleeps == [0.5, 0.5]


def test_safe_url_strips_query_and_userinfo():
    assert safe_url("https://u:p@api.x.org/a/b?api_key=1#f") == "api.x.org/a/b"
