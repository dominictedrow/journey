"""Module G — the Ollama advisory client. The one property that matters: `narrate` fails open,
returning None on every failure mode (connection refused, timeout, HTTP error, malformed body)
and never raising, so a down/slow model can never affect a trading decision."""

import httpx
import pytest

import app.ai_advisor.client as client_mod
from app.ai_advisor import narrate
from app.config import Settings

SETTINGS = Settings(ollama_url="http://ollama:11434", ollama_model="test-model")


class _FakeResponse:
    def __init__(self, *, json_data=None, raises: Exception | None = None) -> None:
        self._json = json_data
        self._raises = raises

    def raise_for_status(self) -> None:
        if self._raises is not None:
            raise self._raises

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _FakeClient:
    """Stand-in for httpx.AsyncClient: returns pre-programmed responses per endpoint, or raises
    a pre-programmed transport error. `tags` defaults to reporting the model as already cached so
    tests exercise the generate path without a pull unless they say otherwise."""

    def __init__(self, *, generate=None, tags=None, error: Exception | None = None, **_) -> None:
        self._generate = generate
        self._tags = tags or _FakeResponse(json_data={"models": [{"name": SETTINGS.ollama_model}]})
        self._error = error

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, **_):
        if self._error is not None:
            raise self._error
        return self._tags

    async def post(self, url, **_):
        if self._error is not None:
            raise self._error
        return self._generate


@pytest.fixture(autouse=True)
def _reset_model_ready():
    """Each test drives `_ensure_model` from a clean slate."""
    client_mod._model_ready = False
    yield
    client_mod._model_ready = False


def _patch_client(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(client_mod.httpx, "AsyncClient", lambda **kw: _FakeClient(**kwargs, **kw))


async def test_narrate_success(monkeypatch) -> None:
    _patch_client(monkeypatch, generate=_FakeResponse(json_data={"response": "  a summary  "}))
    assert await narrate("prompt", settings=SETTINGS) == "a summary"


async def test_narrate_connection_refused_returns_none(monkeypatch) -> None:
    _patch_client(monkeypatch, error=httpx.ConnectError("connection refused"))
    assert await narrate("prompt", settings=SETTINGS) is None


async def test_narrate_timeout_returns_none(monkeypatch) -> None:
    _patch_client(monkeypatch, error=httpx.TimeoutException("timed out"))
    assert await narrate("prompt", settings=SETTINGS) is None


async def test_narrate_http_error_returns_none(monkeypatch) -> None:
    err = httpx.HTTPStatusError("500", request=None, response=None)
    _patch_client(monkeypatch, generate=_FakeResponse(raises=err))
    assert await narrate("prompt", settings=SETTINGS) is None


async def test_narrate_malformed_body_returns_none(monkeypatch) -> None:
    # .json() raises (not valid JSON) — must be swallowed like any other failure.
    _patch_client(monkeypatch, generate=_FakeResponse(json_data=ValueError("not json")))
    assert await narrate("prompt", settings=SETTINGS) is None


async def test_narrate_empty_response_returns_none(monkeypatch) -> None:
    # A well-formed body with an empty completion is "no narrative", not a value.
    _patch_client(monkeypatch, generate=_FakeResponse(json_data={"response": "   "}))
    assert await narrate("prompt", settings=SETTINGS) is None


async def test_narrate_missing_model_triggers_pull_then_generates(monkeypatch) -> None:
    posts: list[str] = []

    class _PullClient(_FakeClient):
        async def post(self, url, **kw):
            posts.append(url)
            if url == "/api/pull":
                return _FakeResponse(json_data={"status": "success"})
            return _FakeResponse(json_data={"response": "ok"})

    # tags reports the model absent → the client should pull before generating.
    monkeypatch.setattr(
        client_mod.httpx,
        "AsyncClient",
        lambda **kw: _PullClient(tags=_FakeResponse(json_data={"models": []}), **kw),
    )
    assert await narrate("prompt", settings=SETTINGS) == "ok"
    assert "/api/pull" in posts and "/api/generate" in posts
