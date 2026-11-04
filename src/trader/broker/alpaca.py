"""Minimal Alpaca client, PAPER ONLY (D-032). Standard library HTTP, no SDK.

Guardrails that make the live-money endpoint unreachable from this code:
- the trading base URL is a constant (paper-api), never read from config or env;
- an ``ALPACA_BASE_URL`` / ``APCA_API_BASE_URL`` env var that points anywhere else aborts;
- key IDs must start with ``PK`` (Alpaca paper keys; live keys start with ``AK``);
- the account number must start with ``PA`` (paper accounts) before any order is sent.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
_BASE_URL_VARS = ("ALPACA_BASE_URL", "APCA_API_BASE_URL")


class AlpacaError(RuntimeError):
    pass


class LiveEndpointRefused(AlpacaError):
    """Anything that looks like live trading. Never caught by the runner."""


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    """Tiny .env reader (KEY=VALUE lines, # comments). Real env vars win over the file."""
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    out.update({k: v for k, v in os.environ.items() if k.startswith(("ALPACA_", "APCA_"))})
    return out


def check_paper_only(env: dict[str, str]) -> tuple[str, str]:
    """Return (key_id, secret) or raise. Refuses anything that could reach live trading."""
    for var in _BASE_URL_VARS:
        url = env.get(var, "").rstrip("/")
        if url and url != PAPER_URL:
            raise LiveEndpointRefused(f"{var}={url!r}: only {PAPER_URL} is allowed (paper only)")
    key = env.get("ALPACA_API_KEY_ID") or env.get("APCA_API_KEY_ID") or ""
    secret = env.get("ALPACA_API_SECRET_KEY") or env.get("APCA_API_SECRET_KEY") or ""
    if not key or not secret:
        raise AlpacaError("missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in .env")
    if not key.startswith("PK"):
        raise LiveEndpointRefused("key ID doesn't start with 'PK': not a paper key, refusing")
    return key, secret


class AlpacaPaper:
    def __init__(self, key_id: str, secret: str, timeout: float = 20.0):
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._timeout = timeout

    @classmethod
    def from_env(cls, path: Path = Path(".env")) -> AlpacaPaper:
        return cls(*check_paper_only(load_env(path)))

    def _req(self, method: str, url: str, body: dict | None = None):
        if not (url.startswith(PAPER_URL + "/") or url.startswith(DATA_URL + "/")):
            raise LiveEndpointRefused(f"refusing request to {url}")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise AlpacaError(f"{method} {url} -> HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise AlpacaError(f"{method} {url} -> {e.reason}") from None
        return json.loads(raw) if raw else None

    # trading API (paper)
    def account(self) -> dict:
        return self._req("GET", f"{PAPER_URL}/v2/account")

    def positions(self) -> list[dict]:
        return self._req("GET", f"{PAPER_URL}/v2/positions")

    def clock(self) -> dict:
        return self._req("GET", f"{PAPER_URL}/v2/clock")

    def open_orders(self) -> list[dict]:
        return self._req("GET", f"{PAPER_URL}/v2/orders?status=open&limit=500")

    def submit_order(self, order: dict) -> dict:
        return self._req("POST", f"{PAPER_URL}/v2/orders", order)

    def get_order(self, order_id: str) -> dict:
        return self._req("GET", f"{PAPER_URL}/v2/orders/{order_id}")

    # market data (free IEX feed)
    def snapshots(self, symbols: list[str]) -> dict:
        syms = ",".join(symbols)
        return self._req("GET", f"{DATA_URL}/v2/stocks/snapshots?symbols={syms}&feed=iex")
