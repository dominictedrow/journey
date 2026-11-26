"""Alpaca REST client for collectors: assets (paper trading API) and daily bars (market data API).

Only the hard-coded paper trading URL is ever used (CLAUDE.md hard rule). Probed behaviour
(2026-11-25) the callers rely on:

- Stock bars are keyed by the symbol requested and resolve renames: ``META`` and ``FB`` both return
  Facebook's 2016 bars. A multi-symbol request de-duplicates one security asked for under two
  symbols, so identity checks query **one symbol per request**.
- A reused symbol returns the old company's bars for old dates and the new one's after (``FB``:
  Facebook to 2022-06-08, then a ProShares ETF), so a long gap in bars marks a different security.
- Equity daily bars start at 00:00 ET; crypto daily bars at 00:00 UTC, from 2021-01-01.
- Free SIP data must be older than 15 min: ``end`` is clamped to now - ``recent_guard_min``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from qt.collect.http import HttpClient
from qt.common.calendar import ET
from qt.common.config import ALPACA_PAPER_BASE_URL, AlpacaCfg, Settings


@dataclass(frozen=True)
class Bar:
    t: datetime  # bar start, UTC
    o: float
    h: float
    low: float
    c: float
    v: float
    vw: float | None
    n: int | None

    @property
    def ohlcv(self) -> tuple[float, float, float, float, float]:
        return (self.o, self.h, self.low, self.c, self.v)

    @property
    def session(self) -> date:
        return self.t.astimezone(ET).date()


def _bar(b: dict[str, Any]) -> Bar:
    t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(UTC)
    return Bar(t, b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw"), b.get("n"))


def _now() -> datetime:
    return datetime.now(UTC)


class Alpaca:
    def __init__(
        self,
        http: HttpClient,
        cfg: AlpacaCfg,
        key_id: str,
        secret: str,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        if not key_id.startswith("PK"):
            raise ValueError("Alpaca key must be a paper key (starts with 'PK')")
        self.http, self.cfg, self.clock = http, cfg, clock
        self._h = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    @classmethod
    def from_settings(cls, settings: Settings, http: HttpClient, **kw: Any) -> Alpaca:
        s = settings.secrets
        if s.alpaca_api_key_id is None or s.alpaca_api_secret_key is None:
            raise ValueError("Alpaca keys missing")
        return cls(
            http, settings.config.collect.alpaca, s.alpaca_api_key_id.get_secret_value(),
            s.alpaca_api_secret_key.get_secret_value(), **kw,
        )  # fmt: skip

    def _get(self, url: str, **params: Any) -> Any:
        return self.http.get(url, params=params, headers=self._h).json()

    # ------------------------------------------------------------ trading API (paper)
    def assets(self, asset_class: str, status: str = "active") -> list[dict[str, Any]]:
        return self._get(
            f"{ALPACA_PAPER_BASE_URL}/v2/assets", asset_class=asset_class, status=status
        )

    # ------------------------------------------------------------ market data
    def _end(self, end: date | None) -> str:
        guard = self.clock() - timedelta(minutes=self.cfg.recent_guard_min)
        if end is not None:
            stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), ET)
            guard = min(guard, stop.astimezone(UTC))
        return guard.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _paged(self, url: str, params: dict[str, Any]) -> dict[str, list[Bar]]:
        out: dict[str, list[Bar]] = {}
        token = None
        while True:
            q = dict(params, limit=self.cfg.page_limit)
            if token:
                q["page_token"] = token
            j = self._get(url, **q)
            for sym, bars in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend(_bar(b) for b in bars)
            token = j.get("next_page_token")
            if not token:
                return out

    def stock_bars(
        self, symbol: str, start: date, end: date | None = None, adjustment: str = "raw"
    ) -> list[Bar]:
        """Daily bars for one symbol (one per request: see the module note on de-duplication)."""
        got = self._paged(
            f"{self.cfg.data_url}/v2/stocks/bars",
            {"symbols": symbol, "timeframe": "1Day", "start": start.isoformat(),
             "end": self._end(end), "adjustment": adjustment, "feed": self.cfg.stock_feed},
        )  # fmt: skip
        return got.get(symbol, [])

    def stock_bars_multi(
        self, symbols: list[str], start: date, end: date | None = None, adjustment: str = "raw"
    ) -> dict[str, list[Bar]]:
        """Daily bars for several symbols. Only for distinct securities (daily increments): one
        security asked for under two symbols comes back once."""
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate symbols in a multi-symbol bars request")
        return self._paged(
            f"{self.cfg.data_url}/v2/stocks/bars",
            {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(),
             "end": self._end(end), "adjustment": adjustment, "feed": self.cfg.stock_feed},
        )  # fmt: skip

    def crypto_bars(
        self, symbols: list[str], start: date, end: date | None = None
    ) -> dict[str, list[Bar]]:
        return self._paged(
            f"{self.cfg.data_url}/v1beta3/crypto/{self.cfg.crypto_loc}/bars",
            {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(),
             "end": self._end(end)},
        )  # fmt: skip
