"""Alpaca asset refresh (component doc 6.1, D-021).

- ``alpaca_assets`` (each stock collect): refresh ``alpaca_tradable`` for member securities linked
  to an Alpaca asset, and upsert the configured ETFs. Members themselves are created and linked
  by the weekly ``membership`` collector, which owns their identity.
- ``crypto_assets`` (each crypto collect): the crypto pool = active, tradable, quoted in
  ``crypto.quote`` (USD), minus stablecoins. Coins that drop out keep their row, untradable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import psycopg

from qt.collect.alpaca import Alpaca
from qt.collect.base import CollectResult, RunCtx
from qt.common.config import CryptoCfg

log = logging.getLogger("qt.collect.assets")

_UPSERT = """
    INSERT INTO assets (ticker, asset_class, name, alpaca_id, alpaca_tradable)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (alpaca_id) DO UPDATE SET
        ticker = EXCLUDED.ticker, asset_class = EXCLUDED.asset_class, name = EXCLUDED.name,
        alpaca_tradable = EXCLUDED.alpaca_tradable, updated_at = now()
"""

AlpacaFactory = Callable[[RunCtx], Any]


def _default_alpaca(ctx: RunCtx) -> Alpaca:
    return Alpaca.from_settings(ctx.settings, ctx.http)


def crypto_pool(assets: list[dict[str, Any]], cfg: CryptoCfg) -> list[dict[str, Any]]:
    out = []
    stables = {s.upper() for s in cfg.stablecoin_exclude}
    for a in assets:
        base, _, quote = a["symbol"].partition("/")
        if quote != cfg.quote or base.upper() in stables:
            continue
        if a.get("status") == "active" and a.get("tradable"):
            out.append(a)
    return sorted(out, key=lambda a: a["symbol"])


def _upsert(conn: psycopg.Connection, a: dict[str, Any], asset_class: str) -> None:
    conn.execute(_UPSERT, (a["symbol"], asset_class, a.get("name"), a["id"], bool(a["tradable"])))


class AlpacaAssetsCollector:
    name = "alpaca_assets"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(self, alpaca: AlpacaFactory = _default_alpaca) -> None:
        self.make_alpaca = alpaca

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        c = ctx.settings.config.collect
        active = self.make_alpaca(ctx).assets("us_equity")
        if not active:
            raise RuntimeError("Alpaca returned no active equity assets")
        by_sym = {a["symbol"]: a for a in active}
        by_id = {a["id"]: a for a in active}
        res = CollectResult(cursor=dict(ctx.cursor))
        missing = [t for t in c.universe.extra_etfs if t not in by_sym]
        if missing:
            log.warning("etf_not_on_alpaca", extra={"tickers": missing})
        for t in c.universe.extra_etfs:
            if t in by_sym:
                _upsert(ctx.conn, by_sym[t], "etf")
        members = ctx.conn.execute(
            "SELECT asset_id, alpaca_id::text, alpaca_tradable FROM assets "
            "WHERE security_key IS NOT NULL AND alpaca_id IS NOT NULL"
        ).fetchall()
        changed = 0
        for asset_id, aid, was in members:
            now = bool(by_id.get(aid, {}).get("tradable", False))
            if now != was:
                changed += 1
                ctx.conn.execute(
                    "UPDATE assets SET alpaca_tradable = %s, updated_at = now() "
                    "WHERE asset_id = %s",
                    (now, asset_id),
                )
        res.rows["assets"] = len(c.universe.extra_etfs) - len(missing) + changed
        res.cursor["members_linked"] = len(members)
        res.cursor["tradable_changed"] = changed
        return res


class CryptoAssetsCollector:
    name = "crypto_assets"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(self, alpaca: AlpacaFactory = _default_alpaca) -> None:
        self.make_alpaca = alpaca

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.crypto
        pool = crypto_pool(self.make_alpaca(ctx).assets("crypto"), cfg)
        if not pool:
            raise RuntimeError("crypto pool is empty")
        for a in pool:
            _upsert(ctx.conn, a, "crypto")
        dropped = ctx.conn.execute(
            "UPDATE assets SET alpaca_tradable = false, updated_at = now() "
            "WHERE asset_class = 'crypto' AND alpaca_tradable "
            "AND NOT (alpaca_id = ANY(%s::uuid[]))",
            ([a["id"] for a in pool],),
        ).rowcount
        if dropped:
            log.warning("crypto_pool_dropped", extra={"count": dropped})
        res = CollectResult(cursor=dict(ctx.cursor))
        res.rows["assets"] = len(pool)
        res.cursor["pool"] = [a["symbol"] for a in pool]
        return res
