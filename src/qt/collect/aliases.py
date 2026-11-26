"""EDGAR ticker map + news alias rebuild (component doc 6.4, 6.6, Q7). Weekly, after membership.

1. ``company_tickers.json`` (SEC) fills ``assets.cik`` for live securities and gives their
   registered names. The request carries ``SEC_EDGAR_USER_AGENT``; without one containing an
   ``@`` the collector refuses to run (SEC fair-access policy).
2. ``aliases`` is rebuilt in one transaction (it's a mutable reference table):

   - ``ticker``: every ticker a member traded under, valid for the dates it was the symbol
     (from ``index_membership``). Case-sensitive and cue-gated: tickers like ALL, NOW, A are words.
   - ``edgar_name``: the SEC-registered name, case-insensitive.
   - ``short``: names with legal suffixes and parentheticals stripped. One-word short names
     ("Target", "Apple") are ambiguous, so they need a case-sensitive match plus a finance cue.
   - ``manual``: ``configs/aliases.yaml`` (Google, Facebook, coins, ...).

   Aliases of a delisted security stop the day after ``listed_until``.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import psycopg

from qt.collect.base import CollectResult, RunCtx
from qt.common.config import AliasesCfg

log = logging.getLogger("qt.collect.aliases")

ALIAS_COLS = (
    "asset_id", "alias", "kind", "case_sensitive", "needs_cue", "valid_from", "valid_to", "source",
)  # fmt: skip


class EdgarUaError(RuntimeError):
    pass


def edgar_ua(settings: Any) -> str:
    v = settings.secrets.sec_edgar_user_agent
    ua = v.get_secret_value().strip() if v is not None else ""
    if "@" not in ua:
        raise EdgarUaError("SEC_EDGAR_USER_AGENT must be set and contain a contact email")
    return ua


def short_name(name: str, cfg: AliasesCfg) -> str:
    s = name
    if cfg.strip_parentheticals:
        s = re.sub(r"\s*\([^)]*\)", "", s)
    suffixes = sorted(cfg.strip_suffixes, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        s = s.strip().rstrip(",&").strip()
        for suf in suffixes:
            if s.lower().endswith(" " + suf.lower()) or s.lower().endswith("," + suf.lower()):
                s = s[: -len(suf)].strip()
                changed = True
                break
    return s.strip().rstrip(",.").strip()


def ticker_eras(conn: psycopg.Connection) -> dict[int, list[tuple[str, date, date | None]]]:
    """asset_id -> [(ticker, first_date, next_ticker_first_date_or_None)], oldest first."""
    rows = conn.execute(
        "SELECT asset_id, ticker, min(snapshot_at)::date FROM index_membership "
        "WHERE asset_id IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 3"
    ).fetchall()
    out: dict[int, list[tuple[str, date, date | None]]] = {}
    for aid, t, first in rows:
        out.setdefault(aid, []).append((t, first, None))
    for aid, eras in out.items():
        out[aid] = [
            (t, f, eras[i + 1][1] if i + 1 < len(eras) else None)
            for i, (t, f, _) in enumerate(eras)
        ]
    return out


def build_aliases(
    assets: list[tuple[int, str, str | None, date | None, bool]],
    eras: dict[int, list[tuple[str, date, date | None]]],
    edgar: dict[str, dict[str, Any]],
    cfg: AliasesCfg,
) -> list[tuple]:
    """``assets``: (asset_id, ticker, name, listed_until, is_member)."""
    rows: dict[tuple, tuple] = {}

    def add(aid, alias, kind, cs, cue, vf, vt, src):  # noqa: ANN001
        alias = alias.strip()
        if alias:
            rows.setdefault((aid, alias, kind, vf), (aid, alias, kind, cs, cue, vf, vt, src))

    by_ticker = {}
    for aid, ticker, name, listed_until, _is_member in assets:
        end = listed_until + timedelta(days=1) if listed_until else None
        by_ticker.setdefault(ticker, []).append(aid)
        era = eras.get(aid) or [(ticker, None, None)]
        for i, (t, vf, vt) in enumerate(era):
            add(aid, t, "ticker", True, True, None if i == 0 else vf, vt or end, "membership")
        names = [n for n in [name, (edgar.get(ticker) or {}).get("title")] if n]
        if listed_until is None and ticker in edgar:
            add(aid, edgar[ticker]["title"], "edgar_name", False, False, None, end, "edgar")
        for n in names:
            s = short_name(n, cfg)
            if len(s) < cfg.min_short_len or s == ticker:
                continue
            one_word = " " not in s
            add(aid, s, "short", one_word, one_word, None, end, "derived")
    for m in cfg.manual:
        for aid in by_ticker.get(m.ticker, []):
            vf = date.fromisoformat(m.valid_from) if m.valid_from else None
            vt = date.fromisoformat(m.valid_to) if m.valid_to else None
            add(aid, m.alias, m.kind, m.case_sensitive, m.needs_cue, vf, vt, "manual")
        if m.ticker not in by_ticker:
            log.warning("manual_alias_unknown_ticker", extra={"ticker": m.ticker})
    return list(rows.values())


class AliasesCollector:
    name = "aliases"
    required_secrets: tuple[str, ...] = ("sec_edgar_user_agent",)

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        ua = edgar_ua(ctx.settings)
        c = ctx.settings.config
        j = ctx.http.get(c.collect.edgar.tickers_url, headers={"User-Agent": ua}).json()
        edgar = {}
        for v in j.values():  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
            t = str(v["ticker"]).upper().replace("-", ".")  # SEC writes BRK-B; Alpaca BRK.B
            edgar.setdefault(t, {"cik": f"{int(v['cik_str']):010d}", "title": v["title"]})
        if len(edgar) < 1000:
            raise RuntimeError(f"EDGAR ticker map has only {len(edgar)} entries")

        conn = ctx.conn
        assets = conn.execute(
            "SELECT asset_id, ticker, name, listed_until, security_key IS NOT NULL FROM assets"
        ).fetchall()
        ciks = 0
        for aid, ticker, _, listed_until, is_member in assets:
            e = edgar.get(ticker)
            if e and is_member and listed_until is None:
                ciks += conn.execute(
                    "UPDATE assets SET cik = %s, updated_at = now() WHERE asset_id = %s "
                    "AND cik IS DISTINCT FROM %s",
                    (e["cik"], aid, e["cik"]),
                ).rowcount
        rows = build_aliases(assets, ticker_eras(conn), edgar, c.aliases)
        conn.execute("DELETE FROM aliases")
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO aliases ({', '.join(ALIAS_COLS)}) "
                f"VALUES ({', '.join(['%s'] * len(ALIAS_COLS))})",
                rows,
            )
        res = CollectResult(cursor=dict(ctx.cursor))
        res.rows["aliases"] = len(rows)
        res.cursor.update({"aliases": len(rows), "cik_updates": ciks, "edgar_tickers": len(edgar)})
        log.info("aliases_rebuilt", extra={"aliases": len(rows), "cik_updates": ciks})
        return res
