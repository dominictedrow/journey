"""EDGAR XBRL facts -> raw_fundamentals (component doc 6.4, D-031). Weekly, after ``aliases``.

For every live member with a CIK (``aliases`` fills ``assets.cik`` from the SEC ticker map):

1. ``submissions/CIK##########.json`` gives each filing's ``acceptanceDateTime`` (true UTC:
   checked against the filing index pages, 2026-11-26). Older filings sit in extra pages listed
   under ``filings.files``; those never change, so they're cached under ``edgar.cache_dir``.
2. If the newest accession equals last week's, the company is skipped (nothing new filed). The
   cursor keys that check on the concept list as well, so adding a concept refetches every
   company once (otherwise its history would never arrive for unchanged filers).
3. ``companyfacts`` for the configured concepts, every unit. One row per
   ``(fiscal_period, metric, accession)``: ``fiscal_period`` = ``start/end`` for durations, ``end``
   for instants, so a filing's prior-year comparatives don't collide with its current figures.
   ``released_at`` = that filing's acceptance time, else ``filed + 1 day`` 00:00 ET
   (``pit.edgar_fact``; the fallback is counted in the cursor).

Predecessor CIKs (``edgar.predecessor_ciks``, holding-company reorganizations) are fetched like
any filer and stored under the successor's asset, for periods ending before their ``until`` date.

Every request carries ``SEC_EDGAR_USER_AGENT``; the collector refuses to start without one that
contains an ``@`` (SEC fair access).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from qt.collect import pit
from qt.collect.aliases import edgar_ua
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.common.calendar import ET
from qt.common.paths import repo_path

log = logging.getLogger("qt.collect.edgar")

FUND_COLS = [
    "asset_id", "fiscal_period", "metric", "value", "unit", "form", "accession", "filed_at",
    "released_at",
]  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def acceptance_map(pages: list[dict[str, Any]]) -> dict[str, datetime]:
    """accession -> acceptance time, from the ``recent`` block and any extra pages."""
    out: dict[str, datetime] = {}
    for p in pages:
        for acc, t in zip(p.get("accessionNumber", []), p.get("acceptanceDateTime", []),
                          strict=False):  # fmt: skip
            if t:
                out[acc] = _utc(t)
    return out


def fact_rows(
    asset_id: int,
    facts: dict[str, Any],
    concepts: list[str],
    accepted: dict[str, datetime],
    pit_cfg: Any,
) -> tuple[list[tuple], int]:
    """raw_fundamentals rows and how many used the filed-date fallback."""
    rows, approx = [], 0
    for c in concepts:
        tax, name = c.split(":")
        units = ((facts.get("facts") or {}).get(tax) or {}).get(name, {}).get("units") or {}
        for unit, items in units.items():
            for f in items:
                acc, end = f.get("accn"), f.get("end")
                if not acc or not end or f.get("val") is None:
                    continue
                filed = date.fromisoformat(f["filed"]) if f.get("filed") else None
                rel, was_approx = pit.edgar_fact(pit_cfg, accepted.get(acc), filed)
                approx += was_approx
                period = f"{f['start']}/{end}" if f.get("start") else end
                filed_at = (datetime.combine(filed, datetime.min.time(), ET).astimezone(UTC)
                            if filed else rel)  # fmt: skip
                rows.append((asset_id, period, c, float(f["val"]), unit, f.get("form"), acc,
                             filed_at, rel))  # fmt: skip
    return rows, approx


class EdgarFactsCollector:
    name = "edgar_facts"
    required_secrets: tuple[str, ...] = ("sec_edgar_user_agent",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def _get(self, ctx: RunCtx, url: str) -> dict[str, Any]:
        return ctx.http.get(url, headers={"User-Agent": self._ua}).json()

    def _page(self, ctx: RunCtx, name: str) -> dict[str, Any]:
        cfg = ctx.settings.config.collect.edgar
        p = repo_path(cfg.cache_dir) / "submissions" / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        j = self._get(ctx, cfg.submissions_file_url.format(name=name))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(j), encoding="utf-8")
        return j

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        self._ua = edgar_ua(ctx.settings)  # raises before any request if unusable
        cfg = ctx.settings.config.collect.edgar
        cursor = dict(ctx.cursor)
        latest: dict[str, str] = cursor.setdefault("latest", {})
        concepts_sha = hashlib.sha256("|".join(sorted(cfg.concepts)).encode()).hexdigest()[:12]
        res = CollectResult(cursor=cursor)
        companies = ctx.conn.execute(
            "SELECT asset_id, cik FROM assets WHERE cik IS NOT NULL AND security_key IS NOT NULL "
            "AND listed_until IS NULL ORDER BY cik"
        ).fetchall()
        if not companies:
            raise RuntimeError("no CIKs mapped: run aliases first")
        failed: dict[str, str] = {}
        skipped = approx = 0
        # share classes (GOOGL/GOOG, FOXA/FOX, ...) are one filer: fetch once, store per asset
        by_cik: dict[str, list[int]] = {}
        for aid, cik in companies:
            by_cik.setdefault(str(cik).zfill(10), []).append(aid)
        # predecessor filers (holding-company reorganizations): their facts go to the same asset,
        # for periods ending before `until`
        until: dict[tuple[str, int], date] = {}
        if cfg.predecessor_ciks:
            keys = dict(ctx.conn.execute(
                "SELECT security_key, asset_id FROM assets WHERE security_key = ANY(%s)",
                (list(cfg.predecessor_ciks),),
            ).fetchall())  # fmt: skip
            for key, preds in cfg.predecessor_ciks.items():
                if key not in keys:
                    log.warning("edgar_predecessor_unknown_security", extra={"security_key": key})
                    continue
                for pr in preds:
                    c10 = pr.cik.zfill(10)
                    by_cik.setdefault(c10, []).append(keys[key])
                    until[(c10, keys[key])] = pr.until
        for cik10, aids in by_cik.items():
            try:
                sub = self._get(ctx, cfg.submissions_url.format(cik=cik10))
                recent = sub["filings"]["recent"]
                newest = recent["accessionNumber"][0] if recent["accessionNumber"] else None
                if newest is not None and latest.get(cik10) == f"{newest}|{concepts_sha}":
                    skipped += 1
                    continue
                pages = [recent] + [
                    self._page(ctx, f["name"]) for f in sub["filings"].get("files", [])
                ]
                facts = self._get(ctx, cfg.facts_url.format(cik=cik10))
                accepted = acceptance_map(pages)
                for aid in aids:
                    rows, n_approx = fact_rows(aid, facts, cfg.concepts, accepted,
                                               ctx.settings.config.collect.pit)  # fmt: skip
                    cut = until.get((cik10, aid))
                    if cut is not None:  # period end is the part after '/' (or the instant)
                        rows = [r for r in rows if date.fromisoformat(r[1].split("/")[-1]) < cut]
                    approx += n_approx
                    stats = insert_rows(ctx.conn, "raw_fundamentals", FUND_COLS, rows)
                    res.add("raw_fundamentals", stats)
                ctx.conn.commit()
                if newest is not None:
                    latest[cik10] = f"{newest}|{concepts_sha}"
            except Exception as e:  # noqa: BLE001  (one filer failing never stops the rest)
                ctx.conn.rollback()
                failed[cik10] = f"{type(e).__name__}: {e}"
                log.warning("edgar_company_failed", extra={"cik": cik10, "error": failed[cik10]})
        cursor.update(
            companies=len(companies), filers=len(by_cik), unchanged=skipped,
            approx_released=approx, failed=len(failed),
        )  # fmt: skip
        # a few filers without XBRL facts (404) are normal; many failures are not
        if len(failed) > max(5, len(by_cik) // 20):
            raise RuntimeError(f"{len(failed)} EDGAR companies failed: "
                               f"{dict(list(failed.items())[:10])}")  # fmt: skip
        if failed:
            log.warning("edgar_some_failed", extra={"count": len(failed)})
        return res
