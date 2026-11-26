"""S&P 500 point-in-time membership -> ``index_membership`` + member ``assets`` (6.5, D-025, D-030).

Sources: the fja05680 history (snapshots on each change date, up to its last update), then our
weekly Wikipedia snapshots, which are authoritative from the first one on.

**Securities, not tickers.** fja05680 records a rename (FB -> META) as one ticker leaving and
another joining on the same date, and tickers get reused (``FB`` is now an ETF). Identity is
traced from Alpaca bars, which resolve symbols as of each date:

1. *Renames:* a ticker that leaves on date d is the same security as a ticker joining on d if
   the old ticker's last bar equals the new ticker's bar on that session (OHLCV, exact).
2. *Re-entries:* two membership spells of one ticker are the same security if its bars run
   between them with no gap longer than ``gap_sessions``; otherwise the symbol was reused.
3. *End of trading:* a former member still trades if its bars run on to today without such a
   gap; otherwise ``listed_until`` is its last bar before the gap (delisted, acquired, or quiet
   before reuse). Only securities that still trade are linked to an Alpaca asset.

Each security is one ``assets`` row keyed by ``security_key`` = ``SPX:<first ticker>:<first
date>``, with ``ticker`` = its latest symbol (the one Alpaca serves its full history under).
Membership rows carry the ticker as it was then, plus the mapped ``asset_id``; the table is
append-only, so mapping happens before the insert.
"""

from __future__ import annotations

import bisect
import csv
import io
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import psycopg

from qt.collect import pit
from qt.collect.alpaca import Alpaca, Bar
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import MembershipCfg
from qt.common.paths import repo_path

log = logging.getLogger("qt.collect.membership")

MEMBER_COLS = ["index_code", "snapshot_at", "ticker", "source", "asset_id", "released_at"]
BarsFn = Callable[[str, date, date], list[Bar]]


# ---------------------------------------------------------------- sources
@dataclass(frozen=True)
class Snapshot:
    on: date  # ET date the list is in force from
    tickers: frozenset[str]
    source: str  # 'fja05680' | 'wikipedia'
    at: datetime  # snapshot_at = released_at


def parse_fja(content: bytes | str) -> list[tuple[date, frozenset[str]]]:
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    rows = csv.DictReader(io.StringIO(text))
    out = [
        (date.fromisoformat(r["date"]), frozenset(t.strip() for t in r["tickers"].split(",") if t))
        for r in rows
    ]
    return sorted(out)


def fja_snapshots(
    rows: list[tuple[date, frozenset[str]]], start: date, before: date | None
) -> list[Snapshot]:
    """Snapshots from ``start`` (plus the one in force at ``start``), ending before ``before``
    (our first Wikipedia snapshot: from then on Wikipedia is authoritative)."""
    keep = [r for r in rows if before is None or r[0] < before]
    first = max((i for i, r in enumerate(keep) if r[0] <= start), default=0)
    return [Snapshot(d, t, "fja05680", pit.membership_fja(d)) for d, t in keep[first:]]


def parse_wikipedia(html: str, table_id: str, min_rows: int = 450) -> list[dict[str, Any]]:
    t = pd.read_html(io.StringIO(html), attrs={"id": table_id})[0]
    need = {"Symbol", "Security", "GICS Sector", "CIK"}
    if not need <= set(t.columns):
        raise ValueError(f"Wikipedia table changed: columns {list(t.columns)}")
    out = [
        {
            "symbol": str(r["Symbol"]).strip(),
            "name": str(r["Security"]).strip(),
            "sector": str(r["GICS Sector"]).strip(),
            "cik": f"{int(r['CIK']):010d}" if pd.notna(r["CIK"]) else None,
        }
        for _, r in t.iterrows()
    ]
    if len(out) < min_rows:  # the index has ~503 lines; a short table means a parse problem
        raise ValueError(f"Wikipedia table has only {len(out)} rows")
    return out


# ---------------------------------------------------------------- listings
@dataclass
class Listing:
    """One continuous spell of a ticker in the index."""

    ticker: str
    first: date
    last: date  # last snapshot date it was present
    removed_on: date | None  # date of the first snapshot without it (None: still a member)
    group: int = -1

    @property
    def key(self) -> tuple[str, date]:
        return (self.ticker, self.first)


def listings(snaps: list[Snapshot]) -> list[Listing]:
    out: list[Listing] = []
    open_: dict[str, Listing] = {}
    for s in snaps:
        for t in sorted(s.tickers):
            if t in open_:
                open_[t].last = s.on
            else:
                open_[t] = Listing(t, s.on, s.on, None)
                out.append(open_[t])
        for t in [t for t in open_ if t not in s.tickers]:
            open_.pop(t).removed_on = s.on
    return out


def transitions(snaps: list[Snapshot]) -> list[tuple[date, set[str], set[str]]]:
    return [
        (b.on, set(a.tickers - b.tickers), set(b.tickers - a.tickers))
        for a, b in zip(snaps, snaps[1:], strict=False)
    ]


# ---------------------------------------------------------------- identity
class Sessions:
    """Fast session arithmetic over a precomputed XNYS session list."""

    def __init__(self, cal: TradingCalendar, start: date, end: date) -> None:
        self.s = cal.sessions(start, end)

    def between(self, a: date, b: date) -> int:
        """Sessions strictly between ``a`` and ``b``."""
        return max(0, bisect.bisect_left(self.s, b) - bisect.bisect_right(self.s, a))

    def last_on_or_before(self, d: date) -> date:
        return self.s[bisect.bisect_right(self.s, d) - 1]


def continuous_end(bars: Iterable[Bar], start: date, ss: Sessions, gap: int) -> date | None:
    """Last session of the run of bars that starts at ``start`` with no gap > ``gap`` sessions.
    None if no bar falls within ``gap`` sessions of ``start``.

    Zero-volume bars don't count: after a delisting Alpaca keeps serving flat filler bars with
    volume 0 (CA: 44.44 from 2018-11 until the symbol was reused), which would hide the gap."""
    prev, seen = start, False
    for b in sorted(bars, key=lambda b: b.t):
        d = b.session
        if d < start or not b.v > 0:
            continue
        if ss.between(prev, d) > gap:
            break
        prev, seen = d, True
    return prev if seen else None


@dataclass
class Security:
    key: str
    listings: list[Listing]
    listed_until: date | None = None  # set when it stopped trading
    current: bool = False  # in the latest snapshot

    @property
    def ticker(self) -> str:
        return self.listings[-1].ticker


@dataclass
class Identity:
    bars: BarsFn
    ss: Sessions
    cfg: MembershipCfg
    today: date
    cache: dict[str, Any] = field(default_factory=dict)  # persisted in the collector cursor

    _bars: dict[tuple[str, date, date], list[Bar]] = field(default_factory=dict)

    def _memo(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def _get(self, sym: str, a: date, b: date) -> list[Bar]:
        k = (sym, a, b)
        if k not in self._bars:  # per run only; results that matter are cached in ``cache``
            self._bars[k] = self.bars(sym, a, b)
        return self._bars[k]

    def renamed(self, old: str, new: str, on: date) -> bool:
        def check() -> bool:
            back = on - timedelta(days=self.cfg.rename_lookback_days)
            ob = [b for b in self._get(old, back, on) if b.session <= on]
            if not ob:
                return False
            last = ob[-1]
            nb = self._get(new, last.session, last.session)
            return bool(nb) and nb[0].ohlcv == last.ohlcv

        return bool(self._memo(f"rename|{old}|{new}|{on}", check))

    def same_across_gap(self, ticker: str, end: date, start: date) -> bool:
        def check() -> bool:
            got = continuous_end(self._get(ticker, end, start), end, self.ss, self.cfg.gap_sessions)
            return got is not None and self.ss.between(got, start) <= self.cfg.gap_sessions

        return bool(self._memo(f"cont|{ticker}|{end}|{start}", check))

    def trading_end(self, ticker: str, last_member: date) -> date | None:
        """None if still trading; else its last session before a gap. Only an end is cached:
        "still trading" is re-checked every run."""
        key = f"end|{ticker}|{last_member}"
        if key in self.cache:
            return date.fromisoformat(self.cache[key])
        got = continuous_end(
            self._get(ticker, last_member, self.today), last_member, self.ss, self.cfg.gap_sessions
        )
        end = got or last_member
        recent = self.ss.last_on_or_before(self.today - timedelta(days=1))
        if self.ss.between(end, recent) <= self.cfg.gap_sessions:
            return None
        self.cache[key] = end.isoformat()
        return end


def resolve(snaps: list[Snapshot], ident: Identity) -> list[Security]:
    """Group listings into securities (renames + re-entries), then find where each stopped."""
    ls = listings(snaps)
    parent = list(range(len(ls)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a: int, b: int) -> None:
        parent[find(b)] = find(a)

    by_end: dict[tuple[str, date], int] = {(x.ticker, x.removed_on): i for i, x in enumerate(ls)
                                           if x.removed_on}  # fmt: skip
    by_start = {(x.ticker, x.first): i for i, x in enumerate(ls)}

    for on, removed, added in transitions(snaps):
        if not removed or not added:
            continue
        free = set(added)
        for r in sorted(removed):
            for a in sorted(free):
                if ident.renamed(r, a, on):
                    join(by_end[(r, on)], by_start[(a, on)])
                    log.info("membership_rename", extra={"old": r, "new": a, "on": on.isoformat()})
                    free.discard(a)
                    break

    spells: dict[str, list[int]] = {}
    for i, x in enumerate(ls):
        spells.setdefault(x.ticker, []).append(i)
    for idx in spells.values():
        for i, j in zip(idx, idx[1:], strict=False):
            if ident.same_across_gap(ls[i].ticker, ls[i].last, ls[j].first):
                join(i, j)

    groups: dict[int, list[Listing]] = {}
    for i, x in enumerate(ls):
        groups.setdefault(find(i), []).append(x)
    out = []
    for members in groups.values():
        members.sort(key=lambda x: x.first)
        f = members[0]
        sec = Security(f"SPX:{f.ticker}:{f.first.isoformat()}", members)
        sec.current = members[-1].removed_on is None
        if not sec.current:
            sec.listed_until = ident.trading_end(sec.ticker, members[-1].last)
        out.append(sec)
    return sorted(out, key=lambda s: s.key)


# ---------------------------------------------------------------- persistence
def upsert_member_assets(
    conn: psycopg.Connection,
    secs: list[Security],
    alpaca_active: dict[str, dict[str, Any]],
    wiki: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Insert/update one assets row per security. Returns security_key -> asset_id."""
    ids: dict[str, int] = {}
    claimed: set[str] = set()
    for s in secs:
        a = alpaca_active.get(s.ticker) if s.listed_until is None else None
        if a is not None and a["id"] in claimed:
            a = None
        if a is not None:
            claimed.add(a["id"])
            conn.execute(  # an id can move between rows only via a new mapping; free it first
                "UPDATE assets SET alpaca_id = NULL WHERE alpaca_id = %s "
                "AND security_key IS DISTINCT FROM %s",
                (a["id"], s.key),
            )
        w = wiki.get(s.ticker) if s.current else None
        row = conn.execute(
            """
            INSERT INTO assets (ticker, asset_class, name, cik, alpaca_id, gics_sector,
                                alpaca_tradable, security_key, listed_until, sector_source)
            VALUES (%s, 'equity', %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s::text IS NOT NULL THEN 'wikipedia_gics' END)
            ON CONFLICT (security_key) DO UPDATE SET
                ticker = EXCLUDED.ticker,
                name = COALESCE(EXCLUDED.name, assets.name),
                cik = COALESCE(EXCLUDED.cik, assets.cik),
                alpaca_id = EXCLUDED.alpaca_id,
                gics_sector = COALESCE(EXCLUDED.gics_sector, assets.gics_sector),
                sector_source = COALESCE(EXCLUDED.sector_source, assets.sector_source),
                alpaca_tradable = EXCLUDED.alpaca_tradable,
                listed_until = EXCLUDED.listed_until,
                updated_at = now()
            RETURNING asset_id
            """,
            (
                s.ticker,
                (w or {}).get("name") or (a or {}).get("name"),
                (w or {}).get("cik"),
                a["id"] if a else None,
                (w or {}).get("sector"),
                bool(a and a.get("tradable")),
                s.key,
                s.listed_until,
                (w or {}).get("sector"),
            ),
        ).fetchone()
        ids[s.key] = row[0]
    return ids


def membership_rows(
    snaps: list[Snapshot], secs: list[Security], ids: dict[str, int], index_code: str
) -> list[tuple]:
    owner = {lst.key: s.key for s in secs for lst in s.listings}
    spans = sorted((lst.ticker, lst.first, owner[lst.key]) for s in secs for lst in s.listings)

    def asset(ticker: str, on: date) -> int:
        i = bisect.bisect_right(spans, (ticker, on, "￿")) - 1
        return ids[spans[i][2]]

    return [
        (index_code, s.at, t, s.source, asset(t, s.on), s.at)
        for s in snaps
        for t in sorted(s.tickers)
    ]


def stored_wikipedia(conn: psycopg.Connection, index_code: str) -> list[Snapshot]:
    rows = conn.execute(
        "SELECT snapshot_at, array_agg(ticker) FROM index_membership "
        "WHERE index_code = %s AND source = 'wikipedia' GROUP BY snapshot_at ORDER BY 1",
        (index_code,),
    ).fetchall()
    return [Snapshot(r[0].astimezone(ET).date(), frozenset(r[1]), "wikipedia", r[0]) for r in rows]


def write_crosscheck(path: str, fja_last: Snapshot, wiki: Snapshot) -> None:
    only_fja = sorted(fja_last.tickers - wiki.tickers)
    only_wiki = sorted(wiki.tickers - fja_last.tickers)
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# S&P 500 membership cross-check (component doc 6.5)\n\n"
        f"fja05680 latest snapshot **{fja_last.on}** ({len(fja_last.tickers)} tickers) vs our "
        f"first Wikipedia snapshot **{wiki.at:%Y-%m-%d %H:%M} UTC** ({len(wiki.tickers)}).\n\n"
        "Differences are index changes after fja05680's last update or errors in either list. "
        "They are not hand-patched: Wikipedia is authoritative from its first snapshot.\n\n"
        f"- Only in fja05680 ({len(only_fja)}): {', '.join(only_fja) or 'none'}\n"
        f"- Only in Wikipedia ({len(only_wiki)}): {', '.join(only_wiki) or 'none'}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------- collector
def _now() -> datetime:
    return datetime.now(UTC)


class MembershipCollector:
    name = "membership"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        alpaca: Callable[[RunCtx], Any] | None = None,
    ) -> None:
        self.clock = clock
        self.make_alpaca = alpaca or (lambda ctx: Alpaca.from_settings(ctx.settings, ctx.http))

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        c = ctx.settings.config.collect
        cfg, index = c.membership, c.universe.index
        now = self.clock()
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        alpaca = self.make_alpaca(ctx)

        wiki_rows = parse_wikipedia(
            ctx.http.get(cfg.wikipedia_url).text, cfg.wikipedia_table_id, cfg.wikipedia_min_rows
        )
        wiki_now = Snapshot(
            now.astimezone(ET).date(), frozenset(r["symbol"] for r in wiki_rows), "wikipedia",
            pit.first_seen(now),
        )  # fmt: skip
        prior = stored_wikipedia(ctx.conn, index)
        first_wiki = prior[0] if prior else wiki_now
        fja = fja_snapshots(
            parse_fja(ctx.http.get(cfg.fja_url).content),
            date.fromisoformat(c.universe.membership_start),
            first_wiki.on,
        )
        if not fja:
            raise RuntimeError("fja05680 history is empty")
        snaps = fja + prior + [wiki_now]

        start = snaps[0].on - timedelta(days=30)
        today = now.astimezone(ET).date()
        ident = Identity(
            bars=lambda sym, a, b: alpaca.stock_bars(sym, a, b),
            ss=Sessions(ctx.cal, start, today),
            cfg=cfg,
            today=today,
            cache=cursor.setdefault("identity", {}),
        )
        secs = resolve(snaps, ident)
        for s in secs:  # D-035: a forced end date wins over the one inferred from bars
            o = c.override(s.key)
            if o is not None and o.listed_until is not None:
                s.listed_until = o.listed_until
        active = {a["symbol"]: a for a in alpaca.assets("us_equity")}
        ids = upsert_member_assets(ctx.conn, secs, active, {r["symbol"]: r for r in wiki_rows})
        # An unchanged weekly snapshot adds nothing (as-of reads take the latest snapshot <= T),
        # so it's used for identity resolution above but not stored again (D-031)
        unchanged = bool(prior) and prior[-1].tickers == wiki_now.tickers
        stored = snaps[:-1] if unchanged else snaps
        rows = membership_rows(stored, secs, ids, index)
        res.add("index_membership", insert_rows(ctx.conn, "index_membership", MEMBER_COLS, rows))
        cursor["snapshot_unchanged"] = unchanged
        if not prior:
            write_crosscheck(cfg.crosscheck_report, fja[-1], wiki_now)
        cursor["securities"] = len(secs)
        cursor["delisted"] = sum(1 for s in secs if s.listed_until)
        cursor["renames"] = sum(len({x.ticker for x in s.listings}) - 1 for s in secs)
        log.info(
            "membership_done",
            extra={k: cursor[k] for k in ("securities", "delisted", "renames")},
        )
        return res
