"""Survivorship gaps: CIKs, names, sectors and bars of former S&P members (doc 03 section 9, D-043).

Run by ``scripts/map_delisted_ciks.py``. Nothing here runs on a schedule except
:func:`apply_overrides`, which the ``aliases`` collector calls every week.

1. **Who was it?** For every former member missing a CIK or a sector, the Wikipedia constituents
   table *as it stood while the security was a member* (the newest page revision before its last
   membership day, stepping back if that revision doesn't list it) gives its name and GICS
   sector. The pinned "Selected changes" table (it left the page in 2026) gives a second name.
2. **CIK (Q9a, spec as written).** Names are normalized and matched against SEC's
   ``cik-lookup-data.txt``. A candidate is accepted only if its ``submissions`` show periodic
   filings (10-K/10-Q...) across the membership span: an exact name match with exactly one such
   filer, else a fuzzy match (rapidfuzz ratio >= threshold) with exactly one. Everything else goes
   to ``reports/cik_review_YYYYMMDD.md`` for Dom. Wikipedia's own CIK column is only a veto: an
   accepted match goes to review when Wikipedia names a different filer whose filings also
   cover the span (a fuzzy SVB Financial -> CVB Financial passed the filing check). A
   Wikipedia CIK is never accepted on its own.
3. **Sector (Q3, D-043).** The historical GICS sector, else the filer's SIC code through
   ``configs/sic_to_gics.yaml``.
4. **Bars (Q9b).** yfinance, then Stooq, under the tickers the security had while a member. Only
   an exact ticker whose bars start and end within ``bar_date_tol_sessions`` of the span is
   accepted (sources ``yfinance_surv`` / ``stooq_surv``); a successor's prices never are.

Results are written to ``configs/cik_overrides.yaml`` and ``configs/sector_overrides.yaml`` (the
source of truth; a rebuilt DB gets them back from the aliases collector) and summarized in
``reports/survivorship_YYYYMMDD.md``.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml

from qt.collect import pit
from qt.collect.base import insert_rows
from qt.collect.http import HttpClient
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import Config, SurvivorshipCfg
from qt.common.paths import CONFIG_DIR, repo_path

log = logging.getLogger("qt.collect.survivorship")

PRICE_COLS = (
    "asset_id",
    "timeframe",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "source",
    "released_at",
)


# ---------------------------------------------------------------- names
def normalize_name(name: str, suffixes: Iterable[str]) -> str:
    """Lowercase, ``&`` -> and, no punctuation / state tags / parentheticals / corporate words."""
    s = str(name).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"/[a-z]{1,4}/?", " ", s)  # EDGAR state tags: /DE/, /NEW/, /PA
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", s)  # "n v" -> "nv", "a o smith" -> "ao smith"
    words = s.split()
    drop = {x.lower().strip(".") for x in suffixes}
    if words and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in drop:
        words.pop()
    return " ".join(words)


def norm_ticker(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(t).upper())


def parse_cik_lookup(text: str, suffixes: Iterable[str]) -> dict[str, set[str]]:
    """``cik-lookup-data.txt`` lines (``NAME:0000123456:``) -> normalized name -> CIKs."""
    suffixes = list(suffixes)
    out: dict[str, set[str]] = {}
    for line in text.splitlines():
        line = line.rstrip()
        if not line.endswith(":"):
            continue
        name, _, cik = line[:-1].rpartition(":")
        if not name or not cik.isdigit():
            continue
        key = normalize_name(name, suffixes)
        if key:
            out.setdefault(key, set()).add(cik.zfill(10))
    return out


# ---------------------------------------------------------------- filers
@dataclass
class Filer:
    cik: str
    name: str
    former_names: list[str]
    sic: str | None
    periodic: list[date]  # filing dates of periodic reports, sorted

    def covers(self, first: date, last: date, cfg: SurvivorshipCfg) -> bool:
        """Periodic filings by span start (+tol) and on/after span end (-tol)."""
        if not self.periodic:
            return False
        early = self.periodic[0] <= first + timedelta(days=cfg.span_start_tol_days)
        late = self.periodic[-1] >= last - timedelta(days=cfg.span_end_tol_days)
        return early and late


def filer_from_pages(
    cik: str, main: dict[str, Any], older: list[dict[str, Any]], cfg: SurvivorshipCfg
) -> Filer:
    forms = set(cfg.periodic_forms)
    dates: list[date] = []
    for page in [main.get("filings", {}).get("recent", {}), *older]:
        for f, d in zip(page.get("form", []), page.get("filingDate", []), strict=False):
            if f in forms and d:
                dates.append(date.fromisoformat(d))
    sic = str(main.get("sic") or "").strip() or None
    former = [str(x.get("name")) for x in main.get("formerNames", []) or [] if x.get("name")]
    return Filer(cik, str(main.get("name") or ""), former, sic, sorted(dates))


# ---------------------------------------------------------------- targets
@dataclass
class Target:
    asset_id: int
    security_key: str
    ticker: str
    first: date  # first / last membership day
    last: date
    listed_until: date | None
    cik: str | None
    sector: str | None
    sector_source: str | None
    name: str | None
    tickers: list[tuple[str, date, date]] = field(default_factory=list)  # (ticker, first, last)


def load_targets(
    conn: psycopg.Connection, cfg: Config | None = None, index_code: str = "SPX"
) -> list[Target]:
    """Equity securities lacking a CIK or a current Wikipedia sector, or filled by this script.

    Securities whose CIK / sector came from a generated (non-manual) override are matched again
    with that value cleared, so every run reports on, and rewrites, the whole set.
    """
    redo_cik = (
        {e.security_key for e in cfg.cik_overrides.entries if e.match != "manual"} if cfg else set()
    )
    keep_sec = {e.security_key for e in cfg.sector_overrides.entries if e.manual} if cfg else set()
    rows = conn.execute(
        """
        SELECT a.asset_id, a.security_key, a.ticker, a.listed_until, a.cik, a.gics_sector,
               a.sector_source, a.name, m.ticker, min(m.snapshot_at), max(m.snapshot_at)
        FROM assets a JOIN index_membership m ON m.asset_id = a.asset_id AND m.index_code = %s
        WHERE a.asset_class = 'equity' AND a.security_key IS NOT NULL
          AND (a.cik IS NULL OR a.sector_source IS DISTINCT FROM 'wikipedia_gics'
               OR a.security_key = ANY(%s))
        GROUP BY a.asset_id, m.ticker ORDER BY a.asset_id, min(m.snapshot_at)
        """,
        (index_code, sorted(redo_cik)),
    ).fetchall()
    out: dict[int, Target] = {}
    for aid, key, tk, lu, cik, sec, src, name, mt, f, la in rows:
        if key in redo_cik:
            cik = None
        if src != "wikipedia_gics" and key not in keep_sec:
            sec = None
        f, la = f.astimezone(ET).date(), la.astimezone(ET).date()
        t = out.get(aid)
        if t is None:
            t = out[aid] = Target(aid, key, tk, f, la, lu, cik, sec, src, name)
        t.tickers.append((mt, f, la))
        t.first, t.last = min(t.first, f), max(t.last, la)
    return list(out.values())


# ---------------------------------------------------------------- wikipedia
@dataclass
class WikiRow:
    symbol: str
    name: str
    sector: str | None
    cik: str | None


@dataclass
class WikiHit:
    name: str
    sector: str | None
    wiki_cik: str | None
    revid: int
    rev_ts: datetime
    ticker: str


def parse_constituents(html: str, min_rows: int) -> dict[str, WikiRow] | None:
    """Constituents table of a page revision -> normalized ticker -> row (None if unusable)."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml", match="Security")
    except ValueError:
        return None
    for t in tables:
        cols = {str(c).split("[")[0].strip(): c for c in t.columns}
        sym = cols.get("Symbol") or cols.get("Ticker symbol")
        if sym is None or "Security" not in cols or len(t) < min_rows:
            continue
        sec, cik = cols.get("GICS Sector"), cols.get("CIK")
        out = {}
        for _, r in t.iterrows():
            s = str(r[sym]).strip()
            c = r[cik] if cik is not None else None
            try:
                c = f"{int(float(c)):010d}" if c is not None and pd.notna(c) else None
            except (TypeError, ValueError):
                c = None
            out[norm_ticker(s)] = WikiRow(
                s,
                str(r[cols["Security"]]).strip(),
                str(r[sec]).strip() if sec is not None and pd.notna(r[sec]) else None,
                c,
            )
        return out
    return None


def parse_changes(html: str) -> pd.DataFrame:
    """The "Selected changes" table: one row per removal (date, ticker, name)."""
    t = pd.read_html(io.StringIO(html), flavor="lxml", attrs={"id": "changes"})[0]
    t.columns = [
        "_".join(c for c in col if not str(c).startswith("Unnamed"))
        if isinstance(col, tuple)
        else str(col)
        for col in t.columns
    ]
    d = pd.to_datetime(t.iloc[:, 0], errors="coerce").dt.date
    tk = next(c for c in t.columns if c.startswith("Removed") and "Ticker" in c)
    nm = next(c for c in t.columns if c.startswith("Removed") and "Security" in c)
    out = pd.DataFrame({"date": d, "ticker": t[tk], "name": t[nm]}).dropna()
    out["key"] = out["ticker"].map(norm_ticker)
    return out


class Wiki:
    """Page revisions, cached on disk (a revision never changes)."""

    def __init__(self, http: HttpClient, cfg: SurvivorshipCfg, now: datetime) -> None:
        self.http, self.cfg, self.now = http, cfg, now
        self.dir = repo_path(cfg.wiki_cache_dir)
        self._tables: dict[int, dict[str, WikiRow] | None] = {}
        self._revs: list[tuple[datetime, int]] | None = None

    def revisions(self, need_until: date) -> list[tuple[datetime, int]]:
        if self._revs is not None:
            return self._revs
        p = self.dir / "revisions.json"
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            if date.fromisoformat(j["fetched_at"][:10]) > need_until:
                self._revs = [(datetime.fromisoformat(ts), rid) for ts, rid in j["revs"]]
                return self._revs
        revs: list[tuple[str, int]] = []
        params: dict[str, Any] = {
            "action": "query",
            "prop": "revisions",
            "titles": self.cfg.wiki_page,
            "rvprop": "ids|timestamp",
            "rvlimit": "max",
            "rvdir": "newer",
            "format": "json",
            "formatversion": "2",
            "rvstart": f"{self.cfg.wiki_revisions_since.isoformat()}T00:00:00Z",
        }
        while True:
            j = self.http.get(self.cfg.wiki_api_url, params=params).json()
            for r in j["query"]["pages"][0].get("revisions", []):
                revs.append((r["timestamp"].replace("Z", "+00:00"), int(r["revid"])))
            if "continue" not in j:
                break
            params.update(j["continue"])
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"fetched_at": self.now.isoformat(), "revs": revs}), encoding="utf-8"
        )
        self._revs = [(datetime.fromisoformat(ts), rid) for ts, rid in revs]
        log.info("wiki_revisions", extra={"revisions": len(revs)})
        return self._revs

    def html(self, revid: int) -> str:
        p = self.dir / f"rev_{revid}.html.gz"
        if p.exists():
            return gzip.decompress(p.read_bytes()).decode("utf-8")
        text = self.http.get(self.cfg.wiki_index_url, params={"oldid": str(revid)}).text
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(text.encode("utf-8")))
        return text

    def table(self, revid: int) -> dict[str, WikiRow] | None:
        if revid not in self._tables:
            self._tables[revid] = parse_constituents(self.html(revid), self.cfg.wiki_min_rows)
        return self._tables[revid]

    def before(self, d: date) -> tuple[datetime, int] | None:
        """The newest revision saved before 00:00 ET of ``d``."""
        cut = datetime.combine(d, datetime.min.time(), ET).astimezone(UTC)
        best = None
        for ts, rid in self.revisions(d):
            if ts < cut:
                best = (ts, rid)
            else:
                break
        return best


def find_in_wiki(t: Target, wiki: Wiki, cfg: SurvivorshipCfg) -> WikiHit | None:
    """The security's row in the newest revision before it left the index that still lists it."""
    for ticker, _, last in sorted(t.tickers, key=lambda x: x[2], reverse=True):
        tried: set[int] = set()
        for lb in cfg.wiki_lookback_days:
            rev = wiki.before(last - timedelta(days=lb))
            if rev is None or rev[1] in tried:
                continue
            tried.add(rev[1])
            table = wiki.table(rev[1])
            row = table.get(norm_ticker(ticker)) if table else None
            if row is not None:
                sector = cfg.sector_aliases.get(row.sector or "", row.sector)
                return WikiHit(row.name, sector, row.cik, rev[1], rev[0], ticker)
    return None


def changes_name(t: Target, changes: pd.DataFrame, cfg: SurvivorshipCfg) -> str | None:
    win = timedelta(days=cfg.wiki_changes_window_days)
    for ticker, _, last in t.tickers:
        m = changes[
            (changes["key"] == norm_ticker(ticker))
            & changes["date"].map(lambda d, la=last: abs(d - la) <= win)
        ]
        if len(m):
            return str(m.iloc[0]["name"]).strip()
    return None


# ---------------------------------------------------------------- matching
@dataclass
class Match:
    status: str  # accepted | review
    cik: str | None = None
    match: str | None = None  # exact | fuzzy
    score: float | None = None
    matched_name: str | None = None
    reason: str = ""
    candidates: list[tuple[str, str, float, bool]] = field(
        default_factory=list
    )  # cik, name, score, ok


def match_cik(
    t: Target,
    names: list[str],
    lookup: dict[str, set[str]],
    keys: list[str],
    get_filer: Callable[[str], Filer | None],
    cfg: SurvivorshipCfg,
) -> Match:
    from rapidfuzz import fuzz, process

    norms = [n for n in dict.fromkeys(normalize_name(x, cfg.name_suffixes) for x in names) if n]
    if not norms:
        return Match("review", reason="no name for this security")
    seen: dict[str, bool] = {}
    cands: list[tuple[str, str, float, bool]] = []

    def check(cik: str, nm: str, score: float) -> bool:
        if cik not in seen:
            f = get_filer(cik)
            seen[cik] = bool(f and f.covers(t.first, t.last, cfg))
            cands.append((cik, nm, score, seen[cik]))
        return seen[cik]

    exact = [(c, n) for n in norms for c in sorted(lookup.get(n, ()))]
    ok = sorted({c for c, n in exact if check(c, n, 100.0)})
    if len(ok) == 1:
        nm = next(n for c, n in exact if c == ok[0])
        return Match("accepted", ok[0], "exact", 100.0, nm, candidates=cands)
    if len(ok) > 1:
        return Match(
            "review", reason=f"exact name: {len(ok)} filers cover the span", candidates=cands
        )

    ranked: list[tuple[str, str, float]] = []
    for n in norms:
        for key, score, _ in process.extract(
            n, keys, scorer=fuzz.ratio, score_cutoff=cfg.fuzzy_threshold, limit=50
        ):
            ranked.extend((c, key, float(score)) for c in sorted(lookup[key]))
    ranked.sort(key=lambda x: -x[2])
    distinct: list[tuple[str, str, float]] = []
    for c, key, score in ranked:
        if c not in {d[0] for d in distinct} and c not in {e[0] for e in exact}:
            distinct.append((c, key, score))
        if len(distinct) >= cfg.fuzzy_candidates:
            break
    ok_f = [(c, k, s) for c, k, s in distinct if check(c, k, s)]
    if len(ok_f) == 1:
        c, k, s = ok_f[0]
        return Match("accepted", c, "fuzzy", s, k, candidates=cands)
    why = ("exact name, no filer covers the span; " if exact else "") + (
        f"fuzzy: {len(ok_f)} filers cover the span" if distinct else "no candidate name"
    )
    return Match("review", reason=why, candidates=cands)


# ---------------------------------------------------------------- bars
@dataclass
class BarTry:
    source: str
    ticker: str
    status: str  # accepted | rejected | empty | blocked | error
    detail: str = ""
    rows: list[tuple] = field(default_factory=list)


def judge_bars(
    df: pd.DataFrame, first: date, end: date, cal: TradingCalendar, tol: int
) -> tuple[bool, str]:
    """Exact-ticker rule: bars start by the span start and end at its end (+-tol sessions)."""
    if df is None or df.empty:
        return False, "no bars"
    d0, d1 = df.index.min(), df.index.max()
    lag_start = len(cal.sessions(first, d0)) - 1 if d0 > first else 0
    gap_end = len(cal.sessions(min(d1, end), max(d1, end))) - 1
    msg = f"bars {d0}..{d1} vs span {first}..{end}"
    return lag_start <= tol and gap_end <= tol, msg


def bar_rows(
    aid: int, df: pd.DataFrame, source: str, cal: TradingCalendar, cfg: Any, first: date, end: date
) -> list[tuple]:
    out = []
    for d, r in df.iterrows():
        if d < first or d > end or not cal.is_session(d) or pd.isna(r.get("Close")):
            continue
        ts = datetime.combine(d, datetime.min.time(), ET).astimezone(UTC)
        rel = pit.yfinance_xnys(cal, cfg, d)
        vals = [
            None if pd.isna(r.get(c)) else float(r.get(c))
            for c in ("Open", "High", "Low", "Close", "Volume")
        ]
        out.append((aid, "1d", ts, *vals, None, None, source, rel))
    return out


def _daily_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = [x.date() for x in idx]
    return df


def try_bars(
    t: Target,
    http: HttpClient,
    cfg: SurvivorshipCfg,
    cal: TradingCalendar,
    pit_cfg: Any,
    yf_fetch: Callable[[str, date], pd.DataFrame],
) -> list[BarTry]:
    """yfinance then Stooq for each ticker the security had; stops at the first accepted series."""
    end = t.listed_until or t.last
    final = max(t.tickers, key=lambda x: x[2])[0]
    tries: list[BarTry] = []
    for ticker, f, la in t.tickers:
        # a renamed ticker's bars end at its rename; the final ticker's at the delisting date
        span_end = max(end, la) if ticker == final else la
        for source in ("yfinance_surv", "stooq_surv"):
            try:
                if source == "yfinance_surv":
                    df = yf_fetch(ticker, f - timedelta(days=10))
                else:
                    r = http.get(cfg.stooq_url.format(symbol=ticker.lower().replace(".", "-")))
                    if not r.text.startswith("Date,"):
                        tries.append(BarTry(source, ticker, "blocked", "no CSV (bot check page)"))
                        continue
                    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
            except Exception as e:  # noqa: BLE001  (a missing symbol is just a miss)
                tries.append(BarTry(source, ticker, "error", f"{type(e).__name__}: {e}"[:200]))
                continue
            if df is None or df.empty:
                tries.append(BarTry(source, ticker, "empty", "no bars"))
                continue
            df = _daily_index(df)
            ok, msg = judge_bars(df, f, span_end, cal, cfg.bar_date_tol_sessions)
            if not ok:
                tries.append(BarTry(source, ticker, "rejected", msg))
                continue
            rows = bar_rows(t.asset_id, df, source, cal, pit_cfg, f, span_end)
            tries.append(BarTry(source, ticker, "accepted", msg, rows))
            break
    return tries


# ---------------------------------------------------------------- overrides files
def _header(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return "\n".join(x for x in lines if x.startswith("#")) + "\n"


def write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    clean = [{k: v for k, v in e.items() if v is not None and v is not False} for e in entries]
    body = yaml.safe_dump({"entries": clean}, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(_header(path) + body, encoding="utf-8")


def merge_entries(
    old: list[Any], new: list[dict[str, Any]], is_manual: Callable[[Any], bool]
) -> list[dict[str, Any]]:
    """Manual entries stay as they are; every other entry is replaced by this run's result."""
    keep = [o.model_dump(by_alias=True) for o in old if is_manual(o)]
    manual = {k["security_key"] for k in keep}
    return keep + [n for n in new if n["security_key"] not in manual]


def apply_overrides(conn: psycopg.Connection, cfg: Config) -> dict[str, int]:
    """Load ``cik_overrides`` / ``sector_overrides`` into ``assets`` (idempotent).

    A CIK fills only an empty ``assets.cik`` (a different existing CIK is logged, not replaced).
    A sector never replaces a current Wikipedia sector (``sector_source = 'wikipedia_gics'``).
    """
    n = {"cik": 0, "cik_conflicts": 0, "sector": 0}
    for e in cfg.cik_overrides.entries:
        row = conn.execute(
            "SELECT cik FROM assets WHERE security_key = %s", (e.security_key,)
        ).fetchone()
        if row is None:
            log.warning("cik_override_unknown_security", extra={"security_key": e.security_key})
        elif row[0] is None:
            n["cik"] += conn.execute(
                "UPDATE assets SET cik = %s, updated_at = now() WHERE security_key = %s",
                (e.cik, e.security_key),
            ).rowcount
        elif row[0] != e.cik:
            n["cik_conflicts"] += 1
            log.warning(
                "cik_override_conflict",
                extra={"security_key": e.security_key, "have": row[0], "override": e.cik},
            )
    for s in cfg.sector_overrides.entries:
        n["sector"] += conn.execute(
            """
            UPDATE assets SET gics_sector = %s, sector_source = %s,
                   name = COALESCE(name, %s), sic = COALESCE(sic, %s), updated_at = now()
            WHERE security_key = %s AND sector_source IS DISTINCT FROM 'wikipedia_gics'
              AND (gics_sector IS DISTINCT FROM %s OR sector_source IS DISTINCT FROM %s
                   OR (name IS NULL AND %s::text IS NOT NULL)
                   OR (sic IS NULL AND %s::text IS NOT NULL))
            """,
            (s.sector, s.source, s.name, s.sic, s.security_key, s.sector, s.source, s.name, s.sic),
        ).rowcount
    return n


# ---------------------------------------------------------------- run
@dataclass
class Outcome:
    target: Target
    hit: WikiHit | None = None
    alt_name: str | None = None
    match: Match | None = None
    filer: Filer | None = None  # of the accepted / existing CIK
    hint: Filer | None = None  # Wikipedia's CIK, when it isn't the accepted one
    sector: str | None = None
    sector_source: str | None = None
    bars: list[BarTry] = field(default_factory=list)


class SecFiles:
    """cik-lookup-data.txt and submissions JSON, cached under ``cache_dir``."""

    def __init__(self, http: HttpClient, ua: str, cfg: SurvivorshipCfg, now: datetime) -> None:
        self.http, self.ua, self.cfg, self.now = http, ua, cfg, now
        self.dir = repo_path(cfg.cache_dir)
        self._filers: dict[str, Filer | None] = {}

    def _get(self, url: str) -> Any:
        return self.http.get(url, headers={"User-Agent": self.ua})

    def lookup_text(self) -> str:
        p = self.dir / "cik-lookup-data.txt"
        fresh = p.exists() and (
            self.now - datetime.fromtimestamp(p.stat().st_mtime, UTC)
            < timedelta(days=self.cfg.cik_lookup_max_age_days)
        )
        if not fresh:
            self.dir.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self._get(self.cfg.cik_lookup_url).content)
        return p.read_bytes().decode("latin-1")

    def _json(self, name: str, url: str) -> dict[str, Any]:
        p = self.dir / "submissions" / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        j = self._get(url).json()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(j), encoding="utf-8")
        return j

    def filer(self, cik: str, since: date | None = None) -> Filer | None:
        cik = cik.zfill(10)
        if cik in self._filers:
            return self._filers[cik]
        try:
            main = self._json(f"CIK{cik}.json", self.cfg.submissions_url.format(cik=cik))
            older = []
            recent = main.get("filings", {}).get("recent", {}).get("filingDate", [])
            oldest = min(recent) if recent else None
            for f in main.get("filings", {}).get("files", []) or []:
                if since is not None and oldest is not None and date.fromisoformat(oldest) <= since:
                    break  # the recent block already reaches back far enough
                older.append(
                    self._json(f["name"], self.cfg.submissions_file_url.format(name=f["name"]))
                )
                oldest = f.get("filingFrom") or oldest
            out: Filer | None = filer_from_pages(cik, main, older, self.cfg)
        except Exception as e:  # noqa: BLE001  (an unknown CIK is just a failed candidate)
            log.warning(
                "submissions_failed", extra={"cik": cik, "error": f"{type(e).__name__}: {e}"}
            )
            out = None
        self._filers[cik] = out
        return out


def run(
    conn: psycopg.Connection,
    cfg: Config,
    http: HttpClient,
    cal: TradingCalendar,
    ua: str,
    now: datetime,
    *,
    yf_fetch: Callable[[str, date], pd.DataFrame] | None = None,
    write_bars: bool = True,
) -> list[Outcome]:
    sc = cfg.features.survivorship
    targets = load_targets(conn, cfg)
    log.info(
        "survivorship_targets",
        extra={
            "targets": len(targets),
            "no_cik": sum(t.cik is None for t in targets),
            "no_sector": sum(t.sector is None for t in targets),
        },
    )
    wiki = Wiki(http, sc, now)
    sec = SecFiles(http, ua, sc, now)
    changes = parse_changes(wiki.html(sc.wiki_changes_revid))
    lookup = parse_cik_lookup(sec.lookup_text(), sc.name_suffixes)
    keys = list(lookup)

    outs: list[Outcome] = []
    for t in targets:
        o = Outcome(t, hit=find_in_wiki(t, wiki, sc), alt_name=changes_name(t, changes, sc))
        if t.cik is None:
            # assets.name may be the symbol's current holder (BBT -> Beacon Financial): last resort
            names = [x for x in (o.hit.name if o.hit else None, o.alt_name) if x] or (
                [t.name] if t.name else []
            )
            o.match = match_cik(t, names, lookup, keys, lambda c, s=t.first: sec.filer(c, s), sc)
            cik = o.match.cik
            hint = o.hit.wiki_cik if o.hit else None
            if hint and hint != cik:
                o.hint = sec.filer(hint, t.first)
                # a veto only, and only from a real rival: a Wikipedia CIK whose own filings cover
                # the span (a predecessor's, like ENDP's or EVHC's, doesn't); it never accepts
                rival = o.hint is not None and o.hint.covers(t.first, t.last, sc)
                if o.match.status == "accepted" and rival:
                    o.match.status = "review"
                    o.match.reason = (
                        f"{o.match.match} match {cik} ({o.match.matched_name}) "
                        f"disagrees with the Wikipedia CIK {hint}"
                    )
                    cik = None
        else:
            cik = t.cik
        if cik:
            o.filer = sec.filer(cik, t.first)
        if t.sector is None:
            if o.hit and o.hit.sector in set(_sectors()):
                o.sector, o.sector_source = o.hit.sector, "wikipedia_gics_hist"
            elif o.filer and cfg.sic_to_gics.sector(o.filer.sic):
                o.sector, o.sector_source = cfg.sic_to_gics.sector(o.filer.sic), "sic_map"
        outs.append(o)

    gaps = set(sc.bar_gaps)
    by_key = {t.security_key: t for t in load_bar_targets(conn, gaps)}
    for o in outs:
        by_key.pop(o.target.security_key, None)
    bar_outs = [o for o in outs if o.target.security_key in gaps]
    bar_outs += [Outcome(t) for t in by_key.values()]
    fetch = yf_fetch or _yf_fetch
    for o in bar_outs:
        o.bars = try_bars(o.target, http, sc, cal, cfg.collect.pit, fetch)
        for b in (b for b in o.bars if b.status == "accepted" and write_bars):
            st = insert_rows(conn, "raw_prices", PRICE_COLS, b.rows)
            log.info(
                "survivorship_bars",
                extra={
                    "security_key": o.target.security_key,
                    "source": b.source,
                    "rows": st.inserted,
                },
            )
    return outs + [o for o in bar_outs if o not in outs]


def _sectors() -> tuple[str, ...]:
    from qt.common.config import GICS_SECTORS

    return GICS_SECTORS


def _yf_fetch(ticker: str, start: date) -> pd.DataFrame:
    from qt.collect.yfin import _yf_fetch as f

    return f(ticker, start)


def load_bar_targets(conn: psycopg.Connection, keys: set[str]) -> list[Target]:
    """Bar-gap securities that already have a CIK and a sector (so load_targets skips them)."""
    if not keys:
        return []
    rows = conn.execute(
        """
        SELECT a.asset_id, a.security_key, a.ticker, a.listed_until, a.cik, a.gics_sector,
               a.sector_source, a.name, m.ticker, min(m.snapshot_at), max(m.snapshot_at)
        FROM assets a JOIN index_membership m ON m.asset_id = a.asset_id
        WHERE a.security_key = ANY(%s)
        GROUP BY a.asset_id, m.ticker ORDER BY a.asset_id, min(m.snapshot_at)
        """,
        (sorted(keys),),
    ).fetchall()
    out: dict[int, Target] = {}
    for aid, key, tk, lu, cik, sec, src, name, mt, f, la in rows:
        f, la = f.astimezone(ET).date(), la.astimezone(ET).date()
        t = out.setdefault(aid, Target(aid, key, tk, f, la, lu, cik, sec, src, name))
        t.tickers.append((mt, f, la))
        t.first, t.last = min(t.first, f), max(t.last, la)
    return list(out.values())


# ---------------------------------------------------------------- outputs
def save_overrides(
    outs: list[Outcome], cfg: Config, config_dir: Path = CONFIG_DIR
) -> tuple[int, int]:
    """Write accepted CIKs and resolved sectors to the two overrides files."""
    ciks = []
    for o in outs:
        m = o.match
        if m and m.status == "accepted":
            ciks.append(
                {
                    "security_key": o.target.security_key,
                    "ticker": o.target.ticker,
                    "cik": m.cik,
                    "match": m.match,
                    "name": o.hit.name if o.hit else o.alt_name,
                    "matched_name": m.matched_name,
                    "score": round(m.score, 1) if m.score else None,
                }
            )
    sectors = []
    for o in outs:
        if o.sector:
            sectors.append(
                {
                    "security_key": o.target.security_key,
                    "ticker": o.target.ticker,
                    "sector": o.sector,
                    "source": o.sector_source,
                    "name": o.hit.name if o.hit else o.alt_name,
                    "sic": o.filer.sic if o.filer else None,
                    "revid": o.hit.revid
                    if o.hit and o.sector_source == "wikipedia_gics_hist"
                    else None,
                }
            )
    ce = merge_entries(cfg.cik_overrides.entries, ciks, lambda e: e.match == "manual")
    se = merge_entries(cfg.sector_overrides.entries, sectors, lambda e: e.manual)
    write_entries(config_dir / "cik_overrides.yaml", ce)
    write_entries(config_dir / "sector_overrides.yaml", se)
    return len(ce), len(se)


def write_reports(
    outs: list[Outcome], cfg: SurvivorshipCfg, out_dir: Path, today: date
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cik_t = [o for o in outs if o.match is not None]
    acc = [o for o in cik_t if o.match.status == "accepted"]
    rev = [o for o in cik_t if o.match.status == "review"]
    exact = sum(o.match.match == "exact" for o in acc)
    share = len(acc) / len(cik_t) if cik_t else 1.0
    sec_t = [o for o in outs if o.target.sector is None]
    by_src: dict[str, int] = {}
    for o in sec_t:
        by_src[o.sector_source or "unresolved"] = by_src.get(o.sector_source or "unresolved", 0) + 1

    lines = [
        f"# Survivorship report {today.isoformat()}",
        "",
        "Component doc 03 section 9, D-043. Written by `scripts/map_delisted_ciks.py`.",
        "",
        "## CIKs (Q9a)",
        "",
        "| | count |",
        "|---|---|",
        f"| securities without a CIK | {len(cik_t)} |",
        f"| mapped, exact name | {exact} |",
        f"| mapped, fuzzy name | {len(acc) - exact} |",
        f"| to review (`cik_review_{today:%Y%m%d}.md`) | {len(rev)} |",
        f"| mapped share | {share:.0%} (target {cfg.target_mapped_share:.0%}: "
        f"{'met' if share >= cfg.target_mapped_share else 'NOT met'}) |",
        "",
        "## Sectors (Q3)",
        "",
        "| source | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(by_src.items())],
        "",
        "## Bars for delisted names (Q9b)",
        "",
        "| security | tried | result |",
        "|---|---|---|",
    ]
    for o in outs:
        if not o.bars:
            continue
        acc_b = [b for b in o.bars if b.status == "accepted"]
        res = (
            f"**accepted** {acc_b[0].source} {acc_b[0].ticker}: {len(acc_b[0].rows)} bars"
            if acc_b
            else "gap (documented)"
        )
        tried = "; ".join(f"{b.source} {b.ticker}: {b.status} ({b.detail})" for b in o.bars)
        lines.append(f"| `{o.target.security_key}` | {tried} | {res} |")
    rep = out_dir / f"survivorship_{today:%Y%m%d}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rl = [
        f"# CIK review {today.isoformat()}",
        "",
        "Former S&P members whose CIK wasn't accepted automatically. To confirm one, add it to "
        "`configs/cik_overrides.yaml` with `match: manual` and rerun the aliases collector.",
        "",
        "Candidates: CIK, matched name, score, whether its periodic filings cover the span. "
        "The Wikipedia CIK is the constituents table's own column (a hint, never accepted alone).",
        "",
    ]
    for o in sorted(rev, key=lambda x: x.target.last):
        t, m = o.target, o.match
        rl.append(f"## {t.ticker} (`{t.security_key}`), member {t.first} → {t.last}")
        rl.append(
            f"- Name: {o.hit.name if o.hit else '-'} "
            f"(Wikipedia rev {o.hit.revid if o.hit else '-'})"
            f"; changes table: {o.alt_name or '-'}"
        )
        rl.append(f"- Reason: {m.reason}")
        for c, nm, s, ok in m.candidates:
            rl.append(f"  - `{c}` {nm} ({s:.0f}) {'covers span' if ok else 'does not cover span'}")
        if o.hint is not None or (o.hit and o.hit.wiki_cik):
            h = o.hint
            cov = h.covers(t.first, t.last, cfg) if h else False
            former = f" (formerly {', '.join(h.former_names[:3])})" if h and h.former_names else ""
            rl.append(
                f"- Wikipedia CIK: `{o.hit.wiki_cik}` = {h.name if h else '?'}"
                f"{former}"
                f", filings {'cover' if cov else 'do not cover'} the span"
            )
        rl.append("")
    review = out_dir / f"cik_review_{today:%Y%m%d}.md"
    review.write_text("\n".join(rl) + "\n", encoding="utf-8")
    return rep, review
