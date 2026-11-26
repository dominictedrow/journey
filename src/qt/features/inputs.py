"""Point-in-time inputs for the feature layer (component doc 03 section 3).

This is the only module under ``qt.features`` that names raw tables, and it reads them only
through ``qt.db.pit`` (``released_at <= asof``). A run loads each history once with its *last*
decision time; every earlier decision time T is then cut in memory on ``released_at <= T``:
- bars: :func:`load_bars` keeps ``released_at`` per row; :func:`bars_known_at` cuts it.
- macro: :class:`MacroState` walks forward through releases, so the series it hands out at T is
  exactly the set of vintages known at T (later revisions stay invisible until released).
- corporate actions: :func:`load_corp_actions` + :func:`map_actions_to_assets`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg

from qt.common.config import FeatureBarsCfg
from qt.db.pit import frame_asof

log = logging.getLogger("qt.features")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

BAR_COLUMNS = ["asset_id", "ts", "open", "high", "low", "close", "volume", "source", "released_at"]


# ---------------------------------------------------------------- reference data (not raw)
def load_assets(conn: psycopg.Connection) -> pd.DataFrame:
    """The ``assets`` reference table (mutable metadata, not a point-in-time table)."""
    cur = conn.execute(
        "SELECT asset_id, ticker, asset_class, name, cik, gics_sector, sector_source, sic, "
        "security_key, listed_until FROM assets ORDER BY asset_id"
    )
    names = [d.name for d in cur.description or []]
    return pd.DataFrame(cur.fetchall(), columns=names)


def ticker_history(conn: psycopg.Connection) -> pd.DataFrame:
    """Ticker intervals per asset from ``aliases`` (kind 'ticker'): ``[valid_from, valid_to)``."""
    cur = conn.execute(
        "SELECT asset_id, alias AS symbol, valid_from, valid_to FROM aliases WHERE kind = 'ticker'"
    )
    return pd.DataFrame(cur.fetchall(), columns=["asset_id", "symbol", "valid_from", "valid_to"])


# ---------------------------------------------------------------- bars
def bar_date(ts: pd.Series, asset_class: pd.Series) -> pd.Series:
    """Session date of a bar: ET date for exchange-traded bars, UTC date for crypto (D-024)."""
    ts = pd.to_datetime(ts, utc=True)
    et = ts.dt.tz_convert(ET).dt.date
    utc = ts.dt.date
    return et.where(asset_class.ne("crypto"), utc)


def load_bars(
    conn: psycopg.Connection,
    asof: datetime,
    cfg: FeatureBarsCfg,
    *,
    asset_ids: Iterable[int] | None = None,
    since: datetime | None = None,
    exclude: Iterable[tuple[int, str]] = (),
) -> pd.DataFrame:
    """Daily bars released by ``asof``, one row per (asset, date) after source priority.

    ``exclude`` lists ``(asset_id, source)`` pairs whose bars are hidden (identity overrides,
    D-035: another company's history stored under a reused symbol).
    Columns: asset_id, date, open, high, low, close, volume, source, released_at, asset_class.
    """
    where_in = {"asset_id": list(asset_ids)} if asset_ids is not None else None
    df = frame_asof(
        conn,
        "raw_prices",
        asof,
        where={"timeframe": "1d"},
        where_in=where_in,
        columns=BAR_COLUMNS,
        order_by=("asset_id", "ts"),
    )
    if df.empty:
        return df.assign(date=pd.Series(dtype=object), asset_class=pd.Series(dtype=object))
    if since is not None:
        df = df[pd.to_datetime(df["ts"], utc=True) >= pd.Timestamp(since)]
    classes = load_assets(conn).set_index("asset_id")["asset_class"]
    df = df.assign(asset_class=df["asset_id"].map(classes))
    df = df[df["source"].isin(cfg.source_priority)]
    excl = set(exclude)
    if excl:
        keys = pd.Series(list(zip(df["asset_id"], df["source"], strict=True)), index=df.index)
        df = df[~keys.isin(excl)]
    df = df[(df["source"] != "yfinance") | df["asset_class"].isin(cfg.yfinance_classes)]
    filler = df["asset_class"].isin(cfg.drop_zero_volume_classes) & ~(df["volume"] > 0)
    df = df[~filler]
    rank = {s: i for i, s in enumerate(cfg.source_priority)}
    df = (
        df.assign(_rank=df["source"].map(rank), date=bar_date(df["ts"], df["asset_class"]))
        .sort_values(["asset_id", "date", "_rank"])
        .drop_duplicates(["asset_id", "date"], keep="first")
        .drop(columns=["_rank", "ts"])
    )
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


def bar_exclusions(conn: psycopg.Connection, overrides: Sequence) -> list[tuple[int, str]]:
    """``(asset_id, source)`` pairs to hide, from ``collect.identity_overrides``."""
    keys = {o.security_key: o.exclude_sources for o in overrides if o.exclude_sources}
    if not keys:
        return []
    rows = conn.execute(
        "SELECT security_key, asset_id FROM assets WHERE security_key = ANY(%s)", (list(keys),)
    ).fetchall()
    return [(int(aid), src) for key, aid in rows for src in keys[key]]


def bars_known_at(bars: pd.DataFrame, asof: datetime) -> pd.DataFrame:
    """The rows of a :func:`load_bars` frame that were released by ``asof``."""
    return bars[bars["released_at"] <= pd.Timestamp(asof)]


# ---------------------------------------------------------------- corporate actions
def load_corp_actions(conn: psycopg.Connection, asof: datetime) -> pd.DataFrame:
    """Alpaca corporate actions released by ``asof`` (``raw_events`` source ``alpaca_ca``).

    Columns: kind, symbol, ex_date, released_at, old_rate, new_rate, rate, special, external_id.
    A re-observed action with changed content is a new raw row; the newest released one wins.
    """
    df = frame_asof(
        conn,
        "raw_events",
        asof,
        where={"source": "alpaca_ca"},
        columns=["external_id", "kind", "payload", "released_at"],
        order_by=("released_at",),
    )
    cols = [
        "external_id",
        "kind",
        "symbol",
        "ex_date",
        "released_at",
        "old_rate",
        "new_rate",
        "rate",
        "special",
        "old_symbol",
        "new_symbol",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    p = df["payload"]
    out = pd.DataFrame(
        {
            "external_id": df["external_id"],
            "kind": df["kind"],
            "symbol": p.map(lambda x: x.get("symbol") or x.get("source_symbol")),
            # name changes carry only a process date
            "ex_date": pd.to_datetime(
                p.map(lambda x: x.get("ex_date") or x.get("process_date"))
            ).dt.date,
            "released_at": pd.to_datetime(df["released_at"], utc=True),
            "old_rate": p.map(lambda x: x.get("old_rate")).astype(float),
            "new_rate": p.map(lambda x: x.get("new_rate")).astype(float),
            "rate": p.map(lambda x: x.get("rate")).astype(float),
            "special": p.map(lambda x: bool(x.get("special", False))),
            "old_symbol": p.map(lambda x: x.get("old_symbol")),
            "new_symbol": p.map(lambda x: x.get("new_symbol")),
        }
    )
    return out.drop_duplicates("external_id", keep="last").reset_index(drop=True)


def map_actions_to_assets(
    actions: pd.DataFrame,
    tickers: pd.DataFrame,
    assets: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``asset_id`` to each action: the asset whose ticker was ``symbol`` on ``ex_date``.

    Ticker intervals come from ``aliases`` (``[valid_from, valid_to)``, open ends allowed). If
    several assets held the symbol (ticker reuse), the one with a bar before ``ex_date`` within
    10 days is kept (each, if several). Symbols with no alias fall back to ``assets.ticker``.
    Unmapped actions are dropped (with no bars for the symbol they can't adjust anything).
    """
    if actions.empty:
        return actions.assign(asset_id=pd.Series(dtype="int64"))
    by_symbol: dict[str, list[tuple[int, date | None, date | None]]] = {}
    for r in tickers.itertuples(index=False):
        by_symbol.setdefault(r.symbol, []).append((int(r.asset_id), r.valid_from, r.valid_to))
    fallback = assets.groupby("ticker")["asset_id"].apply(list).to_dict()
    bar_dates = bars.groupby("asset_id")["date"].apply(lambda s: pd.Index(sorted(s))).to_dict()

    def has_bar_before(aid: int, d: date) -> bool:
        idx = bar_dates.get(aid)
        if idx is None or len(idx) == 0:
            return False
        pos = idx.searchsorted(d) - 1
        return pos >= 0 and (d - idx[pos]).days <= 10

    rows: list[tuple[int, int]] = []  # (action position, asset_id)
    for pos, r in enumerate(actions.itertuples(index=False)):
        d = r.ex_date
        cands = [
            aid
            for aid, lo, hi in by_symbol.get(r.symbol, [])
            if (lo is None or lo <= d) and (hi is None or d < hi)
        ]
        if not cands and r.symbol not in by_symbol:
            cands = list(fallback.get(r.symbol, []))
        cands = list(dict.fromkeys(cands))
        if len(cands) > 1:
            # Several assets held the symbol: keep those trading around the ex date. If more
            # than one still qualifies (two assets carrying the same history), the action goes
            # to each; splits are re-checked against each asset's own prices in adjust.py.
            cands = [a for a in cands if has_bar_before(a, d)]
        rows.extend((pos, a) for a in cands)
    dropped = len(actions) - len({p for p, _ in rows})
    if dropped:
        log.info("corp_actions_unmapped", extra={"count": dropped})
    if not rows:
        return actions.iloc[0:0].assign(asset_id=pd.Series(dtype="int64"))
    pos, aids = zip(*rows, strict=True)
    out = actions.iloc[list(pos)].assign(asset_id=list(aids))
    return out.astype({"asset_id": "int64"}).reset_index(drop=True)


def with_name_changes(
    tickers: pd.DataFrame, actions: pd.DataFrame, assets: pd.DataFrame
) -> pd.DataFrame:
    """Add ``[.., change date)`` intervals for old symbols from Alpaca name changes.

    Membership-based ticker history (D-030) only covers renames seen while a security was in the
    index; CHK→EXE (2024) happened outside it, so CHK-era actions couldn't find EXE. An interval
    is only added when the asset holding the new symbol has no interval for the old one yet.
    """
    nc = actions[actions["kind"] == "ca_name_changes"].dropna(subset=["old_symbol", "new_symbol"])
    if nc.empty:
        return tickers
    held = {(int(r.asset_id), r.symbol) for r in tickers.itertuples(index=False)}
    rows = []
    for r in nc.itertuples(index=False):
        d = r.ex_date
        owners = [
            int(x.asset_id)
            for x in tickers.itertuples(index=False)
            if x.symbol == r.new_symbol
            and (x.valid_from is None or x.valid_from <= d)
            and (x.valid_to is None or d < x.valid_to)
        ]
        if not owners:
            owners = assets.loc[assets["ticker"] == r.new_symbol, "asset_id"].astype(int).tolist()
        if len(owners) != 1 or (owners[0], r.old_symbol) in held:
            continue
        rows.append((owners[0], r.old_symbol, None, d))
        held.add((owners[0], r.old_symbol))
    if not rows:
        return tickers
    extra = pd.DataFrame(rows, columns=["asset_id", "symbol", "valid_from", "valid_to"])
    return pd.concat([tickers, extra], ignore_index=True)


def manual_actions(manual: Sequence, assets: pd.DataFrame) -> pd.DataFrame:
    """Hand-added actions from ``features.adjust.manual``, already mapped to ``asset_id``."""
    keys = assets.dropna(subset=["security_key"]).set_index("security_key")["asset_id"]
    rows = []
    for m in manual:
        if m.security_key not in keys.index:
            log.warning("manual_action_unknown_security", extra={"security_key": m.security_key})
            continue
        ex = pd.Timestamp(m.ex_date).tz_localize(ET)
        rows.append(
            {
                "external_id": f"manual:{m.security_key}:{m.kind}:{m.ex_date}",
                "kind": m.kind,
                "symbol": None,
                "ex_date": m.ex_date,
                "released_at": ex.tz_convert(UTC),
                "old_rate": m.old_rate if m.old_rate is not None else float("nan"),
                "new_rate": m.new_rate if m.new_rate is not None else float("nan"),
                "rate": m.rate if m.rate is not None else float("nan"),
                "special": m.special,
                "old_symbol": None,
                "new_symbol": None,
                "asset_id": int(keys[m.security_key]),
                "manual": True,
            }
        )
    return pd.DataFrame(rows)


def mapped_corp_actions(
    conn: psycopg.Connection, asof: datetime, bars: pd.DataFrame, manual: Sequence = ()
) -> pd.DataFrame:
    """Alpaca + manual corporate actions known at ``asof``, each with its ``asset_id``."""
    assets = load_assets(conn)
    acts = load_corp_actions(conn, asof)
    tickers = with_name_changes(ticker_history(conn), acts, assets)
    out = map_actions_to_assets(acts, tickers, assets, bars).assign(manual=False)
    extra = manual_actions(manual, assets)
    if not extra.empty:
        extra = extra[extra["released_at"] <= pd.Timestamp(asof)]
        out = pd.concat([out, extra], ignore_index=True)
    return out


# ---------------------------------------------------------------- earnings calendar
EARNINGS_COLUMNS = [
    "asset_id",
    "quarter_key",
    "report_date",
    "pre_market",
    "surprise",
    "approx",
    "released_at",
]


def _pct(x: object) -> float:
    try:
        return float(str(x).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def load_earnings(conn: psycopg.Connection, asof: datetime, bars: pd.DataFrame) -> pd.DataFrame:
    """Earnings-calendar observations released by ``asof`` (``raw_events`` source ``nasdaq``).

    One row per observation (a moved date is a new row; the older one is superseded from the
    newer row's ``released_at`` on). ``surprise`` is Nasdaq's % surprise, present only once the
    report happened; features must not use it before the report (see ``price.py``). Each row is
    mapped to the asset that held its symbol on the report date, like corporate actions.
    """
    df = frame_asof(
        conn,
        "raw_events",
        asof,
        where={"source": "nasdaq", "kind": "earnings"},
        columns=["external_id", "payload", "released_at"],
        order_by=("released_at",),
    )
    if df.empty:
        return pd.DataFrame(columns=EARNINGS_COLUMNS)
    p = df["payload"]
    obs = pd.DataFrame(
        {
            "quarter_key": df["external_id"],
            "symbol": p.map(lambda x: x.get("symbol")),
            "ex_date": pd.to_datetime(p.map(lambda x: x.get("report_date"))).dt.date,
            "pre_market": p.map(lambda x: x.get("time") == "time-pre-market"),
            "surprise": p.map(lambda x: _pct(x.get("surprise"))),
            "approx": p.map(lambda x: bool(x.get("approx", False))),
            "released_at": pd.to_datetime(df["released_at"], utc=True),
        }
    )
    assets = load_assets(conn)
    mapped = map_actions_to_assets(obs, ticker_history(conn), assets, bars)
    return mapped.rename(columns={"ex_date": "report_date"})[EARNINGS_COLUMNS]


# ---------------------------------------------------------------- fundamentals, membership
def load_fundamentals(
    conn: psycopg.Connection,
    asof: datetime,
    metrics: Sequence[str],
    asset_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """EDGAR facts released by ``asof`` (every accession, so restatements stay visible in time)."""
    where_in: dict[str, list] = {"metric": list(metrics)}
    if asset_ids is not None:
        where_in["asset_id"] = list(asset_ids)
    df = frame_asof(
        conn,
        "raw_fundamentals",
        asof,
        where_in=where_in,
        columns=["asset_id", "fiscal_period", "metric", "value", "unit", "accession",
                 "released_at"],  # fmt: skip
        order_by=("asset_id", "released_at", "accession"),
    )
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    return df


def load_membership(conn: psycopg.Connection, asof: datetime, index_code: str) -> pd.DataFrame:
    """Every index snapshot row released by ``asof``: snapshot_at, ticker, asset_id, released_at."""
    df = frame_asof(
        conn,
        "index_membership",
        asof,
        where={"index_code": index_code},
        columns=["snapshot_at", "ticker", "source", "asset_id", "released_at"],
        order_by=("released_at", "snapshot_at"),
    )
    for c in ("snapshot_at", "released_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def members_at(membership: pd.DataFrame, asof: datetime) -> set[int]:
    """Mapped members of the newest snapshot released by ``asof`` (as-of rule, D-031 item 9)."""
    m = membership[membership["released_at"] <= pd.Timestamp(asof)]
    if m.empty:
        return set()
    snap = m[m["snapshot_at"] == m["snapshot_at"].max()]
    return {int(a) for a in snap["asset_id"].dropna()}


# ---------------------------------------------------------------- macro
def load_macro(
    conn: psycopg.Connection, series: Sequence[str], asof: datetime, fixes=None
) -> pd.DataFrame:
    """Every vintage of ``series`` released by ``asof`` (ALFRED rows, source 'alfred').

    ``fixes`` (``features.market.macro_fixes``) rescales vintages from before a unit change and
    moves the first vintage of late-listed series back to period + lag (see the config).
    """
    late = dict(fixes.late_listed_lag_days) if fixes is not None else {}
    cols = ["series_id", "period", "value", "vintage", "released_at"]
    normal = [s for s in series if s not in late]
    df = frame_asof(conn, "raw_macro", asof, where={"source": "alfred"},
                    where_in={"series_id": normal}, columns=cols,
                    order_by=("released_at", "series_id", "period"))  # fmt: skip
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    for sid, lag in late.items():
        if sid not in series:
            continue
        # every vintage (the first one moves earlier); then cut on the moved release times
        x = frame_asof(conn, "raw_macro", datetime.max.replace(tzinfo=UTC),
                       where={"source": "alfred", "series_id": sid}, columns=cols,
                       order_by=("released_at", "period"))  # fmt: skip
        if x.empty:
            continue
        x["released_at"] = pd.to_datetime(x["released_at"], utc=True)
        first = x["vintage"].min()
        moved = pd.to_datetime(x["period"]).dt.tz_localize(ET) + pd.Timedelta(days=lag)
        is_first = x["vintage"] == first
        x.loc[is_first, "released_at"] = moved[is_first].dt.tz_convert(UTC).where(
            moved[is_first].dt.tz_convert(UTC) < x.loc[is_first, "released_at"],
            x.loc[is_first, "released_at"],
        )
        x = x[x["released_at"] <= pd.Timestamp(asof)]
        df = pd.concat([df, x], ignore_index=True)
    if fixes is not None:
        for u in fixes.unit_changes:
            hit = (df["series_id"] == u.series) & (df["vintage"] < u.before_vintage)
            df.loc[hit, "value"] = df.loc[hit, "value"].astype(float) * u.multiply
    return df.sort_values(["released_at", "series_id", "period"], kind="stable").reset_index(
        drop=True
    )


class MacroState:
    """Macro series as known at a moving decision time (never back-fills, never peeks).

    Feed it the rows of :func:`load_macro` once, then call :meth:`advance` with increasing
    decision times. :meth:`series` returns, per period, the newest vintage released so far.
    """

    def __init__(self, rows: pd.DataFrame) -> None:
        self._rows = rows.sort_values(["released_at", "vintage"], kind="stable")
        # as_unit: pandas 3 infers us/s resolutions, and int64 would then not be nanoseconds
        rel = pd.to_datetime(self._rows["released_at"], utc=True).dt.as_unit("ns")
        self._rel_ns = rel.astype("int64").to_numpy()  # UTC epoch ns
        self._rel = rel.to_numpy()
        self._sid = self._rows["series_id"].to_numpy()
        self._period = self._rows["period"].to_numpy()
        self._value = self._rows["value"].to_numpy()
        self._vintage = self._rows["vintage"].to_numpy()
        self._i = 0
        self._asof: pd.Timestamp | None = None
        self._known: dict[str, dict[date, tuple[date, float | None]]] = {}
        self._last_release: dict[str, pd.Timestamp] = {}
        self._cache: dict[str, pd.Series] = {}

    def advance(self, asof: datetime) -> None:
        t = pd.Timestamp(asof)
        if t.tzinfo is None:
            raise ValueError("asof must be timezone-aware")
        if self._asof is not None and t < self._asof:
            raise ValueError("MacroState only moves forward in time")
        self._asof = t
        n = len(self._rel_ns)
        t_ns = t.tz_convert(UTC).as_unit("ns").value
        while self._i < n and self._rel_ns[self._i] <= t_ns:
            sid = str(self._sid[self._i])
            per = self._period[self._i]
            vin = self._vintage[self._i]
            known = self._known.setdefault(sid, {})
            old = known.get(per)
            if old is None or vin >= old[0]:
                known[per] = (vin, self._value[self._i])
            self._last_release[sid] = pd.Timestamp(self._rel_ns[self._i], tz=UTC)
            self._cache.pop(sid, None)
            self._i += 1

    def series(self, series_id: str) -> pd.Series:
        """Values by period as known now (``None`` from the source becomes NaN)."""
        if self._asof is None:
            raise RuntimeError("call advance() first")
        s = self._cache.get(series_id)
        if s is None:
            known = self._known.get(series_id, {})
            items = sorted(known.items())
            s = pd.Series(
                [float("nan") if v[1] is None else float(v[1]) for _, v in items],
                index=pd.Index([k for k, _ in items], name="period"),
                dtype=float,
                name=series_id,
            )
            self._cache[series_id] = s
        return s

    def last_release(self, series_id: str) -> pd.Timestamp | None:
        """When the newest row of ``series_id`` known now was released."""
        return self._last_release.get(series_id)
