"""Alias matching -> ``articles.tickers`` (component doc 6.6, Q7).

- plain aliases match case-insensitively on word boundaries
- ``case_sensitive`` aliases match exactly
- ``needs_cue`` aliases (ambiguous names like "Target", "Apple") need a case-sensitive match AND a
  finance cue anywhere in the same article
- validity dates map a name to the ticker valid on the article's date (FB -> META)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import psycopg


@dataclass(frozen=True)
class Alias:
    ticker: str
    alias: str
    case_sensitive: bool = False
    needs_cue: bool = False
    valid_from: date | None = None
    valid_to: date | None = None

    def valid_on(self, d: date) -> bool:
        return (self.valid_from is None or self.valid_from <= d) and (
            self.valid_to is None or d < self.valid_to
        )


def _word(s: str) -> str:
    return r"(?<![\w&])" + re.escape(s) + r"(?![\w&])"


class Tagger:
    def __init__(self, aliases: Iterable[Alias], finance_cues: Iterable[str]) -> None:
        self._rules = []
        for a in aliases:
            flags = 0 if (a.case_sensitive or a.needs_cue) else re.IGNORECASE
            self._rules.append((a, re.compile(_word(a.alias), flags)))
        cues = list(finance_cues)
        self._cue = re.compile("|".join(_word(c) for c in cues), re.IGNORECASE) if cues else None

    def __len__(self) -> int:
        return len(self._rules)

    def tag(self, text: str, on: date) -> list[str]:
        if not text or not self._rules:
            return []
        has_cue = bool(self._cue and self._cue.search(text))
        out = set()
        for a, rx in self._rules:
            if not a.valid_on(on) or (a.needs_cue and not has_cue):
                continue
            if rx.search(text):
                out.add(a.ticker)
        return sorted(out)


def _overlap(a0: date | None, a1: date | None, b0: date | None, b1: date | None):
    lo = max((x for x in (a0, b0) if x is not None), default=None)
    hi = min((x for x in (a1, b1) if x is not None), default=None)
    return (lo, hi) if lo is None or hi is None or lo < hi else None


def load_aliases(conn: psycopg.Connection) -> list[Alias]:
    """Aliases with each name expanded over its asset's ticker eras, so a tag is the ticker that
    was valid on the article's date ("Facebook" in 2019 -> FB, in 2023 -> META). Eras come from
    the ``ticker`` aliases; an asset without them (ETF, coin) has one era, its current ticker."""
    rows = conn.execute(
        "SELECT a.asset_id, s.ticker, a.alias, a.kind, a.case_sensitive, a.needs_cue, "
        "a.valid_from, a.valid_to FROM aliases a JOIN assets s USING (asset_id)"
    ).fetchall()
    eras: dict[int, list[tuple[str, date | None, date | None]]] = {}
    for aid, _, alias, kind, _, _, vf, vt in rows:
        if kind == "ticker":
            eras.setdefault(aid, []).append((alias, vf, vt))
    out = []
    for aid, current, alias, _, cs, cue, vf, vt in rows:
        for t, ef, et in eras.get(aid, [(current, None, None)]):
            win = _overlap(vf, vt, ef, et)
            if win is not None:
                out.append(Alias(t, alias, cs, cue, win[0], win[1]))
    return out
