"""Exact + near dedup (component doc 6.7, D-009: in-house 64-bit SimHash).

- exact: canonical URL sha256 (``article_seen``) and normalized-title sha256
  (``articles.title_hash``)
- near: 64-bit SimHash over word 5-shingles; Hamming distance <= 3 against the last 30 days

Duplicates found here are still stored, with ``near_dup_of`` pointing at the earlier article;
only an already-seen URL is skipped. The Phase 8 quota step decides what to keep.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta

import psycopg

_WORD = re.compile(r"\w+", re.UNICODE)
_MASK = (1 << 64) - 1


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(_WORD.findall(t))


def title_hash(title: str | None) -> str | None:
    n = normalize_title(title or "")
    return hashlib.sha256(n.encode("utf-8")).hexdigest() if n else None


def shingles(text: str, k: int) -> list[str]:
    toks = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    if not toks:
        return []
    if len(toks) <= k:
        return [" ".join(toks)]
    return [" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)]


def simhash64(text: str, k: int = 5) -> int | None:
    """Unsigned 64-bit SimHash of ``text``; None for text without words."""
    sh = shingles(text, k)
    if not sh:
        return None
    v = [0] * 64
    for s in sh:
        h = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(64) if v[i] > 0)


def to_signed(u: int) -> int:
    """Unsigned 64-bit -> Postgres bigint."""
    return u - (1 << 64) if u >= (1 << 63) else u


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & _MASK).bit_count()


def find_title_dup(conn: psycopg.Connection, th: str | None) -> int | None:
    if th is None:
        return None
    row = conn.execute(
        "SELECT article_id FROM articles WHERE title_hash = %s ORDER BY released_at LIMIT 1", (th,)
    ).fetchone()
    return row[0] if row else None


def find_near_dup(
    conn: psycopg.Connection,
    simhash_signed: int | None,
    now: datetime,
    window_days: int,
    max_hamming: int,
) -> int | None:
    """Earliest article in the window within ``max_hamming`` bits (SQL ``bit_count`` on XOR)."""
    if simhash_signed is None:
        return None
    row = conn.execute(
        """
        SELECT article_id FROM articles
        WHERE simhash IS NOT NULL AND released_at >= %s
          AND bit_count((simhash # %s)::bit(64)) <= %s
        ORDER BY released_at LIMIT 1
        """,
        (now - timedelta(days=window_days), simhash_signed, max_hamming),
    ).fetchone()
    return row[0] if row else None
