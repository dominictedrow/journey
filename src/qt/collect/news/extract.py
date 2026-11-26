"""Full-text extraction (trafilatura) and the raw HTML archive (component doc 6.7)."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import trafilatura

Status = Literal["ok", "short", "failed", "skipped"]


@dataclass(frozen=True)
class Extracted:
    body: str | None
    word_count: int
    status: Status


def word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def extract(html: str | None, url: str, min_words: int) -> Extracted:
    """Extract the article body. Short bodies are kept and flagged; filtering is downstream."""
    if not html:
        return Extracted(None, 0, "failed")
    try:
        body = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    except Exception:  # noqa: BLE001  (malformed pages must never break the loop)
        body = None
    if not body:
        return Extracted(None, 0, "failed")
    n = word_count(body)
    return Extracted(body, n, "ok" if n >= min_words else "short")


class HtmlArchive:
    """Append-only raw HTML archive: ``<root>/YYYY/MM/DD/<source>.jsonl.gz`` (UTC fetch date).

    Each write appends one gzip member holding one JSON line, so the file is always valid gzip
    and a crash loses at most the line being written. Allows re-extraction later (D-022).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, source: str, fetched_at: datetime) -> Path:
        d = fetched_at.astimezone(UTC)
        return self.root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / f"{source}.jsonl.gz"

    def write(
        self, source: str, fetched_at: datetime, url: str, status: int | None, html: str | None
    ) -> Path:
        p = self.path_for(source, fetched_at)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"url": url, "fetched_at": fetched_at.isoformat(), "status": status, "html": html},
            ensure_ascii=False,
        )
        with gzip.open(p, "ab") as f:
            f.write(line.encode("utf-8") + b"\n")
        return p

    @staticmethod
    def read(path: Path) -> list[dict]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
