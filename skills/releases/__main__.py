"""Monthly release radar across Dom's eight hype categories.

Checks upcoming/rumored releases (Pokemon set, One Piece TCG set, MTG set,
major-artist album, hyped sneaker, hyped tech, revolutionary AI model/concept,
hyped movie) via one `claude -p` web-search call, caches results in the
`release_items` table keyed on (category, item), and only re-checks a category
if it hasn't been checked yet this calendar month. Rendering always happens
from the durable cache, not just this run's freshly-fetched items, so a
category found last month still shows up until it ages out of the window.

Run: python -m skills.releases   (no CLI inputs, scheduled skill)
"""

import json
import re
from datetime import datetime

from .._lib import claude, envelope, store

CATEGORIES = [
    "Pokémon TCG set",
    "One Piece TCG set",
    "MTG set",
    "major-artist album",
    "hyped sneaker",
    "hyped tech",
    "revolutionary AI model/concept",
    "hyped movie",
]

# How far back/forward of "now" counts as "reasonably upcoming/recent" when
# rendering from the cache. Generous on both sides since release dates are
# often vague ("Fall 2026") and Dom would rather see a borderline item than
# miss one.
_MONTHS_BACK = 3
_MONTHS_FORWARD = 12

_SEASON_MONTH = {
    "q1": 2, "q2": 5, "q3": 8, "q4": 11,
    "spring": 4, "summer": 7, "fall": 10, "autumn": 10, "winter": 1,
}


def _month_key(year, month):
    return year * 12 + month


def _current_month_str():
    return datetime.now().strftime("%Y-%m")


def _parse_date_key(date_str):
    """Best-effort (year*12+month) key for date-window filtering.

    Release dates from web search vary wildly in precision ("2026-09-15",
    "Fall 2026", "2027", "TBD"). Returns None when nothing parseable is
    found, in which case the caller should include the item by default
    rather than silently dropping it.
    """
    if not date_str:
        return None
    s = date_str.strip().lower()

    m = re.match(r"(\d{4})-(\d{2})", s)
    if m:
        return _month_key(int(m.group(1)), int(m.group(2)))

    m = re.search(r"(20\d{2})", s)
    if not m:
        return None
    year = int(m.group(1))
    month = 6
    for key, mo in _SEASON_MONTH.items():
        if key in s:
            month = mo
            break
    return _month_key(year, month)


def _build_prompt(categories):
    cat_list = "\n".join(f"- {c}" for c in categories)
    return (
        "Search the web for current known and rumored upcoming releases in "
        "each of the following hype categories:\n"
        f"{cat_list}\n\n"
        "For each category, return 1 to 4 of the most relevant upcoming or "
        "very-recent releases (confirmed or credibly rumored). For each item "
        "give:\n"
        "- item: the specific name (e.g. set name, album title, model name)\n"
        "- category: exactly one of the category strings above\n"
        "- date: the release date, as precise as known (YYYY-MM-DD, "
        "YYYY-MM, a season/quarter like \"Fall 2026\", or \"TBD\")\n"
        "- confidence: one of \"confirmed\", \"rumored\", \"unconfirmed\"\n"
        "- source_url: a real URL backing the claim\n\n"
        "Respond with ONLY strict JSON matching this exact schema, no "
        "markdown fences, no commentary:\n"
        '{"releases": [{"item": "...", "category": "...", "date": "...", '
        '"confidence": "...", "source_url": "..."}]}'
    )


def _extract_json(raw_result):
    """Claude's `result` text should be pure JSON, but strip markdown fences
    defensively in case a code block slipped in."""
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _checked_categories_this_month(conn, current_month):
    rows = conn.execute(
        "SELECT DISTINCT category FROM release_items WHERE checked_month = ?",
        (current_month,),
    ).fetchall()
    return {row["category"] for row in rows}


def _upsert_releases(conn, releases, current_month):
    count = 0
    for r in releases:
        item = (r.get("item") or "").strip()
        category = (r.get("category") or "").strip()
        if not item or not category:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO release_items "
            "(category, item, date, confidence, source_url, checked_month) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                category,
                item,
                r.get("date") or "",
                r.get("confidence") or "unconfirmed",
                r.get("source_url") or "",
                current_month,
            ),
        )
        count += 1
    conn.commit()
    return count


def _render_rows(conn):
    """Select the durable cache, filter to the upcoming/recent window, sort
    by date. Returns (rows_for_table, all_filtered_records)."""
    now_key = _month_key(datetime.now().year, datetime.now().month)
    lower, upper = now_key - _MONTHS_BACK, now_key + _MONTHS_FORWARD

    records = conn.execute(
        "SELECT category, item, date, confidence, source_url FROM release_items"
    ).fetchall()

    filtered = []
    for rec in records:
        key = _parse_date_key(rec["date"])
        if key is None or (lower <= key <= upper):
            filtered.append((key, rec))

    # Unparseable dates (key is None) sort last, otherwise chronological.
    filtered.sort(key=lambda pair: (pair[0] is None, pair[0] or 0))

    rows = [
        [rec["item"], rec["category"], rec["date"] or "TBD", rec["confidence"] or "unconfirmed"]
        for _, rec in filtered
    ]
    return rows, [rec for _, rec in filtered]


def main():
    current_month = _current_month_str()
    conn = store.connect()

    already_checked = _checked_categories_this_month(conn, current_month)
    to_check = [c for c in CATEGORIES if c not in already_checked]

    envelope.progress(
        f"releases: month={current_month} "
        f"skipped={sorted(already_checked)} to_check={sorted(to_check)}"
    )

    new_count = 0
    if to_check:
        envelope.progress(f"releases: asking Claude about {len(to_check)} categories")
        try:
            response = claude.ask(_build_prompt(to_check), use_cache=False)
            parsed = _extract_json(response["result"])
            releases = parsed.get("releases", [])
            if not isinstance(releases, list):
                raise ValueError("Claude response 'releases' was not a list")
            new_count = _upsert_releases(conn, releases, current_month)
            envelope.progress(f"releases: upserted {new_count} items")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
            envelope.progress(f"releases: Claude check failed - {exc}")
            conn.close()
            envelope.emit(
                "releases",
                ok=False,
                summary="Failed to check upcoming releases",
                sections=[],
                error=str(exc),
            )
            return
    else:
        envelope.progress("releases: all categories already checked this month, using cache only")

    rows, filtered_records = _render_rows(conn)
    conn.close()

    confirmed_count = sum(1 for r in filtered_records if (r["confidence"] or "").lower() == "confirmed")
    category_count = len({r["category"] for r in filtered_records})

    sections = [
        {
            "type": "table",
            "title": "Upcoming releases",
            "columns": ["Item", "Category", "Date", "Confidence"],
            "rows": rows,
        },
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": "Confirmed", "value": confirmed_count},
                {"label": "Categories covered", "value": category_count},
                {"label": "Checked this run", "value": len(to_check)},
            ],
        },
    ]

    summary = (
        f"{new_count} item(s) updated across {len(to_check)} category(ies) checked this run; "
        f"{confirmed_count} confirmed release(s) upcoming across {category_count} categories."
    )

    envelope.emit("releases", ok=True, summary=summary, sections=sections)


if __name__ == "__main__":
    main()
