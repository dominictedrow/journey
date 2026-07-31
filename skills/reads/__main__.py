"""reads — monthly book recommendations, researched by Claude.

Book-world sibling of `concepts`: one `claude -p` web-search call, same
repeat-avoidance-by-recent-history trick (name the last `_RECENT_LIMIT`
delivered titles in the prompt and ask Claude not to repeat them, rather
than filtering after the fact).

Run: python -m skills.reads   (no CLI inputs, scheduled skill)
"""

import json
import re

from .._lib import claude, envelope, store

SKILL_ID = "reads"

_RECENT_LIMIT = 30
_PICK_COUNT = 3


def _recent_delivered_titles(conn, limit=_RECENT_LIMIT):
    rows = conn.execute(
        "SELECT item_hash FROM delivered_items "
        "WHERE skill = 'reads' ORDER BY delivered_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [row["item_hash"] for row in rows]


def _mark_delivered(conn, titles):
    for title in titles:
        conn.execute(
            "INSERT OR REPLACE INTO delivered_items (item_hash, skill, delivered_at) "
            "VALUES (?, 'reads', datetime('now'))",
            (title,),
        )
    conn.commit()


def _build_prompt(avoid_titles):
    avoid_line = ""
    if avoid_titles:
        avoid_list = "\n".join(f"- {t}" for t in avoid_titles)
        avoid_line = (
            "Do NOT recommend any of these titles already recommended recently:\n"
            f"{avoid_list}\n\n"
        )

    return (
        "Search the web for notable, well-reviewed books worth reading right "
        "now — a mix of recent releases and acclaimed backlist titles, "
        "fiction or nonfiction.\n\n"
        f"{avoid_line}"
        f"Pick exactly {_PICK_COUNT} books worth recommending this month. "
        "For each, write:\n"
        "- title: the book's title\n"
        "- author: the author's name\n"
        "- hook: one sentence on what it's about\n"
        "- why_recommended: 1-2 sentences on why it's worth reading now\n\n"
        "Respond with ONLY strict JSON (no markdown fences, no commentary) "
        "matching exactly this schema:\n"
        '{"books": [{"title": "string", "author": "string", "hook": '
        '"string", "why_recommended": "string"}]}'
    )


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def main():
    conn = store.connect()
    avoid_titles = _recent_delivered_titles(conn)

    envelope.progress(
        f"reads: asking Claude for this month's books (avoiding {len(avoid_titles)} recent)"
    )
    try:
        response = claude.ask(_build_prompt(avoid_titles), use_cache=False)
        books = _extract_json(response["result"]).get("books", [])
        if not isinstance(books, list):
            raise ValueError("Claude response 'books' was not a list")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        conn.close()
        envelope.emit(
            SKILL_ID,
            ok=False,
            summary="Failed to research this month's books",
            sections=[],
            error=str(exc),
        )
        return

    books = books[:_PICK_COUNT]
    _mark_delivered(conn, [b.get("title") for b in books if b.get("title")])
    conn.close()

    sections = []
    for b in books:
        title = b.get("title") or "Untitled"
        author = b.get("author") or "Unknown"
        body = (
            f"**By:** {author}\n\n"
            f"**What it's about:** {b.get('hook', '')}\n\n"
            f"**Why now:** {b.get('why_recommended', '')}"
        )
        sections.append({"type": "text", "title": title, "body": body})

    sections.append(
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [{"label": "Picks this month", "value": len(books)}],
        }
    )

    envelope.emit(
        SKILL_ID,
        ok=len(books) > 0,
        summary=f"{len(books)} book(s) recommended this month",
        sections=sections,
        error=None if books else "Claude returned no valid picks",
    )


if __name__ == "__main__":
    main()
