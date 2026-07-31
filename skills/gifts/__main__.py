"""gifts — upcoming occasions (from the manually-maintained People tab) -> gift ideas.

Weekly sibling of the Assignments tracker's manual-CRUD-plus-judgment-layer
shape: `ui/tabs/Gifts.jsx` (mounted at the "People" sidebar tab) is plain
CRUD over the `people`/`occasions` tables, zero Claude calls, exactly like
Assignments. This skill is the scheduled judgment layer on top of it: reads
whatever's in those tables, finds occasions landing within the next
`_LOOKAHEAD_DAYS`, and asks Claude for gift ideas for all of them in a
single call.

`occasions.month`/`occasions.day` describe a recurring annual date (no
year), so "next occurrence" is always computed relative to today rather
than stored.

Run: python -m skills.gifts   (no CLI inputs, scheduled skill)
"""

import json
import re
from datetime import date, timedelta

from .._lib import claude, envelope, store

SKILL_ID = "gifts"

# How far ahead an occasion has to be to surface here — long enough to
# actually shop/plan, short enough that "in 2 weeks" still reads as urgent.
_LOOKAHEAD_DAYS = 21


def _next_occurrence(today, month, day):
    """Next calendar date (>= today) matching (month, day), any year."""
    try:
        candidate = date(today.year, month, day)
    except ValueError:
        return None  # e.g. Feb 29 in a non-leap year
    if candidate < today:
        try:
            candidate = date(today.year + 1, month, day)
        except ValueError:
            return None
    return candidate


def _upcoming_occasions(conn):
    today = date.today()
    rows = conn.execute(
        """
        SELECT o.id, o.label, o.month, o.day, p.name AS person_name
        FROM occasions o JOIN people p ON p.id = o.person_id
        """
    ).fetchall()

    upcoming = []
    for row in rows:
        next_date = _next_occurrence(today, row["month"], row["day"])
        if next_date is None:
            continue
        days_until = (next_date - today).days
        if days_until <= _LOOKAHEAD_DAYS:
            upcoming.append(
                {
                    "person": row["person_name"],
                    "label": row["label"],
                    "date": next_date.isoformat(),
                    "days_until": days_until,
                }
            )
    upcoming.sort(key=lambda o: o["days_until"])
    return upcoming


def _build_prompt(occasions):
    lines = [f"{o['person']} — {o['label']} in {o['days_until']} day(s) ({o['date']})" for o in occasions]
    occasions_text = "\n".join(lines)
    return (
        "Here are upcoming personal occasions Dom needs gift ideas for:\n\n"
        f"{occasions_text}\n\n"
        "For each occasion, suggest 3 concrete, specific gift ideas (not "
        "generic categories) suited to the occasion type. Return ONLY "
        "strict JSON, no markdown fences, no commentary, matching this "
        "schema:\n"
        '{"occasions": [{"person": "...", "label": "...", '
        '"ideas": ["...", "...", "..."]}]}\n'
        "Use the exact person and label strings given above."
    )


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def main():
    conn = store.connect()
    upcoming = _upcoming_occasions(conn)
    conn.close()

    if not upcoming:
        envelope.emit(
            SKILL_ID,
            ok=True,
            summary=f"No occasions in the next {_LOOKAHEAD_DAYS} days",
            sections=[
                {
                    "type": "text",
                    "title": "Nothing upcoming",
                    "body": f"No tracked occasions land within the next {_LOOKAHEAD_DAYS} days.",
                }
            ],
        )
        return

    envelope.progress(f"gifts: {len(upcoming)} occasion(s) upcoming, asking Claude for ideas")
    try:
        response = claude.ask(_build_prompt(upcoming), use_cache=False)
        idea_map = {}
        for entry in _extract_json(response["result"]).get("occasions", []):
            key = (entry.get("person", ""), entry.get("label", ""))
            idea_map[key] = entry.get("ideas", [])
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        envelope.progress(f"gifts: Claude call failed - {exc}")
        envelope.emit(
            SKILL_ID,
            ok=False,
            summary="Failed to generate gift ideas",
            sections=[],
            error=str(exc),
        )
        return

    rows = []
    for o in upcoming:
        ideas = idea_map.get((o["person"], o["label"]), [])
        rows.append(
            [
                o["person"],
                o["label"],
                f"in {o['days_until']}d" if o["days_until"] > 0 else "today",
                "; ".join(ideas) if ideas else "—",
            ]
        )

    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [{"label": "Upcoming occasions", "value": len(upcoming)}],
        },
        {
            "type": "table",
            "title": "Gift ideas",
            "columns": ["Person", "Occasion", "When", "Ideas"],
            "rows": rows,
        },
    ]

    envelope.progress("gifts: done")
    envelope.emit(
        SKILL_ID,
        ok=True,
        summary=f"{len(upcoming)} occasion(s) in the next {_LOOKAHEAD_DAYS} days",
        sections=sections,
    )


if __name__ == "__main__":
    main()
