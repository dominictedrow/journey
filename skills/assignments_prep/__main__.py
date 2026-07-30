"""assignments-prep — nearest due assignment -> a short Claude study plan.

Gives the Assignments tracker's zero-Claude corner a judgment layer without
touching its schema: reads the existing classes/assignments tables (the same
ones app/db.js's CRUD writes to) and asks Claude to turn the single nearest
undone assignment into a concrete study plan. No new network host, no new
storage.

Run: python -m skills.assignments_prep   (no CLI inputs, on-demand skill)
"""

import json
import re
from datetime import date

from .._lib import claude, envelope, store

SKILL_ID = "assignments_prep"


def _nearest_assignment(conn):
    """Earliest-due, not-done assignment across all classes. Overdue items
    still have the earliest due_date, so they naturally sort first."""
    row = conn.execute(
        """
        SELECT a.id, a.title, a.due_date, c.name AS class_name
        FROM assignments a JOIN classes c ON c.id = a.class_id
        WHERE a.done = 0
        ORDER BY a.due_date ASC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


def _days_until(due_date_str):
    try:
        due = date.fromisoformat(due_date_str)
    except ValueError:
        return None
    return (due - date.today()).days


def _build_prompt(assignment, days_until):
    if days_until is None:
        timing = f"due date: {assignment['due_date']} (format unclear)"
    elif days_until < 0:
        timing = f"OVERDUE by {-days_until} day(s) (was due {assignment['due_date']})"
    elif days_until == 0:
        timing = "due TODAY"
    else:
        timing = f"due in {days_until} day(s) ({assignment['due_date']})"

    return (
        "A student needs a short, concrete study/work plan for one assignment.\n\n"
        f"Class: {assignment['class_name']}\n"
        f"Assignment: {assignment['title']}\n"
        f"Timing: {timing}\n\n"
        "Return ONLY strict JSON, no markdown fences, no commentary, matching "
        "this schema:\n"
        '{"steps": [{"step": "...", "time_estimate": "..."}], '
        '"first_action": "..."}\n\n'
        "Rules:\n"
        "- steps: 3-6 concrete, ordered steps to actually finish this "
        "assignment, each with a rough time estimate (e.g. \"20 min\").\n"
        "- first_action: the single most useful thing to do right now, in "
        "one sentence.\n"
        "- The assignment title is all the context you have about the "
        "subject matter — infer reasonably from it, don't ask clarifying "
        "questions.\n"
    )


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def main():
    conn = store.connect()
    assignment = _nearest_assignment(conn)
    conn.close()

    if assignment is None:
        envelope.emit(
            SKILL_ID,
            ok=True,
            summary="No open assignments",
            sections=[
                {
                    "type": "text",
                    "title": "Nothing due",
                    "body": "No undone assignments in the tracker — nothing to prep for.",
                }
            ],
        )
        return

    days_until = _days_until(assignment["due_date"])

    envelope.progress(
        f"assignments_prep: nearest due is '{assignment['title']}' "
        f"({assignment['class_name']}, {assignment['due_date']})"
    )

    try:
        response = claude.ask(_build_prompt(assignment, days_until))
        parsed = _extract_json(response["result"])
        steps = parsed["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("Claude response contained no steps")
        first_action = str(parsed.get("first_action", "")).strip()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        envelope.progress(f"assignments_prep: failed - {exc}")
        envelope.emit(
            SKILL_ID,
            ok=False,
            summary=f"Failed to build a study plan for '{assignment['title']}'",
            sections=[],
            error=str(exc),
        )
        return

    if days_until is None:
        due_label = assignment["due_date"]
    elif days_until < 0:
        due_label = f"overdue by {-days_until}d"
    elif days_until == 0:
        due_label = "today"
    else:
        due_label = f"in {days_until}d"

    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": "Class", "value": assignment["class_name"]},
                {"label": "Due", "value": due_label},
                {"label": "Steps", "value": len(steps)},
            ],
        },
        {
            "type": "table",
            "title": f"Plan: {assignment['title']}",
            "columns": ["Step", "Time"],
            "rows": [[s.get("step", ""), s.get("time_estimate", "")] for s in steps],
        },
    ]
    if first_action:
        sections.append({"type": "text", "title": "Start here", "body": first_action})

    envelope.progress("assignments_prep: done")
    envelope.emit(
        SKILL_ID,
        ok=True,
        summary=f"Study plan ready for '{assignment['title']}' ({due_label})",
        sections=sections,
    )


if __name__ == "__main__":
    main()
