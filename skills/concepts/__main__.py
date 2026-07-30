"""2 AI + 2 software-engineering concept deep-dives, researched by Claude.

Run: python -m skills.concepts   (no CLI inputs, scheduled skill)

Originally planned to read Dom's own X timeline via an authenticated
Playwright session (see plan.md's "Parker Portal automation" section for
the earlier, related investigation). Tried it in practice: X's bot
detection flags the automated browser profile as "not secure" right at the
login screen -- the same wall the Parker Portal spike hit with Blackbaud --
so a personal login was never going to get through. Dropped it.

Claude now does the research itself with one web-search call (covering
X/Twitter alongside arXiv, Hacker News, GitHub, and HuggingFace), the same
pattern `releases` already uses. This trades reading Dom's specific
personal feed/lists for a fully unattended, reliable daily run.

Avoids repeating a concept covered on a recent day by naming the last
`_RECENT_LIMIT` delivered concepts directly in the prompt and asking Claude
not to repeat them, rather than filtering after the fact.
"""

import json
import re

from .._lib import claude, envelope, store

_RECENT_LIMIT = 30


def _recent_delivered_names(conn, limit=_RECENT_LIMIT):
    rows = conn.execute(
        "SELECT item_hash FROM delivered_items "
        "WHERE skill = 'concepts' ORDER BY delivered_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [row["item_hash"] for row in rows]


def _mark_delivered(conn, names):
    for name in names:
        conn.execute(
            "INSERT OR REPLACE INTO delivered_items (item_hash, skill, delivered_at) "
            "VALUES (?, 'concepts', datetime('now'))",
            (name,),
        )
    conn.commit()


def _build_prompt(avoid_names):
    avoid_line = ""
    if avoid_names:
        avoid_list = "\n".join(f"- {n}" for n in avoid_names)
        avoid_line = (
            "Do NOT repeat any of these concepts already covered recently:\n"
            f"{avoid_list}\n\n"
        )

    return f"""Search the web (including X/Twitter, Hacker News, arXiv, GitHub \
trending, and HuggingFace trending) for what's currently interesting and \
worth knowing about in AI and in software engineering.

{avoid_line}Pick exactly 2 AI concepts and 2 software-engineering concepts \
worth a deep-dive today. For each, write:
- name: a short concept name
- what_it_is: 1-2 sentences
- why_it_matters: 1-2 sentences
- resource: one real paper, repo, or post to read (name + URL)
- exercise: a concrete, hands-on 30-minute exercise

Respond with ONLY strict JSON (no markdown fences, no commentary) matching \
exactly this schema:
{{"concepts": [{{"category": "ai"|"swe", "name": "string", "what_it_is": \
"string", "why_it_matters": "string", "resource": "string", "exercise": \
"string"}}]}}"""


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def main():
    conn = store.connect()
    avoid_names = _recent_delivered_names(conn)

    envelope.progress(
        f"concepts: asking Claude to research today's concepts (avoiding {len(avoid_names)} recent)"
    )
    try:
        response = claude.ask(_build_prompt(avoid_names), use_cache=False)
        picks = _extract_json(response["result"]).get("concepts", [])
        if not isinstance(picks, list):
            raise ValueError("Claude response 'concepts' was not a list")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        conn.close()
        envelope.emit(
            "concepts",
            ok=False,
            summary="Failed to research today's concepts",
            sections=[],
            error=str(exc),
        )
        return

    ai_picks = [p for p in picks if p.get("category") == "ai"][:2]
    swe_picks = [p for p in picks if p.get("category") == "swe"][:2]
    final = ai_picks + swe_picks

    _mark_delivered(conn, [p.get("name") for p in final if p.get("name")])
    conn.close()

    sections = []
    for p in final:
        category_label = "AI" if p.get("category") == "ai" else "Software Engineering"
        body = (
            f"**Category:** {category_label}\n\n"
            f"**What it is:** {p.get('what_it_is', '')}\n\n"
            f"**Why it matters:** {p.get('why_it_matters', '')}\n\n"
            f"**Read:** {p.get('resource', '')}\n\n"
            f"**30-minute exercise:** {p.get('exercise', '')}"
        )
        sections.append({"type": "text", "title": p.get("name") or "Untitled concept", "body": body})

    sections.append(
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": "AI picks", "value": len(ai_picks)},
                {"label": "SWE picks", "value": len(swe_picks)},
            ],
        }
    )

    envelope.emit(
        "concepts",
        ok=len(final) > 0,
        summary=f"{len(ai_picks)} AI + {len(swe_picks)} software-engineering concept(s)",
        sections=sections,
        error=None if final else "Claude returned no valid picks",
    )


if __name__ == "__main__":
    main()
