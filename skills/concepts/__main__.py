"""X timeline (+ optional lists) -> 2 AI + 2 software-engineering deep-dives.

Run: python -m skills.concepts   (no CLI inputs, scheduled skill)

Primary path reads Dom's X home timeline (and any lists configured in
config.json under `concepts.x_lists`) through a persistent, headless
Playwright context (skills/_lib/browser.py), pulling ~40 candidate post
texts. Per plan.md, whether a saved X session survives headless reuse is
UNVERIFIED going into Phase 2 (the Parker Portal spike found short-lived
tokens don't survive a closed/reopened session) -- so any failure on that
path (no session, redirected to login, zero posts extracted) falls back to
free sources (arXiv, Hacker News, GitHub, HuggingFace) and the summary says
so explicitly, rather than failing silently.

Candidates are deduped against everything already delivered (`delivered_items`
table) before being handed to Claude, so the same post/paper/repo isn't
suggested twice. Claude picks exactly 2 AI + 2 SWE concepts and writes a
deep-dive brief for each: what it is, why it matters, one resource to read,
and a 30-minute exercise.
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

from .._lib import claude, envelope, net, store
from ._common import CANDIDATE_LIMIT

_FALLBACK_PER_SOURCE = 10

# Wall-clock bound on the X-fetch subprocess. Playwright's own navigation/
# selector timeouts were observed to NOT reliably fire when X redirects an
# invalid/expired session to its login page (see _x_fetch.py's docstring) --
# subprocess.run's OS-enforced timeout is the one guard that actually holds.
_X_FETCH_TIMEOUT_S = 45


# ---- primary source: X ------------------------------------------------


def _fetch_from_x():
    """Returns candidate post texts from X, or raises on any failure (missing
    session, hung/timed-out fetch, zero posts) so the caller falls back to
    free sources. Runs the actual Playwright work in a subprocess so a stuck
    browser can be hard-killed rather than hanging the whole daily run."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "skills.concepts._x_fetch"],
            capture_output=True,
            text=True,
            timeout=_X_FETCH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"X fetch timed out after {_X_FETCH_TIMEOUT_S}s (session likely invalid/stuck)"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"X fetch subprocess failed: {result.stderr.strip()[:300]}")

    candidates = json.loads(result.stdout)
    if not candidates:
        raise RuntimeError("no posts extracted from X (session likely invalid or expired)")

    return candidates[:CANDIDATE_LIMIT]


# ---- fallback sources: arXiv, Hacker News, GitHub, HuggingFace --------


def _fetch_arxiv(category, limit):
    raw = net.get(f"https://export.arxiv.org/rss/{category}")
    root = ET.fromstring(raw)
    items = root.findall(".//item")[:limit]
    out = []
    for it in items:
        title = (it.findtext("title") or "").strip()
        desc = (it.findtext("description") or "").strip()
        if title:
            out.append(f"{title} — {desc[:200]}" if desc else title)
    return out


def _fetch_hn(limit):
    ids = net.get_json("https://hacker-news.firebaseio.com/v0/topstories.json")[:limit]
    out = []
    for i in ids:
        item = net.get_json(f"https://hacker-news.firebaseio.com/v0/item/{i}.json")
        if item and item.get("title"):
            out.append(item["title"])
    return out


def _fetch_github_trending(limit):
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    data = net.get_json(
        "https://api.github.com/search/repositories"
        f"?q=created:>{since}&sort=stars&order=desc&per_page={limit}"
    )
    return [
        f"{r['full_name']}: {r.get('description') or ''}".strip(": ")
        for r in data.get("items", [])
    ]


def _fetch_hf_trending(limit):
    data = net.get_json(f"https://huggingface.co/api/models?sort=likes&direction=-1&limit={limit}")
    return [f"{m.get('id', '?')} (likes: {m.get('likes', 0)})" for m in data]


def _fetch_fallback():
    candidates = []
    sources = [
        ("arXiv cs.AI", lambda: _fetch_arxiv("cs.AI", _FALLBACK_PER_SOURCE)),
        ("arXiv cs.SE", lambda: _fetch_arxiv("cs.SE", _FALLBACK_PER_SOURCE)),
        ("Hacker News", lambda: _fetch_hn(_FALLBACK_PER_SOURCE)),
        ("GitHub Trending", lambda: _fetch_github_trending(_FALLBACK_PER_SOURCE)),
        ("HuggingFace Trending", lambda: _fetch_hf_trending(_FALLBACK_PER_SOURCE)),
    ]
    for label, fetch in sources:
        try:
            items = fetch()
            envelope.progress(f"concepts: fallback source {label} returned {len(items)} item(s)")
            candidates.extend(items)
        except Exception as exc:  # noqa: BLE001 - one bad source shouldn't sink the run
            envelope.progress(f"concepts: fallback source {label} failed - {exc}")
    return candidates[:CANDIDATE_LIMIT]


# ---- dedup against delivered_items ------------------------------------


def _hash(text):
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _filter_undelivered(conn, candidates):
    hashes = {c: _hash(c) for c in candidates}
    existing = set()
    if hashes:
        placeholders = ",".join("?" for _ in hashes)
        rows = conn.execute(
            "SELECT item_hash FROM delivered_items "
            f"WHERE skill = 'concepts' AND item_hash IN ({placeholders})",
            list(hashes.values()),
        ).fetchall()
        existing = {row["item_hash"] for row in rows}
    undelivered = [c for c in candidates if hashes[c] not in existing]
    return undelivered, hashes


def _mark_delivered(conn, texts, hashes):
    for t in texts:
        conn.execute(
            "INSERT OR IGNORE INTO delivered_items (item_hash, skill) VALUES (?, 'concepts')",
            (hashes[t],),
        )
    conn.commit()


# ---- Claude selection ---------------------------------------------------


def _build_prompt(pool):
    payload = json.dumps(
        [{"index": i, "text": t[:300]} for i, t in enumerate(pool)], ensure_ascii=False
    )
    return f"""Dom wants his daily "concepts" digest. Below is a JSON list of \
candidate posts/items (index, text) gathered from his feeds/free sources: {payload}

Pick exactly 2 AI concepts and 2 software-engineering concepts worth a \
deep-dive today, using these candidates as inspiration for what's currently \
relevant (you don't need to quote them directly). For each pick, reference \
the index of the single candidate that most inspired it, and write:
- name: a short concept name
- what_it_is: 1-2 sentences
- why_it_matters: 1-2 sentences
- resource: one real paper or repo to read (name + URL if you know one)
- exercise: a concrete, hands-on 30-minute exercise

Respond with ONLY strict JSON (no markdown fences, no commentary) matching \
exactly this schema:
{{"concepts": [{{"index": number, "category": "ai"|"swe", "name": "string", \
"what_it_is": "string", "why_it_matters": "string", "resource": "string", \
"exercise": "string"}}]}}"""


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def main():
    conn = store.connect()

    source = "x"
    try:
        candidates = _fetch_from_x()
        envelope.progress(f"concepts: fetched {len(candidates)} candidate(s) from X")
    except Exception as exc:  # noqa: BLE001 - any X failure degrades to fallback
        envelope.progress(f"concepts: X fetch failed ({exc}), falling back to free sources")
        source = "fallback"
        candidates = _fetch_fallback()

    if not candidates:
        conn.close()
        envelope.emit(
            "concepts",
            ok=False,
            summary="No candidate posts/items found from X or fallback sources",
            sections=[],
            error="empty candidate set",
        )
        return

    undelivered, hashes = _filter_undelivered(conn, candidates)
    envelope.progress(f"concepts: {len(undelivered)}/{len(candidates)} candidate(s) not yet delivered")
    # If every candidate has already been delivered before (small fallback
    # pools can repeat day to day), re-pick from the full set rather than
    # stall the run entirely.
    pool = undelivered or candidates

    envelope.progress(f"concepts: asking Claude to pick from {len(pool)} candidate(s)")
    try:
        response = claude.ask(_build_prompt(pool), use_cache=False)
        picks = _extract_json(response["result"]).get("concepts", [])
        if not isinstance(picks, list):
            raise ValueError("Claude response 'concepts' was not a list")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        conn.close()
        envelope.emit(
            "concepts",
            ok=False,
            summary="Failed to pick today's concepts",
            sections=[],
            error=str(exc),
        )
        return

    ai_picks = [p for p in picks if p.get("category") == "ai"][:2]
    swe_picks = [p for p in picks if p.get("category") == "swe"][:2]
    final = ai_picks + swe_picks

    delivered_texts = [pool[p["index"]] for p in final if 0 <= p.get("index", -1) < len(pool)]
    _mark_delivered(conn, delivered_texts, hashes)
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
                {"label": "Source", "value": "X" if source == "x" else "Fallback feeds"},
                {"label": "Candidates considered", "value": len(pool)},
                {"label": "AI picks", "value": len(ai_picks)},
                {"label": "SWE picks", "value": len(swe_picks)},
            ],
        }
    )

    summary = f"{len(ai_picks)} AI + {len(swe_picks)} software-engineering concept(s)"
    if source == "fallback":
        summary += " (fallback sources — X session unavailable)"

    envelope.emit(
        "concepts",
        ok=len(final) > 0,
        summary=summary,
        sections=sections,
        error=None if final else "Claude returned no valid picks",
    )


if __name__ == "__main__":
    main()
