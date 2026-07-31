"""places — natural-language query -> Overpass POIs -> Claude-ranked shortlist.

Run: conda run -n c_work python -m skills.places --input '{"query":"calm study spot"}'

Flow (2 Claude calls, 1 Overpass call, per plan.md's token-discipline table):
  1. Load Dom's home coordinate from cwork's shared config.json.
  2. Claude call #1: turn the free-text query into OSM tag selectors.
  3. Query Overpass for POIs near home matching those selectors.
  4. Haversine-sort candidates by distance, keep the closest ~30.
  5. Claude call #2: rank/shortlist the candidates with a one-line rationale each.
  6. Emit the envelope: a stat section + a table of picks with map links.
"""

import argparse
import json
from pathlib import Path

from .._lib import envelope, osm, claude

SKILL_ID = "places"

# Overpass search radius around the origin coordinate. A few km covers a
# "near me" query without the candidate pool getting too large to rank.
DEFAULT_RADIUS_M = 5000

# How many nearest candidates (post-distance-sort) to hand to Claude call #2.
CANDIDATE_LIMIT = 30

# How many of Claude's ranked picks to actually render.
SHORTLIST_LIMIT = 10


def _config_path():
    """Mirrors _lib/store.py's db_path() — same base dir, sibling file."""
    return Path.home() / "Library" / "Application Support" / "cwork" / "config.json"


def _load_home_coord():
    """Read the home coordinate from cwork's shared config.json.

    Deviation note: plan.md allows either (a) failing loudly if config is
    missing/malformed, or (b) proceeding without a distance filter if the
    query names a specific area. We take (a) here — this skill has no
    geocoding capability (no geocoding host is on _lib/net.py's allowlist,
    and we're not permitted to touch net.py), so there is no way to resolve
    a named area to coordinates anyway. Without home coordinates there is
    no origin to query Overpass around at all, so failing loudly with a
    clear message is more honest than silently guessing.
    """
    path = _config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        home = data["home"]
        lat = float(home["lat"])
        lon = float(home["lon"])
        label = home.get("label", "home")
        return lat, lon, label
    except (FileNotFoundError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _extract_json(claude_response):
    """Pull the JSON payload Claude was asked to produce out of the CLI envelope.

    `claude.ask()` returns the raw `claude -p --output-format json` wrapper
    (fields like result/usage/etc.) — the actual answer text lives in
    `result`, which itself may be fenced in ```json ... ``` even when asked
    for strict JSON, so strip that before parsing.
    """
    text = claude_response.get("result", "")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _tags_prompt(query):
    return (
        "Convert this place-search query into OpenStreetMap tag selectors.\n"
        f'Query: "{query}"\n\n'
        "Return ONLY strict JSON, no prose, matching this schema:\n"
        '{"selectors": [{"key": "amenity", "value": "cafe"}], '
        '"soft_tags": [{"key": "internet_access", "value": "yes"}], '
        '"quiet_leaning": true}\n\n'
        "Rules:\n"
        "- selectors: 1-4 real OSM primary tag key/value pairs that identify the "
        "kind of place (e.g. amenity=cafe, amenity=library, leisure=park, "
        "shop=books). These will be OR-ed together as the Overpass query filter, "
        "so keep them broad enough to return results.\n"
        "- soft_tags: 0-4 real OSM tags that are nice-to-have signals for this "
        "query (e.g. internet_access=yes, outdoor_seating=yes) but must NOT be "
        "required in the query — they're used later only to help rank results, "
        "since requiring all of them at once often returns zero matches.\n"
        "- quiet_leaning: true if the query implies a preference for calm/quiet "
        "places, else false.\n"
    )


def _get_tag_selectors(query):
    response = claude.ask(_tags_prompt(query))
    parsed = _extract_json(response)
    selectors = []
    for sel in parsed.get("selectors", [])[:4]:
        key = str(sel.get("key", "")).strip().replace('"', "")
        value = str(sel.get("value", "")).strip().replace('"', "")
        if key and value:
            selectors.append({"key": key, "value": value})
    soft_tags = []
    for sel in parsed.get("soft_tags", [])[:4]:
        key = str(sel.get("key", "")).strip().replace('"', "")
        value = str(sel.get("value", "")).strip().replace('"', "")
        if key and value:
            soft_tags.append({"key": key, "value": value})
    quiet_leaning = bool(parsed.get("quiet_leaning", False))
    return selectors, soft_tags, quiet_leaning


def _rank_prompt(query, pois, quiet_leaning):
    lines = [
        f'Query: "{query}" (quiet-leaning preference: {quiet_leaning})',
        "Candidates (index. name — type — distance — tags):",
    ]
    for i, p in enumerate(pois):
        tag_str = ", ".join(f"{k}={v}" for k, v in p["soft_tags"].items()) or "none"
        lines.append(
            f'{i}. {p["name"]} — {p["amenity"]} — {osm.format_distance(p["distance_m"])} — {tag_str}'
        )
    lines.append(
        "\nRank the best matches for the query. Return ONLY strict JSON: "
        '{"picks": [{"index": 0, "rationale": "one short sentence"}]}, '
        f"ordered best first, at most {SHORTLIST_LIMIT} picks, using only indices from the list above."
    )
    return "\n".join(lines)


def _rank_pois(query, pois, quiet_leaning):
    response = claude.ask(_rank_prompt(query, pois, quiet_leaning))
    parsed = _extract_json(response)
    picks = []
    for pick in parsed.get("picks", [])[:SHORTLIST_LIMIT]:
        idx = pick.get("index")
        if isinstance(idx, int) and 0 <= idx < len(pois):
            picks.append({**pois[idx], "rationale": str(pick.get("rationale", "")).strip()})
    return picks


def run(query):
    home = _load_home_coord()
    if home is None:
        return envelope.build(
            SKILL_ID,
            ok=False,
            summary="Home coordinate isn't configured",
            sections=[],
            error=(
                f"Could not read a valid home coordinate from {_config_path()}. "
                'Expected {"home": {"lat": ..., "lon": ..., "label": ...}}. '
                "Set this up before running places."
            ),
        )
    lat, lon, label = home

    envelope.progress(f"places: resolving tags for query {query!r}")
    selectors, soft_tags, quiet_leaning = _get_tag_selectors(query)
    if not selectors:
        return envelope.build(
            SKILL_ID,
            ok=False,
            summary="Could not map query to OSM tags",
            sections=[],
            error="Claude did not return any usable OSM tag selectors for this query.",
        )

    envelope.progress(f"places: querying Overpass around {label} ({DEFAULT_RADIUS_M}m radius)")
    elements = osm.query_overpass(selectors, lat, lon, DEFAULT_RADIUS_M)

    envelope.progress(f"places: {len(elements)} raw elements returned, filtering + ranking by distance")
    pois = osm.elements_to_pois(elements, lat, lon, soft_tags, CANDIDATE_LIMIT)
    if not pois:
        return envelope.build(
            SKILL_ID,
            ok=True,
            summary=f'No named places found near {label} for "{query}"',
            sections=[
                {
                    "type": "stat",
                    "title": "At a glance",
                    "stats": [{"label": "Candidates found", "value": 0}],
                }
            ],
        )

    envelope.progress(f"places: asking Claude to shortlist {len(pois)} candidates")
    picks = _rank_pois(query, pois, quiet_leaning)
    if not picks:
        picks = [{**p, "rationale": "Closest match by distance"} for p in pois[:SHORTLIST_LIMIT]]

    rows = [
        [p["name"], osm.format_distance(p["distance_m"]), p["rationale"], osm.map_link(p["lat"], p["lon"])]
        for p in picks
    ]

    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": "Candidates found", "value": len(pois)},
                {"label": "Shortlisted", "value": len(picks)},
                {"label": "Origin", "value": label},
            ],
        },
        {
            "type": "table",
            "title": "Ranked picks",
            "columns": ["Name", "Distance", "Why", "Map"],
            "rows": rows,
        },
    ]

    top = picks[0]["name"]
    summary = f'"{query}" near {label}: top pick is {top} ({len(picks)} of {len(pois)} candidates shown)'
    envelope.progress("places: done")
    return envelope.build(SKILL_ID, ok=True, summary=summary, sections=sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{}")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as e:
        env = envelope.build(SKILL_ID, ok=False, summary="Invalid --input JSON", sections=[], error=str(e))
        envelope.validate(env)
        print(json.dumps(env))
        return

    query = (payload.get("query") or "").strip()
    if not query:
        env = envelope.build(
            SKILL_ID,
            ok=False,
            summary="No query provided",
            sections=[],
            error='Input must include a non-empty "query" string, e.g. {"query": "calm study spot"}.',
        )
        envelope.validate(env)
        print(json.dumps(env))
        return

    try:
        env = run(query)
    except Exception as e:  # noqa: BLE001 - skill must always emit a valid envelope, never crash bare
        env = envelope.build(
            SKILL_ID, ok=False, summary="places failed", sections=[], error=f"{type(e).__name__}: {e}"
        )

    envelope.validate(env)
    print(json.dumps(env))


if __name__ == "__main__":
    main()
