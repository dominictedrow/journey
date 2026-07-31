"""trips — natural-language "weekend getaway" query -> Overpass destinations -> Claude-ranked shortlist.

Run: python -m skills.trips --input '{"query":"quiet beach town","hours":"3"}'

Same shape as `places` (see skills/_lib/osm.py, shared by both): 2 Claude
calls, 1 Overpass call. What differs:
  - Framing is "worth a weekend trip", not "near me" — the tag-selector and
    ranking prompts bias toward destination-worthy tags (tourism, natural,
    leisure) instead of everyday amenities.
  - Radius is driving-time-derived from an "hours away" input (default
    _DEFAULT_HOURS), not a fixed walking/short-drive radius.

Flow:
  1. Load Dom's home coordinate from cwork's shared config.json.
  2. Claude call #1: turn the free-text query into OSM tag selectors for
     trip-worthy destinations.
  3. Query Overpass for POIs within driving-time radius of home.
  4. Haversine-sort candidates by distance, keep the closest ~30.
  5. Claude call #2: rank/shortlist the candidates with a one-line rationale each.
  6. Emit the envelope: a stat section + a table of picks with map links.
"""

import argparse
import json
from pathlib import Path

from .._lib import envelope, osm, claude

SKILL_ID = "trips"

# Rough average driving speed used to turn an "hours away" input into an
# Overpass search radius. Deliberately conservative (mixed highway/surface
# roads) since this is just a candidate-pool radius, not a route estimate.
_AVG_SPEED_KMH = 70

_DEFAULT_HOURS = 3
_MIN_HOURS = 1
_MAX_HOURS = 8

CANDIDATE_LIMIT = 30
SHORTLIST_LIMIT = 10


def _config_path():
    """Mirrors skills/places' _config_path() — same shared config.json."""
    return Path.home() / "Library" / "Application Support" / "cwork" / "config.json"


def _load_home_coord():
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


def _parse_hours(raw):
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_HOURS
    return max(_MIN_HOURS, min(_MAX_HOURS, hours))


def _radius_m(hours):
    return int(hours * _AVG_SPEED_KMH * 1000)


def _extract_json(claude_response):
    text = claude_response.get("result", "")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _tags_prompt(query, hours):
    return (
        "Convert this weekend-getaway search query into OpenStreetMap tag "
        "selectors for trip-worthy destinations (not everyday errands).\n"
        f'Query: "{query}" (looking within about {hours:g} hours\' drive)\n\n'
        "Return ONLY strict JSON, no prose, matching this schema:\n"
        '{"selectors": [{"key": "tourism", "value": "attraction"}], '
        '"soft_tags": [{"key": "natural", "value": "beach"}]}\n\n'
        "Rules:\n"
        "- selectors: 1-4 real OSM primary tag key/value pairs that identify "
        "the kind of destination (e.g. tourism=attraction, "
        "tourism=viewpoint, natural=beach, natural=peak, "
        "leisure=nature_reserve, tourism=camp_site, tourism=museum). "
        "Prefer tags that mark a place worth traveling to, not routine "
        "amenities like cafes or shops. These will be OR-ed together as the "
        "Overpass query filter, so keep them broad enough to return "
        "results.\n"
        "- soft_tags: 0-4 real OSM tags that are nice-to-have signals for "
        "this query but must NOT be required in the query — used only to "
        "help rank results afterward.\n"
    )


def _get_tag_selectors(query, hours):
    response = claude.ask(_tags_prompt(query, hours))
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
    return selectors, soft_tags


def _rank_prompt(query, pois, hours):
    lines = [
        f'Query: "{query}" (weekend getaway within about {hours:g} hours\' drive)',
        "Candidates (index. name — type — distance — tags):",
    ]
    for i, p in enumerate(pois):
        tag_str = ", ".join(f"{k}={v}" for k, v in p["soft_tags"].items()) or "none"
        lines.append(
            f'{i}. {p["name"]} — {p["amenity"]} — {osm.format_distance(p["distance_m"])} — {tag_str}'
        )
    lines.append(
        "\nRank the best weekend-getaway matches for the query. Return ONLY "
        'strict JSON: {"picks": [{"index": 0, "rationale": "one short '
        'sentence"}]}, ordered best first, at most '
        f"{SHORTLIST_LIMIT} picks, using only indices from the list above."
    )
    return "\n".join(lines)


def _rank_pois(query, pois, hours):
    response = claude.ask(_rank_prompt(query, pois, hours))
    parsed = _extract_json(response)
    picks = []
    for pick in parsed.get("picks", [])[:SHORTLIST_LIMIT]:
        idx = pick.get("index")
        if isinstance(idx, int) and 0 <= idx < len(pois):
            picks.append({**pois[idx], "rationale": str(pick.get("rationale", "")).strip()})
    return picks


def run(query, hours):
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
                "Set this up in Settings before running trips."
            ),
        )
    lat, lon, label = home
    radius_m = _radius_m(hours)

    envelope.progress(f"trips: resolving tags for query {query!r} ({hours:g}h -> {radius_m}m radius)")
    selectors, soft_tags = _get_tag_selectors(query, hours)
    if not selectors:
        return envelope.build(
            SKILL_ID,
            ok=False,
            summary="Could not map query to OSM tags",
            sections=[],
            error="Claude did not return any usable OSM tag selectors for this query.",
        )

    envelope.progress(f"trips: querying Overpass around {label} ({radius_m}m radius)")
    elements = osm.query_overpass(selectors, lat, lon, radius_m)

    envelope.progress(f"trips: {len(elements)} raw elements returned, filtering + ranking by distance")
    pois = osm.elements_to_pois(elements, lat, lon, soft_tags, CANDIDATE_LIMIT)
    if not pois:
        return envelope.build(
            SKILL_ID,
            ok=True,
            summary=f'No named destinations found within {hours:g}h of {label} for "{query}"',
            sections=[
                {
                    "type": "stat",
                    "title": "At a glance",
                    "stats": [{"label": "Candidates found", "value": 0}],
                }
            ],
        )

    envelope.progress(f"trips: asking Claude to shortlist {len(pois)} candidates")
    picks = _rank_pois(query, pois, hours)
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
                {"label": "Within", "value": f"{hours:g}h of {label}"},
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
    summary = f'"{query}" within {hours:g}h of {label}: top pick is {top} ({len(picks)} of {len(pois)} candidates shown)'
    envelope.progress("trips: done")
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
            error='Input must include a non-empty "query" string, e.g. {"query": "quiet beach town"}.',
        )
        envelope.validate(env)
        print(json.dumps(env))
        return

    hours = _parse_hours(payload.get("hours"))

    try:
        env = run(query, hours)
    except Exception as e:  # noqa: BLE001 - skill must always emit a valid envelope, never crash bare
        env = envelope.build(
            SKILL_ID, ok=False, summary="trips failed", sections=[], error=f"{type(e).__name__}: {e}"
        )

    envelope.validate(env)
    print(json.dumps(env))


if __name__ == "__main__":
    main()
