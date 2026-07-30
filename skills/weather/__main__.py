"""weather — daily NOAA forecast for Dom's home coordinate + a Claude framing pass.

Flow (0-1 Claude calls, 2 NOAA calls, per plan.md's token-discipline table):
  1. Load Dom's home coordinate from cwork's shared config.json (same file
     and same shape skills/places reads).
  2. GET api.weather.gov/points/{lat},{lon} to resolve the forecast URL for
     that gridpoint (NOAA's API is keyless but requires resolving a gridpoint
     first — it has no direct lat/lon forecast endpoint).
  3. GET that forecast URL for the period list (today, tonight, the next
     few days).
  4. Claude call: given today's and tonight's forecast text only (never raw
     HTML, never the full multi-day payload), write a short "plan your day
     around this" note.
  5. Emit the envelope: a stat section (today's high/low/conditions) + a
     table of the next few periods + Claude's framing note.

Run: python -m skills.weather   (no CLI inputs, scheduled skill)
"""

import json
from pathlib import Path

from .._lib import claude, envelope, net

SKILL_ID = "weather"

# How many forecast periods (each ~12h: "Today", "Tonight", "Wednesday", ...)
# to show beyond the current one.
PERIODS_SHOWN = 5

_HEADERS = {"User-Agent": "cwork/0.1 (personal automation, dominictedrow@gmail.com)"}


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


def _forecast_url(lat, lon):
    points = net.get_json(f"https://api.weather.gov/points/{lat},{lon}", headers=_HEADERS)
    return points["properties"]["forecast"]


def _fetch_periods(forecast_url):
    data = net.get_json(forecast_url, headers=_HEADERS)
    return data["properties"]["periods"]


def _framing_prompt(current, upcoming):
    lines = [
        f"{current['name']}: {current.get('detailedForecast', current.get('shortForecast', ''))}"
    ]
    for p in upcoming:
        lines.append(f"{p['name']}: {p.get('shortForecast', '')}")
    forecast_text = "\n".join(lines)
    return (
        "Here is today's weather forecast for Dom's home area:\n\n"
        f"{forecast_text}\n\n"
        "Write a short note (2-3 sentences) on how to plan the day around "
        "this: what to wear/bring, and whether it's a good day for outdoor "
        "plans. Respond with ONLY strict JSON, no markdown fences, no "
        'commentary: {"note": "..."}'
    )


def _extract_json(raw_result):
    text = raw_result.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def run():
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
                "Set this up in Settings before running weather."
            ),
        )
    lat, lon, label = home

    envelope.progress(f"weather: resolving NOAA gridpoint for {label} ({lat},{lon})")
    forecast_url = _forecast_url(lat, lon)

    envelope.progress("weather: fetching forecast periods")
    periods = _fetch_periods(forecast_url)
    if not periods:
        return envelope.build(
            SKILL_ID,
            ok=False,
            summary="NOAA returned no forecast periods",
            sections=[],
            error="api.weather.gov's forecast response had an empty periods list.",
        )

    current = periods[0]
    upcoming = periods[1 : 1 + PERIODS_SHOWN]

    note = ""
    try:
        envelope.progress("weather: asking Claude to frame the day")
        response = claude.ask(_framing_prompt(current, upcoming))
        note = str(_extract_json(response["result"]).get("note", "")).strip()
    except Exception as exc:  # noqa: BLE001 - forecast data still stands without the framing note
        envelope.progress(f"weather: framing note failed - {exc}")

    rows = [[p["name"], p.get("shortForecast", ""), f"{p.get('temperature', '?')}°{p.get('temperatureUnit', '')}"] for p in upcoming]

    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": current["name"], "value": f"{current.get('temperature', '?')}°{current.get('temperatureUnit', '')}"},
                {"label": "Conditions", "value": current.get("shortForecast", "")},
                {"label": "Wind", "value": f"{current.get('windSpeed', '?')} {current.get('windDirection', '')}".strip()},
            ],
        },
        {
            "type": "table",
            "title": "Coming up",
            "columns": ["Period", "Conditions", "Temp"],
            "rows": rows,
        },
    ]
    if note:
        sections.append({"type": "text", "title": "Plan your day", "body": note})

    summary = f"{current['name']} near {label}: {current.get('shortForecast', '')}, {current.get('temperature', '?')}°{current.get('temperatureUnit', '')}"
    envelope.progress("weather: done")
    return envelope.build(SKILL_ID, ok=True, summary=summary, sections=sections)


def main():
    try:
        env = run()
    except Exception as e:  # noqa: BLE001 - skill must always emit a valid envelope, never crash bare
        env = envelope.build(
            SKILL_ID, ok=False, summary="weather failed", sections=[], error=f"{type(e).__name__}: {e}"
        )

    envelope.validate(env)
    print(json.dumps(env))


if __name__ == "__main__":
    main()
