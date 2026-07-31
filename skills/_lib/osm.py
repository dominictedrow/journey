"""Shared Overpass QL query + POI-ranking helpers, used by `places` and `trips`.

Both skills follow the same shape: Claude turns a free-text query into OSM
tag selectors, this module queries Overpass around an origin coordinate and
distance-sorts the results, then the caller hands a trimmed candidate list
back to Claude for ranking. Only the prompt wording and radius differ
between the two skills.
"""

import math
import urllib.parse

from . import net


def build_overpass_ql(selectors, lat, lon, radius_m):
    clauses = []
    for sel in selectors:
        key, value = sel["key"], sel["value"]
        clauses.append(f'node["{key}"="{value}"](around:{radius_m},{lat},{lon});')
        clauses.append(f'way["{key}"="{value}"](around:{radius_m},{lat},{lon});')
    body = "".join(clauses)
    return f"[out:json][timeout:25];({body});out center 60;"


def query_overpass(selectors, lat, lon, radius_m):
    ql = build_overpass_ql(selectors, lat, lon, radius_m)
    url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(ql)
    data = net.get_json(url)
    return data.get("elements", [])


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def elements_to_pois(elements, origin_lat, origin_lon, soft_tags, candidate_limit):
    soft_keys = {t["key"] for t in soft_tags}
    pois = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        distance_m = haversine_m(origin_lat, origin_lon, lat, lon)
        matched_soft = {k: tags[k] for k in soft_keys if k in tags}
        pois.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "distance_m": distance_m,
                "amenity": (
                    tags.get("amenity")
                    or tags.get("shop")
                    or tags.get("leisure")
                    or tags.get("tourism")
                    or tags.get("natural")
                    or "place"
                ),
                "soft_tags": matched_soft,
            }
        )
    pois.sort(key=lambda p: p["distance_m"])
    return pois[:candidate_limit]


def format_distance(distance_m):
    if distance_m >= 1000:
        return f"{distance_m / 1000:.1f} km"
    return f"{distance_m:.0f} m"


def map_link(lat, lon):
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}"
