"""One-off (2026-11-09): port the Mac calibration (content 1470x797) to Steam (1920x1080).

The game lays out by height and centres horizontally, so a Mac normalized point maps as
x' = 0.5 + K (x - 0.5), y' = y, with K = Mac aspect / Steam aspect; x lengths scale by K.
Checked against direct Steam measurements (shop columns, panel buttons, gear, HUD: within ~3 px).

Reads ui.mac.json / templates.mac.json / maps/*.mac.json, writes ui.json / templates.json / maps/*.json.
Sections measured directly on Steam (ui.json 'hud', 'shop') are kept from the current ui.json.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "bloons/data"
K = (1470 / 797) / (1920 / 1080)


def X(x):
    return round(0.5 + K * (x - 0.5), 5)


def L(x):
    return round(K * x, 5)


def pt(p):
    return [X(p[0]), p[1]]


def box(b):
    return [X(b[0]), b[1], X(b[2]), b[3]]


def geom(v):
    """A point, a box, or a list of them."""
    if isinstance(v, list) and v and isinstance(v[0], list):
        return [geom(e) for e in v]
    if isinstance(v, list) and len(v) == 2:
        return pt(v)
    if isinstance(v, list) and len(v) == 4:
        return box(v)
    return v


def port_ui(mac, steam):
    u = json.loads(json.dumps(mac))
    for sec in ("buttons", "pause", "menus"):
        u[sec] = {k: geom(v) for k, v in mac[sec].items()}
    ms = u["map_select"]
    for k in ("next_page", "prev_page", "slots", "category_beginner"):
        ms[k] = geom(ms[k])
    ms["name_half"] = [L(ms["name_half"][0]), ms["name_half"][1]]
    up = u["upgrade_panel"]
    up["dx"] = L(up["dx"])
    for side in ("left", "right"):
        up[side] = {k: geom(v) for k, v in up[side].items()}
    ab = u["ability_bar"]
    ab["chevron"], ab["slot0"] = pt(ab["chevron"]), pt(ab["slot0"])
    ab["slot_pitch"], ab["panel_left_dx"] = L(ab["slot_pitch"]), L(ab["panel_left_dx"])
    ab["icon_half"] = [L(ab["icon_half"][0]), ab["icon_half"][1]]
    ab["ring_radii"] = [L(r) for r in ab["ring_radii"]]
    rc = u["range_circle"]
    for k in ("min_r", "max_r", "margin", "start_r", "min_run"):
        rc[k] = L(rc[k])
    rc["ability_icon_half"] = [L(rc["ability_icon_half"][0]), rc["ability_icon_half"][1]]
    rc["exclude_always"] = geom(rc["exclude_always"])
    for sec in ("_meta", "window", "hud", "shop"):
        u[sec] = steam[sec]
    return u


def port_templates(t):
    t = json.loads(json.dumps(t))
    for spec in t["templates"].values():
        spec["box"] = box(spec["box"])
        if "search" in spec:
            spec["search"] = box(spec["search"])
    for p in t["probes"].values():
        p["box"] = box(p["box"])
    return t


def port_grid(g):
    return {**g, "x0": X(g["x0"]), "dx": L(g["dx"])}


def port_map(m):
    m = json.loads(json.dumps(m))
    for mask in m.get("masks", {}).values():
        mask["grid"] = port_grid(mask["grid"])
    p = m.get("path", {})
    if "polyline" in p:
        p["polyline"] = [pt(q) for q in p["polyline"]]
    if "band" in p:
        p["band"]["grid"] = port_grid(p["band"]["grid"])
    if "scale" in m:
        m["scale"]["unit_norm"][0] = round(m["scale"]["unit_norm"][0] * K, 7)
    for o in m.get("obstacles", []):
        o["xy"] = pt(o["xy"])
    if "classes" in m:  # spots memory
        m["classes"] = {c: [[X(x), y, ok] for x, y, ok in v] for c, v in m["classes"].items()}
        m["_meta"]["cell"] = [L(m["_meta"]["cell"][0]), m["_meta"]["cell"][1]]
    m.setdefault("_meta", {})["ported"] = f"from the Mac calibration by scripts/port_mac_to_steam.py (K={K:.5f})"
    return m


def main():
    steam = json.loads((DATA / "ui.json").read_text())
    (DATA / "ui.json").write_text(json.dumps(port_ui(json.loads((DATA / "ui.mac.json").read_text()), steam), indent=1))
    tm = DATA / "templates.mac.json"
    (DATA / "templates.json").write_text(json.dumps(port_templates(json.loads(tm.read_text())), indent=1))
    for src in sorted((DATA / "maps").glob("*.mac.json")):
        dst = src.with_name(src.name.replace(".mac.json", ".json"))
        dst.write_text(json.dumps(port_map(json.loads(src.read_text())), indent=None if "spots" in src.name else 1))
        print("ported", dst.name)


if __name__ == "__main__":
    main()
