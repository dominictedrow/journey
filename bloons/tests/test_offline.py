"""Offline tests (no game needed). Run: pytest tests/test_offline.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bloons.coords import box_to_pixels, to_norm, to_screen
from bloons.hud import Hud, HudFilter, parse_hud, parse_lives, parse_money, parse_round
from bloons.rules import (cost, hero_abilities, hero_level_cost, hero_range, hero_xp_for_round, hero_xp_to_reach,
                          tower_abilities, tower_range, tower_shape, upgrade_allowed)


# (base $, Easy price seen in game) - step 2.5, Monkey Meadow Easy sandbox, game 57.4
EASY_PRICES_SEEN = [
    (200, 170), (375, 320), (260, 220), (400, 340), (225, 190), (300, 255), (350, 300), (325, 275),
    (800, 680), (1200, 1020),                                              # shop (+ hero)
    (140, 120), (200, 170), (320, 275), (1800, 1530), (15000, 12750), (100, 85), (190, 160), (90, 75),
    (280, 240), (600, 510), (2000, 1700), (175, 150), (300, 255),          # dart, boomerang
    (225, 190), (550, 470), (2700, 2295), (150, 130), (220, 190),          # tack
    (500, 425), (250, 215), (450, 385), (220, 190), (200, 170),            # engineer, bomb
]


def test_cost():
    for base, seen in EASY_PRICES_SEEN:
        assert cost(base, "easy") == seen, (base, seen)
    assert [cost(200, d) for d in ("easy", "medium", "hard", "impoppable")] == [170, 200, 215, 240]
    assert cost(320, "easy") == 275 and cost(220, "easy") == 190   # x20 bases round up (wiki: down)
    assert cost(250, "easy") == 215 and cost(450, "easy") == 385   # ties round up (wiki: to even)
    assert cost(315, "easy") - 50 == 220                            # Boomerang after MK "Cheap 'Rangs"


# Easy shop prices seen in game (3.1, all 26 towers + hero, Dom's Monkey Knowledge active)
EASY_SHOP_SEEN = {
    "dart_monkey": 170, "boomerang_monkey": 220, "bomb_shooter": 320, "tack_shooter": 220, "ice_monkey": 340,
    "glue_gunner": 190, "desperado": 255, "sniper_monkey": 300, "monkey_sub": 275, "monkey_buccaneer": 340,
    "monkey_ace": 680, "heli_pilot": 1275, "mortar_monkey": 510, "dartling_gunner": 720, "wizard_monkey": 210,
    "super_monkey": 2125, "ninja_monkey": 340, "alchemist": 470, "druid": 340, "mermonkey": 255,
    "skywarden": 175, "banana_farm": 945, "spike_factory": 700, "monkey_village": 1000,
    "engineer_monkey": 300, "beast_handler": 210,
}


def test_shop_prices():
    import json
    from bloons.rules import shop_price, upgrade_price
    root = Path(__file__).resolve().parents[1] / "bloons/data"
    towers = json.loads((root / "towers.json").read_text())["towers"]
    acc = json.loads((root / "account.json").read_text())
    assert set(EASY_SHOP_SEEN) == set(towers)
    for name, seen in EASY_SHOP_SEEN.items():
        assert shop_price(name, towers[name]["cost"], "easy", acc) == seen, (name, seen)
    assert cost(1200, "easy", "tower") == 1020                                    # hero
    assert cost(250, "easy", "tower") == 210 and cost(250, "easy") == 215         # tie: tower even, upgrade up
    # first-only discounts drop off for the 2nd Farm / Spike Factory
    assert shop_price("spike_factory", 1000, "easy", acc, first=False) == 850
    assert shop_price("banana_farm", 1250, "easy", acc, first=False) == 1040      # 1062.5 x 0.98 = 1041.25
    assert shop_price("dart_monkey", 200, "easy") == 170                           # no account
    # MK upgrade discounts (read off the tree, not yet checked in game)
    assert upgrade_price("ninja_monkey", 0, 3, 900, "medium", acc) == 800
    assert upgrade_price("ninja_monkey", 0, 2, 350, "medium", acc) == 350


def test_account_data():
    import json
    acc = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/account.json").read_text())
    assert acc["_meta"]["game_version"]
    mk = acc["monkey_knowledge"]
    assert mk["cheap_rangs"]["active"] and mk["scholarships"]["training_discount"] == 0.1
    assert all("active" in v and v["evidence"] for v in mk.values())


def test_tower_data():
    import json
    towers = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/towers.json").read_text())["towers"]
    assert towers["dart_monkey"]["cost"] == 200
    for t in towers.values():
        assert len(t["upgrades"]) == 3 and all(len(p) == 5 and all(u["cost"] > 0 for u in p) for p in t["upgrades"])


def test_upgrade_allowed():
    assert upgrade_allowed((0, 0, 0), 0)
    assert upgrade_allowed((4, 2, 0), 0)       # -> 5-2-0
    assert not upgrade_allowed((5, 0, 0), 0)   # past tier 5
    assert not upgrade_allowed((2, 2, 0), 2)   # third path
    assert not upgrade_allowed((3, 2, 0), 1)   # second path past 2
    assert upgrade_allowed((2, 2, 0), 0)       # -> 3-2-0


def test_coords():
    bounds, tb = (55.0, 33.0, 1470.0, 830.0), 33   # windowed calibration layout
    assert to_screen(0, 0, bounds, tb) == (55.0, 66.0)          # content starts below the title bar
    assert to_screen(1, 1, bounds, tb) == (1525.0, 863.0)
    x, y = to_screen(0.3, 0.7, bounds, tb)
    assert all(abs(a - b) < 1e-9 for a, b in zip(to_norm(x, y, bounds, tb), (0.3, 0.7)))
    # 2x Retina capture of the same window: title bar is 66 px
    assert box_to_pixels([0, 0, 1, 1], 2940, 1660, 66) == (0, 66, 2940, 1594)
    assert box_to_pixels([0.5, 0.5, 0.75, 1], 2940, 1660, 66) == (1470, 863, 735, 797)


def test_ui_data():
    import json
    ui = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/ui.json").read_text())
    assert ui["_meta"]["game_version"]
    for x0, y0, x1, y1 in ui["hud"].values():
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    def points(v):   # every [x, y] point / [x0, y0, x1, y1] box anywhere in a section
        if isinstance(v, dict):
            return [p for k, x in v.items() if not k.startswith("_") for p in points(x)]
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return [v] if len(v) in (2, 4) else []
        return [p for x in v for p in points(x)] if isinstance(v, list) else []
    for sec in ("buttons", "upgrade_panel", "ability_bar", "pause", "menus"):
        pts = points(ui[sec])
        assert pts, sec
        for p in pts:
            assert all(0 <= c <= 1 for c in p), (sec, p)
            if len(p) == 4:
                assert p[0] < p[2] and p[1] < p[3], (sec, p)


UI_FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/ui"


def test_ui_shop():
    import json
    import cv2
    from bloons.ui import UI, shop_cell
    towers = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/towers.json").read_text())["towers"]
    order = UI["shop"]["order"]
    assert len(order) == len(set(order)) and set(order) == set(towers) | {"hero"}
    # towers without a hotkey must be clickable in one of the two scroll layouts
    assert shop_cell("desperado")[0] == "top"
    assert {shop_cell(n)[0] for n in ("mermonkey", "skywarden")} == {"bottom"}
    for n, t in towers.items():
        if t["hotkey"] is None:
            shop_cell(n)
    lay, (x, y) = shop_cell("dart_monkey")
    assert lay == "top" and abs(x - 0.935) < 1e-9 and abs(y - 0.2077) < 1e-9
    # scrolled-bottom frame: magic rows (8-10) have purple cells, support rows (11-13) orange
    img = cv2.imread(str(UI_FIX / "shop_scrolled_bottom.jpg")).astype(int)
    h, w = img.shape[:2]
    cw, ch = UI["shop"]["cell_size"]
    for n in order[16:]:
        lay, (x, y) = shop_cell(n)
        c = img[int((y - ch / 2) * h):int((y + ch / 2) * h), int((x - cw / 2) * w):int((x + cw / 2) * w)]
        b, g, r = c[..., 0], c[..., 1], c[..., 2]
        purple = ((b > 200) & (r > 130) & (g < 180)).mean()
        orange = ((r > 220) & (b < 170) & (g > 120) & (g < 225)).mean()
        magic = order.index(n) < 22
        assert lay == "bottom" and (purple if magic else orange) > 0.25 and (orange if magic else purple) < 0.1, \
            (n, purple, orange)


def test_ui_panel():
    import cv2
    from bloons.state import read_screen
    from bloons.ui import UI, ability_slot, panel
    from bloons.vision import Templates
    t = Templates()
    dx = UI["upgrade_panel"]["dx"]
    R, L = panel("right"), panel("left")
    assert abs(R["sell"][0] - L["sell"][0] - dx) < 1e-6 and R["path_buttons"][1][1] == L["path_buttons"][1][1]
    frames = [(s, f["sell_side"], img) for s, f, img in _frames() if "sell_side" in f]
    frames += [(n, side, cv2.imread(str(UI_FIX / f"{n}.jpg"))) for n, side in
               (("panel_left_ability_boomerang_040", "left"), ("hero_panel_right_benjamin_l3", "right"))]
    for stem, side, img in frames:
        assert read_screen(img).panel == side, stem
        x, y = t.match(img, "sell_button").center
        sx, sy = panel(side)["sell"]
        assert abs(x - sx) < 0.005 and abs(y - sy) < 0.005, stem
    # ability bar: 2 slots in this frame, nothing at a 3rd slot (plain grass there)
    img = cv2.imread(str(UI_FIX / "ability_bar_two_slots.jpg"))
    h, w = img.shape[:2]
    def dark_frac(i):   # icons have dark outlines; grass has none
        x, y = ability_slot(i)
        patch = img[int((y - 0.03) * h):int((y + 0.02) * h), int((x - 0.02) * w):int((x + 0.02) * w)]
        return float((patch.max(axis=2) < 70).mean())
    assert dark_frac(0) > 0.04 and dark_frac(1) > 0.04 and dark_frac(2) < 0.01, [dark_frac(i) for i in range(3)]
    assert ability_slot(0, "left")[0] > ability_slot(0)[0] + 0.2


def test_fixture_crops():
    import json
    fix = Path(__file__).resolve().parents[1] / "bloons/data/fixtures"
    crops = json.loads((fix / "crops.json").read_text())["crops"]
    assert len(crops) >= 30
    for name, c in crops.items():
        assert (fix / f"{name}.png").exists(), name
        x0, y0, x1, y1 = c["box"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    assert all("truth" in c for n, c in crops.items() if n.startswith("hud_"))


def _hud_truth():
    """(name, field, raw OCR text, truth) for every hud_* fixture crop."""
    import json
    crops = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/fixtures/crops.json").read_text())["crops"]
    return [(n, n.rsplit("_", 1)[1], c["ocr"], c["truth"]) for n, c in crops.items() if n.startswith("hud_")]


PARSERS = {"lives": parse_lives, "money": parse_money, "round": lambda t: list(parse_round(t) or ())}


def test_hud_parse_fixtures():
    """Recorded Vision output of each committed HUD crop parses to its truth value."""
    rows = _hud_truth()
    assert len(rows) >= 18
    for name, field, raw, truth in rows:
        assert PARSERS[field](raw) == truth, (name, raw)


def test_hud_parse_edge_cases():
    # junk from the 'ROUND' label (Latin and Cyrillic look-alikes) is ignored
    assert parse_round("KOMID 1/40") == (1, 40)
    assert parse_round("\u041a\u041e\u041c\u041f\u0412 30/40") == (30, 40)
    # upgrade panel open: HUD shifted, boxes clip it (upgrade_panel_left/right.png)
    assert parse_money("20") is None
    assert parse_round("TINYN") is None
    # nothing read (defeat screen lives box, menus)
    assert parse_lives("") is None and parse_money("") is None and parse_round("") is None
    assert parse_hud({}) == Hud()
    # thousands separators must be well-formed
    assert parse_money("$12,345,678") == 12345678
    assert parse_money("$1,35") is None and parse_money("$13,57") is None
    assert parse_money("$9.999.609") == 9999609 and parse_money("$1.020") == 1020   # seen in game
    assert parse_money("$1.5") is None
    assert parse_money("·9,998,181") == 9998181 and parse_money("·20") is None and parse_money("·1,35") is None
    # digit confusables, only inside numeric-looking tokens
    assert parse_lives("2OO") == 200 and parse_lives("l89") == 189
    assert parse_money("$1,3S7") == 1357
    assert parse_lives("ROUND") is None
    # impossible values
    assert parse_round("41/40") is None and parse_round("0/40") is None
    assert parse_lives("12345") is None
    # two different candidates in one box is ambiguous
    assert parse_lives("200 189") is None and parse_lives("200 200") == 200
    full = parse_hud({"lives": "189", "money": "$1,630", "round": "39/40"})
    assert full == Hud(189, 1630, 39, 40) and full.complete
    assert not Hud(189, None, 39, 40).complete


def test_hud_filter():
    f = HudFilter(confirm=2)
    assert f.update(Hud(200, 650, 1, 40)) == Hud(200, 650, 1, 40)   # first reading taken as is
    assert f.update(Hud(200, 65, 1, 40)).money == 650                # one-off misread held back
    assert f.update(Hud(200, 650, 1, 40)).money == 650
    assert f.update(Hud(200, 480, 1, 40)).money == 650               # real change: needs a 2nd read
    assert f.update(Hud(200, 480, 1, 40)).money == 480
    assert f.update(Hud(None, None, None, None)) == Hud(200, 480, 1, 40)   # unreadable: keep values
    assert f.misses == {"lives": 1, "money": 1, "round": 1, "rounds_total": 1}
    f.update(Hud(200, 480, 2, 40))
    assert f.misses["round"] == 0 and f.current.round == 1
    f.update(Hud(200, 480, 3, 40))                                   # different candidate restarts the count
    assert f.current.round == 1
    assert f.update(Hud(200, 480, 3, 40)).round == 3
    f.reset()
    assert f.current == Hud() and f.update(Hud(200, 650, 1, 40)).round == 1
    assert HudFilter(confirm=1).update(Hud(1, 2, 3, 4)) == Hud(1, 2, 3, 4)


def test_hud_filter_money_income():
    """Live bug (step 1.5): money rising every read must not freeze the filter."""
    f = HudFilter()
    f.update(Hud(200, 356, 4, 40))
    seq = [241, 244, 247, 250, 253, 256]          # spent $115, then pop income each read
    out = [f.update(Hud(200, m, 4, 40)).money for m in seq]
    assert out == [356, 244, 247, 250, 253, 256]  # one read of lag, then tracks
    # a dropped digit is still held back, and accepted only if it repeats
    f = HudFilter()
    f.update(Hud(money=1357))
    assert f.update(Hud(money=357)).money == 1357
    assert f.update(Hud(money=1360)).money == 1357   # disagrees with 357: restart count
    assert f.update(Hud(money=1362)).money == 1362
    assert f.update(Hud(money=357)).money == 1362
    assert f.update(Hud(money=357)).money == 357     # repeated: genuinely spent
    # lives / round still need exact agreement
    f = HudFilter()
    f.update(Hud(lives=200))
    assert f.update(Hud(lives=199)).lives == 200 and f.update(Hud(lives=198)).lives == 200


def test_hud_ocr_live():
    """Mac only: run Vision on the committed crops and parse (skipped elsewhere)."""
    try:
        from bloons.platform_mac import load_png, ocr_image
    except ImportError:
        print("skip test_hud_ocr_live: not on macOS")
        return
    fix = Path(__file__).resolve().parents[1] / "bloons/data/fixtures"
    for name, field, _, truth in _hud_truth():
        raw = ocr_image(load_png(fix / f"{name}.png"))
        assert PARSERS[field](raw) == truth, (name, raw)


FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures"


def _frames():
    import json
    import cv2
    man = json.loads((FIX / "frames.json").read_text())["frames"]
    return [(stem, f, cv2.imread(str(FIX / "frames" / f"{stem}.jpg"))) for stem, f in man.items()]


def test_vision_spec():
    from bloons.vision import SPEC, Templates
    t = Templates()
    for name, s in SPEC["templates"].items():
        x0, y0, x1, y1 = s["box"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, name
        assert 0.5 <= s["threshold"] < 1, name
        assert t.image(name).size > 0
    for p in SPEC["probes"].values():
        assert p["classes"] and p["max_dist"] > 0


def test_vision_templates():
    """Every frame: listed templates match, all others (except 'ignore') don't."""
    from bloons.vision import Templates
    t = Templates()
    frames = _frames()
    assert len(frames) >= 17
    for stem, f, img in frames:
        assert img is not None, stem
        for name, m in t.match_all(img).items():
            if name in f.get("ignore", []):
                continue
            assert m.found == (name in f["present"]), (stem, name, round(m.score, 3))


def test_vision_button_state():
    from bloons.vision import Templates
    t = Templates()
    for stem, f, img in _frames():
        assert t.button_state(img) == f["button"], stem


def test_vision_match_location():
    from bloons.vision import Templates
    t = Templates()
    for stem, f, img in _frames():
        if "sell_side" in f:   # one sell template finds the panel on either side
            x, _ = t.match(img, "sell_button").center
            assert ("left" if x < 0.5 else "right") == f["sell_side"], stem
    stem, f, img = next(r for r in _frames() if r[0] == "victory_monkey_meadow_easy")
    x, y = t.match(img, "victory_next").center   # ~ where to click Next (RESEARCH: 0.50, 0.84)
    assert abs(x - 0.50) < 0.01 and abs(y - 0.84) < 0.01


def test_frame_bgr_mac():
    """Mac only: Quartz -> numpy conversion gives OpenCV's BGR layout (skipped elsewhere)."""
    try:
        from bloons.platform_mac import cgimage_bgr, load_png
    except ImportError:
        print("skip test_frame_bgr_mac: not on macOS")
        return
    import cv2
    import numpy as np
    for name in ("btn_ff_on", "victory_banner"):
        a = cgimage_bgr(load_png(FIX / f"{name}.png"))
        b = cv2.imread(str(FIX / f"{name}.png"))
        assert a.shape == b.shape and np.abs(a.astype(int) - b).max() <= 1, name


def test_read_screen():
    from bloons.state import read_screen
    for stem, f, img in _frames():
        sc = read_screen(img)
        assert sc.kind == f["screen"], (stem, sc.kind)
        assert sc.button == f["button"], stem
        assert sc.panel == f.get("sell_side"), stem


def test_read_screen_bright_button():
    """Live (step 1.5): the FF button glowing brighter than any probe class (hover?) is
    still a game frame, with button None, and the tracker keeps its state."""
    import numpy as np
    from bloons.state import RoundTracker, read_screen
    from bloons.vision import box_px, SPEC
    _, _, orig = next(r for r in _frames() if r[0] == "hud_r05")
    img = orig.copy()
    x0, y0, x1, y1 = box_px(SPEC["templates"]["btn_ff"]["box"], img)
    btn = img[y0:y1, x0:x1].astype(int)
    img[y0:y1, x0:x1] = np.clip(btn + (33, 5, 20), 0, 255).astype(np.uint8)   # (76,236,152) -> (109,241,172)
    sc = read_screen(img)
    assert sc.kind == "game" and sc.button is None
    t = RoundTracker(confirm=1)
    assert t.update(read_screen(orig)).round_active is True
    assert t.update(sc).round_active is True and t.current.fast_forward is True


def _feed(tracker, seq):
    """seq of (kind, button, round) -> list of RoundState."""
    from bloons.hud import Hud
    from bloons.state import Screen
    return [tracker.update(Screen(k, b), Hud(round=r, rounds_total=40 if r else None)) for k, b, r in seq]


def test_round_tracker_manual_start():
    from bloons.state import RoundTracker
    t = RoundTracker(confirm=2)
    out = _feed(t, [("game", "idle", 1), ("game", "idle", 1),
                    ("game", "ff_on", 1), ("game", "ff_on", 1), ("game", "ff_on", 1),
                    ("game", "idle", 1),                     # one-off misread: ignored
                    ("game", "ff_on", 1),
                    ("pause", None, 1), ("game", "ff_on", 1),  # pause keeps the state
                    ("game", "idle", 2), ("game", "idle", 2), ("game", "idle", 2),
                    ("game", "ff_off", 2), ("game", "ff_off", 2)])
    assert [s.round_active for s in out] == [False, False, False, True, True, True, True, True, True,
                                             True, False, False, False, True]
    assert [i for i, s in enumerate(out) if s.round_ended] == [10]   # exactly once, round 1
    assert out[4].fast_forward is True and out[13].fast_forward is False
    assert out[7].screen == "pause" and out[7].round_active is True
    assert all(s.outcome is None for s in out)


def test_round_tracker_auto_start():
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "idle", 1), ("game", "ff_on", 1), ("game", "ff_on", 1),
                    ("game", "ff_on", 1), ("game", "ff_on", 2), ("game", "ff_on", 2),
                    ("game", None, 2), ("game", "ff_on", 3)])
    assert [i for i, s in enumerate(out) if s.round_ended] == [4, 7]
    assert out[-1].round == 3 and out[-1].rounds_total == 40


def test_round_tracker_outcome():
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 40), ("game", "ff_on", 40),
                    ("victory", None, 40),                   # needs a second frame
                    ("victory", None, 40), ("unknown", None, None), ("map_select", None, None)])
    assert [s.outcome for s in out] == [None, None, None, "win", "win", "win"]
    assert out[3].round_active is False and not any(s.round_ended for s in out)
    t.reset()
    out = _feed(t, [("game", "ff_on", 13), ("defeat", None, 13), ("defeat", None, 13),
                    ("restart_confirm", None, None), ("defeat", None, 13)])
    assert [s.outcome for s in out] == [None, None, "loss", "loss", "loss"]
    t.reset()
    assert t.current is None and _feed(t, [("game", "idle", 1)])[0].outcome is None


def test_round_tracker_on_frames():
    """Real frames end to end: idle -> ff_off -> ff_on -> victory."""
    from bloons.hud import Hud
    from bloons.state import RoundTracker, read_screen
    frames = {stem: img for stem, _, img in _frames()}
    t = RoundTracker(confirm=1)
    seq = ["hud_r1_idle_play_button", "hud_r1_running_ff_off", "hud_r1_running_ff_on",
           "pause_menu", "hud_r39", "victory_monkey_meadow_easy"]
    out = [t.update(read_screen(frames[s]), Hud()) for s in seq]
    assert [s.round_active for s in out] == [False, True, True, True, True, False]
    assert [s.fast_forward for s in out] == [False, False, True, True, True, None]
    assert out[-1].outcome == "win"


def _towers():
    import json
    return json.loads((Path(__file__).resolve().parents[1] / "bloons/data/towers.json").read_text())["towers"]


def test_tower_stats_data():
    towers = _towers()
    missing = {k for k, t in towers.items() if t["stats"] is None}
    assert missing == {"banana_farm", "skywarden"}   # no wiki dump: measure in game
    for k, t in towers.items():
        st = t["stats"]
        if st is None:
            continue
        assert st["placement"]["land"] or st["placement"]["water"], k
        fp = st["footprint"]
        assert ("radius" in fp and fp["radius"] > 0) or ("rect" in fp and min(fp["rect"]) > 0), k
        assert st["range"] > 0 and len(st["range_by_tier"]) == 3 and all(len(p) == 5 for p in st["range_by_tier"]), k
        for a in st["abilities"] or []:
            assert 0 <= a["path"] <= 2 and 1 <= a["tier"] <= 5 and a["cooldown"] > 0 and a["name"], (k, a)
    # spot checks against the game
    d = towers["dart_monkey"]["stats"]
    assert (d["range"], d["footprint"], d["placement"]) == (32, {"radius": 6}, {"land": True, "water": False})
    assert towers["monkey_sub"]["stats"]["placement"] == {"land": False, "water": True}
    assert towers["monkey_buccaneer"]["stats"]["placement"] == {"land": False, "water": True}
    assert "rect" in towers["heli_pilot"]["stats"]["footprint"] and "rect" in towers["monkey_ace"]["stats"]["footprint"]
    assert "patrol_points" in towers["heli_pilot"]["stats"]["targeting"]
    assert towers["monkey_village"]["stats"]["footprint"] == {"radius": 11}


def test_tower_range():
    t = _towers()
    dart = t["dart_monkey"]
    assert tower_range(dart, (0, 0, 0)) == (32, True)
    assert tower_range(dart, (0, 0, 2)) == (48, True)      # Enhanced Eyesight
    assert tower_range(dart, (3, 0, 0)) == (36.8, True)    # Spike-o-pult x1.15
    assert tower_range(dart, (3, 0, 2)) == (52.8, False)   # crosspath: estimate
    assert tower_range(t["banana_farm"], (0, 0, 0)) == (None, False)
    assert tower_range(t["monkey_village"], (1, 0, 0)) == (48, True)
    assert tower_range(t["monkey_village"], (0, 1, 0)) == (None, False)   # not on the wiki


def test_tower_abilities_and_shape():
    t = _towers()
    dart, sup = t["dart_monkey"], t["super_monkey"]
    assert [a["name"] for a in tower_abilities(dart, (0, 4, 0))] == ["Super Monkey Fan Club"]
    assert [a["name"] for a in tower_abilities(dart, (2, 5, 0))] == ["Plasma Monkey Fan Club"]
    assert tower_abilities(dart, (2, 0, 3)) == []
    assert tower_abilities(t["banana_farm"], (0, 0, 0)) is None
    heli = [a["name"] for a in tower_abilities(t["heli_pilot"], (0, 5, 0))]
    assert "Support Drop" in heli and len(heli) == 3
    assert tower_shape(sup, (0, 0, 0))["footprint"] == {"radius": 11}
    sun = tower_shape(sup, (4, 0, 2))                        # Sun Temple: bigger, can go on water
    assert sun["footprint"] == {"rect": [37, 40]} and sun["placement"]["water"]
    assert "strong" in tower_shape(t["ice_monkey"], (0, 0, 3))["targeting"]
    assert tower_shape(t["ice_monkey"], (0, 0, 0))["targeting"] == ["close"]


def _heroes():
    import json
    return json.loads((Path(__file__).resolve().parents[1] / "bloons/data/heroes.json").read_text())["heroes"]


def test_hero_data():
    heroes = _heroes()
    assert len(heroes) >= 18
    for k, h in heroes.items():
        assert len(h["level_xp"]) == 19 and all(x > 0 for x in h["level_xp"]), k
        assert len(h["levels"]) == 20 and all(h["levels"]), k
        if h["stats"]:
            assert len(h["stats"]["range_by_level"]) == 20 and h["stats"]["footprint"], k
    assert {k for k, h in heroes.items() if h["stats"]} >= {"quincy", "adora", "gwendolin", "sauda", "striker_jones"}
    assert heroes["quincy"]["level_xp"][:4] == [180, 460, 1000, 1860]       # xp_scale 1
    assert heroes["benjamin"]["level_xp"][:2] == [270, 690]                  # xp_scale 1.5
    assert heroes["benjamin"]["levels"][2].startswith("Biohack")             # level 3


def test_hero_rules():
    h = _heroes()
    q = h["quincy"]
    assert [hero_xp_for_round(r) for r in (1, 20, 21, 50, 51)] == [40, 420, 460, 1620, 1710]
    assert hero_xp_to_reach(q, 1) == 0 and hero_xp_to_reach(q, 3) == 640   # = the module's xpLv3
    assert hero_level_cost(q, 1) == 180 and hero_level_cost(q, 1, 100) == 80 and hero_level_cost(q, 20) is None
    assert hero_level_cost(h["benjamin"], 1, training_discount=0.1) == 243   # seen in game (Scholarships)
    assert hero_abilities(q, 2) == []
    assert hero_abilities(q, 3) == [{"name": "Rapid Shot", "level": 3, "cooldown": 60}]
    late = {a["name"]: a["cooldown"] for a in hero_abilities(q, 20)}
    assert late == {"Rapid Shot": 45, "Storm of Arrows": 55}
    assert hero_range(q, 1) == 50 and hero_range(q, 20) == 54
    assert hero_abilities(h["benjamin"], 10) is None and hero_range(h["benjamin"], 1) is None


def test_rounds_data():
    """rounds.json: RBE / cash / duration were checked against the wiki's own rendered table
    for all 140 rounds when generated (2026-11-04); these pin a few of those values."""
    import json
    r = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/rounds.json").read_text())["rounds"]
    assert [x["round"] for x in r] == list(range(1, 141))
    for x in r:
        assert x["groups"] and x["duration"] > 0 and x["rbe"] > 0, x["round"]
        assert sum(x["counts"].values()) == sum(g["count"] for g in x["groups"]), x["round"]
        # game data has a few slightly negative starts (round 86: -0.15 s)
        assert all(-1 < g["start"] <= g["end"] <= x["duration"] + 1e-6 for g in x["groups"]), x["round"]
        assert x["end_bonus"] == 100 + x["round"]
    assert (r[0]["rbe"], r[0]["pop_cash"], r[0]["counts"]) == (20, 20, {"red": 20})
    assert r[39]["counts"] == {"moab": 1} and r[39]["rbe"] == 616
    assert r[79]["counts"] == {"zomg": 1} and r[62]["rbe"] == 14413
    assert sum(x["rbe"] for x in r[:40]) == 12289                   # wiki's running total at 40
    first = {f: next(x["round"] for x in r if x["has"][f]) for f in ("regrow", "camo", "lead", "moab_class")}
    assert first == {"regrow": 17, "camo": 24, "lead": 28, "moab_class": 40}
    assert r[49]["income_multiplier"] == 1 and r[50]["income_multiplier"] == 0.5
    ddt = [g for x in r for g in x["groups"] if g["type"] == "ddt"]
    assert ddt and all(g["camo"] for g in ddt)                     # DDTs are always camo


def test_placement_tint():
    """Held tower over grass (ok) and the path (blocked), vs the same crop before holding."""
    import cv2
    from bloons.placement import held_tint
    d = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/placement"
    cases = (("dart_grass", "grass", "ok"), ("dart_path", "path", "blocked"),
             ("heli_grass", "heli_grass", "ok"))   # the Heli pad reaches into the ring
    for held_name, spot, want in cases:
        held, base = cv2.imread(str(d / f"held_{held_name}.png")), cv2.imread(str(d / f"base_{spot}.png"))
        assert held_tint(held, base) == want, spot
        assert held_tint(base, base) == "none", spot


def test_placement_grid_mask():
    import numpy as np
    from bloons.placement import Grid, Mask
    g = Grid.over([0.03, 0.0, 0.84, 1.0], 0.01, 0.0184)
    assert (g.nx, g.ny) == (81, 54) and abs(g.x0 - 0.035) < 1e-9
    assert g.cell(*g.centre(10, 20)) == (10, 20) and g.cell(0.0, 0.5) is None
    ok = np.zeros((g.ny, g.nx), bool)
    ok[20, 10] = True
    m = Mask.from_json(Mask(g, ok).to_json())
    assert m.placeable(*g.centre(10, 20)) and not m.placeable(*g.centre(11, 20))


def test_mask_classes():
    import json
    import numpy as np
    from bloons.placement import Grid, Mask, mask_classes, placeable
    from bloons.rules import tower_shape
    towers = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/towers.json").read_text())["towers"]
    cls = {n: mask_classes(n, tower_shape(t, (0, 0, 0)) if t.get("stats") else None) for n, t in towers.items()}
    assert cls["dart_monkey"] == ["land_r6"] and cls["ice_monkey"] == ["land_r6", "water_r6"]
    assert cls["monkey_sub"] == ["water_r7"] and cls["heli_pilot"] == ["land_heli"] and cls["monkey_ace"] == ["land_ace"]
    assert cls["banana_farm"] == ["land_r11"]                      # assumed until measured
    assert mask_classes("super_monkey", tower_shape(towers["super_monkey"], (4, 0, 0))) == \
        ["land_rect37x40", "water_rect37x40"]                      # Sun Temple: no mask yet -> never placeable
    g = Grid.over([0, 0, 1, 1], 0.5, 0.5)
    ok6 = np.array([[True, True], [True, False]])
    ok11 = np.array([[True, False], [False, False]])
    masks = {"land_r6": Mask(g, ok6), "land_r11": Mask(g, ok11)}
    assert placeable(masks, ["land_r6"], 0.75, 0.25)
    assert not placeable(masks, ["land_r8"], 0.75, 0.25) and placeable(masks, ["land_r8"], 0.25, 0.25)   # -> r11
    assert not placeable(masks, ["water_r7"], 0.25, 0.25)          # no water mask
    assert placeable(masks, ["land_r6", "water_r6"], 0.25, 0.75)   # either class


def test_map_masks():
    """Committed map masks: well-formed, and Monkey Meadow's path is blocked, open grass is ok."""
    import json
    from bloons.placement import Mask
    maps = Path(__file__).resolve().parents[1] / "bloons/data/maps"
    files = sorted(f for f in maps.glob("*.json")
                   if not f.name.startswith("._") and "." not in f.stem) if maps.exists() else []   # FAT32 sidecars; <map>.spots / .bg
    for f in files:
        data = json.loads(f.read_text())
        assert data["_meta"]["game_version"], f.name
        for cls, d in data.get("masks", {}).items():      # maps without a sweep have none
            m = Mask.from_json(d)
            assert 0.05 < m.ok.mean() < 0.95, (f.name, cls)
    mm = maps / "monkey_meadow.json"
    if mm.exists():
        m = Mask.from_json(json.loads(mm.read_text())["masks"]["land_r6"])
        assert not m.placeable(0.30, 0.42) and m.placeable(0.36, 0.30)   # path / grass in the loop


def test_beginner_maps():
    """The 4 Phase-3 maps: scale, obstacles list, and a recorded path whose ends sit on the map
    box edge (entry / exit) and that loads as a keep-out."""
    import json
    from bloons.placement import PathKeepout, unit_norm
    maps = Path(__file__).resolve().parents[1] / "bloons/data/maps"
    box = (0.03, 0.0, 0.84, 1.0)
    for name in ("monkey_meadow", "tree_stump", "town_center", "logs"):
        f = maps / f"{name}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        assert data["_meta"]["game_version"] and isinstance(data["obstacles"], list), name
        unit = unit_norm(data)
        assert abs(unit[0] - UNIT[0]) < 1e-6, name                        # window scale, same on every map
        poly = data["path"]["polyline"]
        assert len(poly) >= 10, name
        for x, y in (poly[0], poly[-1]):
            assert min(abs(x - box[0]), abs(x - box[2]), abs(y - box[1]), abs(y - box[3])) < 1e-3, (name, x, y)
        ko = PathKeepout.from_json(data["path"], unit)
        x, y = poly[len(poly) // 2]
        assert ko.blocked(x, y, 6) and ko.distance(x, y) <= 0, name                # on the line


# --- layered placement + guards (docs/PLACEMENT-PLAN.md) -------------------------------------

UNIT = (0.0027041, 0.0049875)        # Monkey Meadow scale: 1 game unit in normalized x / y


def test_spot_memory(tmp_path):
    from bloons.placement import SpotMemory
    mem = SpotMemory(cell=(0.01, 0.02), path=tmp_path / "m.spots.json", meta={"game_version": "57.4"})
    assert mem.lookup("land_r6", 0.5, 0.5) == "unknown"
    mem.record("land_r6", 0.50, 0.50, False)
    assert mem.lookup("land_r6", 0.50, 0.50) == "blocked"
    assert mem.lookup("land_r6", 0.509, 0.50) == "blocked"          # inside the 1-cell disc
    assert mem.lookup("land_r6", 0.50, 0.519) == "blocked"
    assert mem.lookup("land_r6", 0.52, 0.50) == "unknown"            # 2 cells away
    mem.record("land_r6", 0.505, 0.50, True)                          # exact ok reading overrides the disc
    assert mem.lookup("land_r6", 0.505, 0.50) == "ok"
    assert mem.lookup("land_r6", 0.50, 0.50) == "blocked"
    assert not mem.record("land_r6", 0.505, 0.50, True)               # no change
    # land radii nest: blocked r6 -> blocked r11; ok r11 -> ok r6
    assert mem.lookup("land_r11", 0.50, 0.50) == "blocked"
    mem.record("land_r11", 0.80, 0.30, True)
    assert mem.lookup("land_r6", 0.80, 0.30) == "ok" and mem.lookup("land_r6", 0.803, 0.30) == "ok"
    assert mem.lookup("land_r7", 0.30, 0.30) == "unknown" and mem.lookup("water_r6", 0.80, 0.30) == "unknown"
    assert mem.lookup_any(["land_r6", "water_r6"], 0.80, 0.30) == "ok"
    assert mem.lookup_any(["land_r6", "water_r6"], 0.50, 0.50) == "unknown"
    mem.save()
    back = SpotMemory.load(tmp_path / "m.spots.json")
    assert back.meta["game_version"] == "57.4" and back.cell == (0.01, 0.02)
    for x, y in ((0.5, 0.5), (0.509, 0.5), (0.505, 0.5), (0.8, 0.3), (0.3, 0.3)):
        assert back.lookup("land_r6", x, y) == mem.lookup("land_r6", x, y)


def test_spot_memory_from_masks():
    import numpy as np
    from bloons.placement import Grid, Mask, SpotMemory
    g = Grid.over([0, 0, 0.1, 0.1], 0.01, 0.01)
    ok = np.ones((g.ny, g.nx), bool)
    ok[5, 5] = False
    mem = SpotMemory.from_masks({"land_r6": Mask(g, ok)})
    assert mem.lookup("land_r6", *g.centre(5, 5)) == "blocked"
    assert mem.lookup("land_r6", *g.centre(6, 5)) == "ok"            # exact sweep reading next to it
    x, y = g.centre(6, 5)
    assert mem.lookup("land_r6", x - 0.002, y) == "blocked"          # off-grid, within the disc
    assert mem.lookup("land_r6", *g.centre(2, 2)) == "ok"
    assert mem.import_mask("land_r6", Mask(g, ~ok)) == 0              # existing readings are kept


def test_seeded_spots_match_masks():
    import json
    from bloons.placement import Mask, SpotMemory
    maps = Path(__file__).resolve().parents[1] / "bloons/data/maps"
    f = maps / "monkey_meadow.spots.json"
    if not f.exists():
        return
    mem = SpotMemory.load(f)
    assert mem.meta["game_version"]
    m = Mask.from_json(json.loads((maps / "monkey_meadow.json").read_text())["masks"]["land_r6"])
    for i, j in ((0, 0), (10, 20), (30, 22), (60, 40)):
        x, y = m.grid.centre(i, j)
        assert mem.lookup("land_r6", x, y) == ("ok" if m.ok[j, i] else "blocked")


def test_path_keepout():
    from bloons.placement import PathKeepout
    ko = PathKeepout(UNIT, polyline=[(0.1, 0.5), (0.9, 0.5)], half_width=10)
    y_at = lambda units: 0.5 + units * UNIT[1]                        # noqa: E731
    assert ko.blocked(0.5, 0.5, 6)
    assert ko.blocked(0.5, y_at(15.9), 6) and not ko.blocked(0.5, y_at(16.1), 6)
    assert not ko.blocked(0.5, y_at(15), 4)
    assert ko.blocked(0.1 - 15 * UNIT[0], 0.5, 6) and not ko.blocked(0.1 - 17 * UNIT[0], 0.5, 6)   # past the end
    back = PathKeepout.from_json(ko.to_json(), UNIT)
    assert back.blocked(0.5, y_at(15.9), 6) and not back.blocked(0.5, y_at(16.1), 6)


def test_path_keepout_band():
    import numpy as np
    from bloons.placement import Grid, Mask, PathKeepout
    g = Grid.over([0, 0, 1, 1], 0.01, 0.01844)
    band = np.zeros((g.ny, g.nx), bool)
    band[27, :] = True                                                # one row of track cells
    ko = PathKeepout.from_json({"band": Mask(g, band).to_json()}, UNIT)
    _, yc = g.centre(0, 27)
    half = 0.5 * 0.01844 / UNIT[1]                                     # half a cell in units (~1.85)
    assert ko.blocked(0.5, yc, 6)
    assert ko.blocked(0.5, yc + (half + 5.9) * UNIT[1], 6) and not ko.blocked(0.5, yc + (half + 6.2) * UNIT[1], 6)


def _place_ctx(**kw):
    import json
    from bloons.placement import PlaceContext
    towers = json.loads((Path(__file__).resolve().parents[1] / "bloons/data/towers.json").read_text())["towers"]
    return PlaceContext(towers=towers, unit=UNIT, **kw)


def test_can_place_layers():
    from bloons.placement import PathKeepout, Placed, SpotMemory, can_place, record_placement
    mem = SpotMemory(cell=(0.01, 0.01844))
    ko = PathKeepout(UNIT, polyline=[(0.1, 0.5), (0.8, 0.5)], half_width=8)
    ctx = _place_ctx(keepout=ko, memory=mem, placed=[Placed("dart_monkey", (0.3, 0.2), tid=1)])
    assert can_place("dart_monkey", 0.95, 0.2, ctx).reason == "off_map"
    assert can_place("nope", 0.5, 0.2, ctx).reason == "no_footprint:nope"
    r = can_place("dart_monkey", 0.3 + 11 * UNIT[0], 0.2, ctx)                 # r6 + r6 = 12 > 11
    assert not r.ok and r.reason == "overlaps_tower:1"
    assert can_place("dart_monkey", 0.3 + 12.5 * UNIT[0], 0.2, ctx).ok
    assert can_place("dart_monkey", 0.5, 0.5 + 13 * UNIT[1], ctx).reason == "path"
    r = can_place("dart_monkey", 0.5, 0.2, ctx)
    assert r.ok and r.reason == "unknown"
    # the tint says blocked -> remembered; the spot and its neighbours aren't tried again
    assert record_placement(ctx, "dart_monkey", 0.5, 0.2, False, save=False)
    assert can_place("dart_monkey", 0.5, 0.2, ctx).reason == "memory_blocked"
    assert can_place("dart_monkey", 0.505, 0.2, ctx).reason == "memory_blocked"
    assert can_place("super_monkey", 0.5, 0.2, ctx).reason == "memory_blocked"  # r11 >= r6
    assert record_placement(ctx, "dart_monkey", 0.6, 0.2, True, save=False)
    assert can_place("dart_monkey", 0.6, 0.2, ctx).reason == "known_ok"
    # blocked next to our own tower: not saved (temporary)
    assert not record_placement(ctx, "dart_monkey", 0.3 + 13 * UNIT[0], 0.2, False, save=False)
    # Ice (land or water): blocked is stored for both classes, ok is not stored
    assert record_placement(ctx, "ice_monkey", 0.7, 0.1, False, save=False)
    assert mem.lookup("water_r6", 0.7, 0.1) == "blocked"
    assert not record_placement(ctx, "ice_monkey", 0.7, 0.3, True, save=False)
    # heroes: radius 7, land
    assert can_place("quincy", 0.5, 0.2, _place_ctx(heroes={"quincy": {}})).ok


def _guard_state(**kw):
    import json
    from bloons.guards import GuardState
    root = Path(__file__).resolve().parents[1] / "bloons/data"
    towers = json.loads((root / "towers.json").read_text())["towers"]
    heroes = json.loads((root / "heroes.json").read_text())["heroes"]
    acc = json.loads((root / "account.json").read_text())
    return GuardState(money=kw.pop("money", 1000), difficulty="easy", towers=towers, heroes=heroes, account=acc, **kw)


def test_guards_upgrade():
    from bloons.guards import TowerRec, can_upgrade
    reg = {1: TowerRec(1, "dart_monkey", (0.3, 0.3), (0, 0, 0), 170)}
    s = _guard_state(money=119, registry=reg)
    r = can_upgrade(1, 0, s)                                    # Sharp Shots $140 -> $120 on Easy
    assert not r.ok and r.reason == "no_money" and r.cost == 120
    s.money = 120
    assert can_upgrade(1, 0, s).ok and can_upgrade(1, 0, s).cost == 120    # exact money is enough
    assert can_upgrade(9, 0, s).reason == "no_tower"
    s.money = 10 ** 6
    reg[1].tiers = (2, 2, 0)
    assert can_upgrade(1, 2, s).reason == "crosspath"           # third path
    reg[1].tiers = (3, 2, 0)
    assert can_upgrade(1, 1, s).reason == "crosspath"           # second path past 2
    reg[1].tiers = (5, 0, 0)
    assert can_upgrade(1, 0, s).reason == "max_tier"            # no tier 6
    reg[2] = TowerRec(2, "dart_monkey", (0.5, 0.3), (4, 0, 0), 0)
    assert can_upgrade(2, 0, s).reason == "tier5_taken"         # one 5-x-x Dart per game
    reg[3] = TowerRec(3, "tack_shooter", (0.6, 0.3), (4, 0, 0), 0)
    assert can_upgrade(3, 0, s).ok                              # other tower type is fine
    assert can_upgrade(2, 1, s).ok


def test_guards_misc():
    from bloons.guards import Ability, HeroRec, TowerRec, can_buy_hero_level, can_buy_tower, can_place_hero, \
        can_sell, can_start_round, can_use_ability
    s = _guard_state(money=169)
    assert can_buy_tower("dart_monkey", s).reason == "no_money"
    s.money = 170
    assert can_buy_tower("dart_monkey", s).cost == 170 and can_buy_tower("nope", s).reason == "unknown_tower"
    s.locked.add("dart_monkey")
    assert not can_buy_tower("dart_monkey", s).ok
    s.money = 2000
    assert can_buy_tower("spike_factory", s).cost == 700        # first-only MK discount ...
    s.bought.add("spike_factory")
    assert can_buy_tower("spike_factory", s).cost == 850        # ... gone for the second
    assert can_place_hero(s).reason == "unknown_hero"
    s.hero_choice = "benjamin"
    assert can_place_hero(s).ok and can_place_hero(s).cost == 1020
    assert can_buy_hero_level(s).reason == "no_hero"
    s.hero = HeroRec("benjamin", (0.4, 0.4), 1, 0, 1020)
    assert can_place_hero(s).reason == "hero_placed"
    assert can_buy_hero_level(s).cost == 243                     # seen in game (2.5)
    s.money = 242
    assert can_buy_hero_level(s).reason == "no_money"
    s.hero.level = 20
    assert can_buy_hero_level(s).reason == "hero_max_level"
    s.registry[1] = TowerRec(1, "dart_monkey", (0.3, 0.3), (0, 0, 0), 170)
    assert can_sell(1, s).cost == -119 and can_sell(2, s).reason == "no_tower" and can_sell("hero", s).cost == -714
    s.abilities = [Ability(0, "Supply Drop", True), Ability(1, "MOAB Barrage", False),
                   Ability(2, "Overclock", True, between_rounds=False)]
    assert can_use_ability(0, s).ok and can_use_ability(1, s).reason == "cooldown"
    assert can_use_ability(2, s).reason == "round_only" and can_use_ability(5, s).reason == "no_ability"
    s.round_active = True
    assert can_use_ability(2, s).ok and can_start_round(s).reason == "round_active"


def test_available_actions():
    from bloons.guards import Ability, TowerRec, available_actions
    s = _guard_state(money=200, registry={1: TowerRec(1, "dart_monkey", (0.3, 0.3), (0, 0, 0), 170)},
                     hero_choice="benjamin", abilities=[Ability(0, "x", False)])
    acts = available_actions(s)
    kinds = {a["kind"] for a in acts if a["action"] == "buy_tower"}
    assert kinds == {"dart_monkey", "glue_gunner", "skywarden"}           # $170, $190, $175 <= $200
    assert {"action": "start_round", "cost": 0} in acts
    ups = {(a["tid"], a["path"]): a["cost"] for a in acts if a["action"] == "upgrade"}
    assert ups == {(1, 0): 120, (1, 1): 85, (1, 2): 75}
    assert {"action": "sell", "tid": 1, "cost": -119} in acts
    assert not any(a["action"] in ("place_hero", "use_ability", "buy_hero_level") for a in acts)
    assert all(a["cost"] is None or a["cost"] <= s.money for a in acts)


def test_path_recorder():
    """Synthetic round 1: a square 'bloon' moves along an L-shaped track over a noisy map."""
    import numpy as np
    from bloons.pathmap import PathRecorder, keepout_agreement
    from bloons.placement import Grid, Mask, PathKeepout
    rng = np.random.default_rng(0)
    h, w = 200, 300
    bg = rng.integers(60, 120, (h, w, 3)).astype(np.uint8)
    flicker = (slice(180, 190), slice(10, 20))                     # animated corner: ignored
    empties = []
    for k in range(4):
        e = bg.copy()
        e[flicker] = 255 if k % 2 else 0
        empties.append(e)
    rec = PathRecorder(empties, exclude=[(0.0, 0.0, 0.2, 0.1)], min_area=60)
    track = [(x, 50) for x in range(20, 200, 3)] + [(200, y) for y in range(50, 170, 3)]
    for t, (cx, cy) in enumerate(track):
        f = bg.copy()
        f[cy - 5:cy + 5, cx - 5:cx + 5] = (30, 30, 230)
        lx = 280 - 4 * t % 260                                    # red 'leaf' drifting left: too small
        f[140:145, lx:lx + 5] = (30, 30, 230)
        f[5:15, 5:40] = 255                                       # HUD change: excluded
        f[flicker] = 255 if t % 2 else 0
        rec.feed(f, t * 0.05)
    band = rec.band_px()
    assert band[50, 100] and band[120, 200] and not band[150, 100] and not band[185, 15] and not band[10, 20]
    assert rec.counts[140:145, :190].sum() == 0                    # the leaf never counted (the bloon crosses at x 200)
    pts = rec.polyline(band, bins=40)
    assert 2 <= len(pts) <= 6
    assert pts[0][0] == 0.0 and abs(pts[0][1] - 50 / h) < 0.03                      # entry first, extended to the edge
    assert abs(pts[-1][0] - 200 / w) < 0.03 and abs(pts[-1][1] - 168 / h) < 0.05      # exit last
    assert any(abs(x - 200 / w) < 0.03 and abs(y - 50 / h) < 0.04 for x, y in pts)    # the corner
    g = Grid.over([0, 0, 1, 1], 0.02, 0.03)
    m = rec.band_mask(g, band)
    assert m.placeable(100 / w, 50 / h) and not m.placeable(100 / w, 150 / h)
    ko = PathKeepout(UNIT, band=m)
    rep = keepout_agreement(ko, Mask(g, ~m.ok), 0.1)
    assert rep["blocked_explained"] == 1.0 and rep["false_block_of_ok"] < 0.02



def test_path_helpers():
    import numpy as np
    from bloons.pathmap import extend_to_box, fit_half_width, lead_path
    from bloons.placement import Grid, Mask, PathKeepout
    # the lead bloon leaves at the bottom edge, then the next bloon (far back) would become 'lead'
    lead = [(t * 0.1, 0.1 + 0.01 * t, 0.5) for t in range(20)] + [(2.0 + 0.1 * t, 0.29, 0.5 + 0.03125 * t) for t in range(17)]
    lead += [(4.0, 0.5, 0.3), (4.1, 0.51, 0.3)]
    pts = lead_path(lead)
    assert pts[-1] == (0.29, 1.0) and (0.5, 0.3) not in pts
    lead2 = [(0.0, 0.1, 0.5), (0.1, 0.11, 0.5), (0.2, 0.6, 0.5), (0.3, 0.12, 0.5)]
    assert lead_path(lead2) == [(0.1, 0.5), (0.11, 0.5), (0.12, 0.5)]   # one-frame jump dropped
    ext = extend_to_box([(0.05, 0.4), (0.3, 0.4), (0.3, 0.9)], (0.0, 0.0, 1.0, 1.0), max_ext=0.08)
    assert ext[0] == (0.0, 0.4) and ext[-1] == (0.3, 0.9)               # end: 0.1 to go > max_ext
    assert extend_to_box([(0.05, 0.4), (0.3, 0.4), (0.3, 0.95)], (0, 0, 1, 1))[-1] == (0.3, 1.0)
    # vectorized distances == scalar distance
    ko = PathKeepout(UNIT, polyline=[(0.1, 0.2), (0.6, 0.2), (0.6, 0.8)], half_width=3)
    xs, ys = np.random.default_rng(1).random((2, 50))
    assert np.allclose(ko.distances(xs, ys), [ko.distance(x, y) for x, y in zip(xs, ys)])
    # fit: a mask blocked within 10 units of a line. The nearest ok cell row is d_ok away, so the
    # widest keep-out with no false block is d_ok - 6 (rounded down to the 0.25 step).
    g = Grid.over([0, 0, 1, 1], 0.01, 0.01844)
    line = [(0.0, 0.5), (1.0, 0.5)]
    jj, ii = np.mgrid[0:g.ny, 0:g.nx]
    d = PathKeepout(UNIT, polyline=line, half_width=0).distances(g.x0 + ii * g.dx, g.y0 + jj * g.dy).reshape(g.ny, g.nx)
    hw, rep, _ = fit_half_width(line, UNIT, {"land_r6": Mask(g, d >= 10)}, max_false=0.0)
    d_ok = d[d >= 10].min()
    assert hw == np.floor((d_ok - 6) * 4) / 4 and rep["land_r6"]["false_block"] == 0


if __name__ == "__main__":
    test_fixture_crops()
    test_coords()
    test_ui_data()
    test_cost()
    test_account_data()
    test_upgrade_allowed()
    test_tower_data()
    test_tower_stats_data()
    test_tower_range()
    test_tower_abilities_and_shape()
    test_hero_data()
    test_hero_rules()
    test_rounds_data()
    test_hud_parse_fixtures()
    test_hud_parse_edge_cases()
    test_hud_filter()
    test_hud_filter_money_income()
    test_hud_ocr_live()
    test_vision_spec()
    test_vision_templates()
    test_vision_button_state()
    test_vision_match_location()
    test_frame_bgr_mac()
    test_read_screen()
    test_read_screen_bright_button()
    test_round_tracker_manual_start()
    test_round_tracker_auto_start()
    test_round_tracker_outcome()
    test_round_tracker_on_frames()
    print("ok")


# --- harness skeleton (PLAN 4.1) -------------------------------------------------------------

class FakeGame:
    """Records input; read_hud returns whatever `hud` is set to."""
    def __init__(self, front=True):
        from bloons.hud import Hud
        self.calls, self.front = [], front
        self.hud = Hud(200, 650, 1, 40)

    on_key = on_click = None      # optional hooks: change self.hud when an input lands

    def key(self, k):
        self.calls.append(("key", k))
        if self.on_key:
            self.on_key(self, k)
        return self.front

    def click(self, x, y):
        self.calls.append(("click", round(x, 4), round(y, 4)))
        if self.on_click:
            self.on_click(self, x, y)
        return self.front

    def grab(self):
        import numpy as np

        class F:
            def content_bgr(self):
                return np.zeros((8, 8, 3), np.uint8)
        return F()

    def move(self, x, y):
        self.calls.append(("move", x, y)); return self.front

    def scroll(self, x, y, lines):
        self.calls.append(("scroll", lines)); return self.front

    def read_hud(self, frame=None):
        return self.hud


def test_harness_registry_and_guards():
    from bloons.harness import BloonsHarness
    from bloons.placement import can_place
    g = FakeGame()
    h = BloonsHarness(g, "tree_stump", hero="benjamin", sleep=lambda s: None)
    assert h.money is None and h.available_actions()[0]["action"] == "start_round"
    assert not any(a["action"] == "buy_tower" for a in h.available_actions())   # money unknown -> $0
    h.refresh_hud()
    assert h.money == 650
    acts = h.available_actions()
    dart = next(a for a in acts if a.get("kind") == "dart_monkey")
    assert dart["cost"] == 170                                         # Easy + Dom's MK, as in the shop
    assert not any(a.get("kind") == "super_monkey" for a in acts)       # too expensive
    tid = h.register("dart_monkey", (0.5, 0.5), 170)
    assert tid == 1 and h.register("dart_monkey", (0.6, 0.6), 170) == 2
    assert any(a["action"] == "upgrade" and a["tid"] == 1 for a in h.available_actions())
    h.record_upgrade(1, 0, 120)
    assert h.registry[1].tiers == (1, 0, 0) and h.registry[1].spent == 290
    r = can_place("dart_monkey", 0.502, 0.5, h.place_context())
    assert not r and r.reason == "overlaps_tower:1"
    h.register_hero((0.2, 0.2), 1020)
    assert not can_place("dart_monkey", 0.201, 0.2, h.place_context())
    assert any(a.get("tid") == "hero" for a in h.available_actions())
    h.unregister(2); h.unregister("hero")
    assert list(h.registry) == [1] and h.hero is None
    h.new_episode()
    assert h.registry == {} and h.money is None


def test_harness_input():
    from bloons.harness import BloonsHarness, SCROLL_LINES
    from bloons.ui import shop_cell
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None)
    assert h.key_for("dart_monkey") == "q" and h.key_for("desperado") is None
    assert h.key_for("upgrade", 2) == "/" and h.key_for("hero") == "u" and h.key_for("powers") is None
    assert h.select_shop("dart_monkey") and g.calls == [("key", "q")]
    g.calls.clear()
    assert h.select_shop("mermonkey") and h.select_shop("skywarden") and h.select_shop("desperado")
    x, y = shop_cell("mermonkey")[1]
    assert g.calls[:2] == [("scroll", -SCROLL_LINES), ("click", round(x, 4), round(y, 4))]
    assert g.calls[2][0] == "click"                                    # same layout: no scroll
    assert g.calls[3] == ("scroll", SCROLL_LINES) and g.calls[4][0] == "click"
    assert h.select_shop("bogus").reason == "unknown_tower"
    g.calls.clear()
    assert h.press("powers", xy=(0.877, 0.94)) and g.calls == [("click", 0.877, 0.94)]
    assert h.press("targeting_prev").reason == "no_input"
    g.front = False
    assert h.press("sell").reason == "not_front"


def test_harness_place_and_sell(tmp_path):
    from bloons.harness import BloonsHarness
    from bloons.hud import Hud
    g = FakeGame()
    h = BloonsHarness(g, "tree_stump", sleep=lambda s: None, spots_path=tmp_path / "s.spots.json")
    tints = []
    h.tint = lambda base, held, x, y: tints.pop(0) if tints else "none"

    def spend(game, x, y):            # a click on the map places: money drops by the Dart price
        held_here = len(game.calls) > 1 and game.calls[-2] == ("move", x, y)   # not a select click
        if held_here and game.hud.money >= 170:
            game.hud = Hud(200, game.hud.money - 170, 1, 40)
    g.on_click = spend
    h.refresh_hud()
    tints[:] = ["ok"]
    r = h.place_tower("dart_monkey", 0.10, 0.50)
    h.refresh_hud()
    assert r and r.value == 1 and r.cost == 170 and h.money == 480
    assert h.registry[1].xy == (0.10, 0.50) and "dart_monkey" in h.bought
    assert h.memory.lookup("land_r6", 0.10, 0.50) == "ok"
    assert ("key", "q") in g.calls
    # the tint says blocked: cancel, remember the spot, nothing registered
    g.calls.clear()
    tints[:] = ["blocked"]
    r = h.place_tower("dart_monkey", 0.10, 0.80)
    assert not r and r.reason == "blocked" and list(h.registry) == [1]
    assert ("click", 0.8218, 0.1066) in g.calls and h.memory.lookup("land_r6", 0.10, 0.80) == "blocked"
    calls = len(g.calls)
    assert h.place_tower("dart_monkey", 0.10, 0.80).reason == "memory_blocked" and len(g.calls) == calls   # no input
    x, y = h.map["path"]["polyline"][10]
    assert h.place_tower("dart_monkey", x, y).reason == "path"
    assert h.place_tower("dart_monkey", 0.101, 0.50).reason == "overlaps_tower:1"
    # no hold at all (tint 'none' twice): cancel, nothing remembered
    assert h.place_tower("dart_monkey", 0.70, 0.90).reason == "no_hold"
    assert h.memory.lookup("land_r6", 0.70, 0.90) == "unknown"
    # sell: Backspace, refund confirmed on the HUD
    g.on_key = lambda game, k: k == "backspace" and setattr(game, "hud", Hud(200, game.hud.money + 119, 1, 40))
    r = h.sell(1)
    assert r and r.cost == -119 and h.registry == {} and g.calls[-1] == ("key", "backspace")
    assert h.sell(1).reason == "no_tower"
    g.hud = Hud(200, 20, 1, 40); h.refresh_hud(); h.refresh_hud()
    assert h.place_tower("dart_monkey", 0.10, 0.50).reason == "no_money"


def test_harness_upgrade_and_hero_level(tmp_path):
    from bloons import guards
    from bloons.harness import BloonsHarness
    from bloons.hud import Hud
    from bloons.ui import panel
    g = FakeGame()
    g.hud = Hud(200, 5000, 1, 40)
    h = BloonsHarness(g, "tree_stump", hero="benjamin", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    h.refresh_hud()
    h.read_panel = lambda frame: "right"
    tid = h.register("dart_monkey", (0.2, 0.5), 170)
    price = guards.upgrade_cost(h.registry[tid], 0, h.guard_state())
    g.on_key = lambda game, k: k == "," and setattr(game, "hud", Hud(200, game.hud.money - price, 1, 40))
    r = h.upgrade(tid, 0)
    assert r and r.value == (1, 0, 0) and r.cost == price and h.registry[tid].spent == 170 + price
    assert g.calls[-3:] == [("click", 0.2, 0.5), ("key", ","), ("click", *[round(v, 4) for v in panel("right")["close"]])]
    # the game didn't take it (money unchanged): unconfirmed, tiers unchanged, panel closed
    g.on_key = None
    r = h.upgrade(tid, 0)
    assert not r and r.reason == "unconfirmed" and h.registry[tid].tiers == (1, 0, 0)
    h.registry[tid].tiers = (2, 0, 3)
    assert h.upgrade(tid, 1).reason == "crosspath" and h.upgrade(99, 0).reason == "no_tower"
    # no panel found: refuse
    h.read_panel = lambda frame: None
    h.registry[tid].tiers = (0, 0, 0)
    assert h.upgrade(tid, 0).reason == "no_panel"
    # hero level: the level-up button on the panel's side
    h.read_panel = lambda frame: "left"
    assert h.buy_hero_level().reason == "no_hero"
    h.register_hero((0.6, 0.3), 1020)
    lv = guards.hero_level_price(h.guard_state())
    up = tuple(round(v, 4) for v in panel("left")["hero_level_up"])
    charge = {"$": lv}
    g.on_click = lambda game, x, y: (round(x, 4), round(y, 4)) == up and setattr(
        game, "hud", Hud(200, game.hud.money - charge["$"], 1, 40))
    h.read_hero_price = lambda side: ""            # button unreadable: the estimate, HUD-confirmed
    r = h.buy_hero_level()
    assert r and r.value == 2 and h.hero.level == 2 and h.hero.spent == 1020 + lv
    # the button's price is what's booked, not the XP estimate (live 2026-11-09: ~$54 high
    # over two levels when the one-read confirm booked the estimate)
    est = guards.hero_level_price(h.guard_state())
    charge["$"] = est - 30
    h.read_hero_price = lambda side: f"${est - 30:,}"
    spent = h.hero.spent
    r = h.buy_hero_level()
    assert r and r.cost == est - 30 and h.hero.level == 3 and h.hero.spent == spent + est - 30
    # a misread bigger than a whole level is ignored (falls back to the estimate)
    charge["$"] = guards.hero_level_price(h.guard_state())
    h.read_hero_price = lambda side: "$999,999"
    assert h.buy_hero_level() and h.hero.level == 4
    # more than the money: refused before pressing, panel closed
    g.hud = Hud(200, 100, 1, 40)
    for _ in range(4):                             # the filter takes a big drop once it repeats
        h.refresh_hud()
    assert h.money == 100
    h.hero.xp_into_level = 10 ** 6                 # the estimate says it's nearly free
    h.read_hero_price = lambda side: "$500"
    g.calls.clear()
    r = h.buy_hero_level()
    assert r.reason == "panel_no_money" and h.hero.level == 4 and ("click", *up) not in g.calls
    assert g.calls[-1] == ("click", *[round(v, 4) for v in panel("left")["close"]])


# --- rounds, fast-forward, auto start (PLAN 4.4) --------------------------------------------
def test_vision_toggle_probes():
    """Live frames 2026-11-05: pause-menu Auto Start on/off, Sandbox shop vs bloon panel."""
    import cv2
    from bloons.vision import probe
    d = FIX / "probes"
    want = {"auto_start_on": ("on", None), "auto_start_off": ("off", None),
            "sandbox_shop": (None, "shop"), "sandbox_bloons": (None, "bloons")}
    for stem, (auto, side) in want.items():
        img = cv2.imread(str(d / f"{stem}.jpg"))
        assert probe(img, "auto_start_toggle").label == auto, stem
        assert probe(img, "sandbox_panel").label == side, stem


def _round_harness(tmp_path, mode="standard"):
    from bloons.harness import BloonsHarness
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json", mode=mode)
    return g, h


def test_harness_start_round_and_ff(tmp_path):
    g, h = _round_harness(tmp_path)
    st = {"button": "idle"}
    h.read_button = lambda frame: st["button"]

    def on_key(game, k):
        if k == "space":
            st["button"] = {"idle": "ff_off", "ff_off": "ff_on", "ff_on": "ff_off"}[st["button"]]
    g.on_key = on_key
    # FF and Auto Start aren't actions: the harness keeps FF on itself
    assert not hasattr(h, "set_fast_forward") and not hasattr(h, "set_auto_start")
    # the round starts at normal speed, so a second press turns FF on
    r = h.start_round()
    assert r and r.reason == "started" and h.round_active and h.fast_forward is True
    assert g.calls == [("key", "space"), ("key", "space")] and st["button"] == "ff_on"
    # the guard now refuses without touching the game
    assert h.start_round().reason == "round_active" and len(g.calls) == 2
    # the button already running but the harness thought idle: refuse and learn it
    h.round_active = False
    assert h.start_round().reason == "round_active" and h.round_active and len(g.calls) == 2
    # a start that comes up on FF already: no second press
    g.calls.clear()
    h.round_active, st["button"] = False, "idle"
    g.on_key = lambda game, k: st.update(button="ff_on")
    assert h.start_round() and g.calls == [("key", "space")] and h.fast_forward is True
    # FF press ignored: the round still counts as started, FF stays off
    g.calls.clear()
    h.round_active, st["button"] = False, "idle"
    g.on_key = lambda game, k: st.update(button="ff_off")
    r = h.start_round()
    assert r and r.reason == "started" and h.fast_forward is False and len(g.calls) == 2
    # the game ignores the key: unconfirmed
    h.round_active, st["button"], g.on_key = False, "idle", None
    assert h.start_round().reason == "unconfirmed" and not h.round_active
    h.read_button = lambda frame: None
    assert h.start_round().reason == "no_button"


def test_harness_start_round_sandbox(tmp_path):
    from bloons.ui import UI
    g, h = _round_harness(tmp_path, "sandbox")
    st = {"side": "shop"}
    h.read_sandbox_panel = lambda frame: st["side"]
    powers = tuple(round(v, 4) for v in UI["buttons"]["powers"])
    send = tuple(round(v, 4) for v in UI["buttons"]["sandbox_send_round"])

    def on_click(game, x, y):
        if (round(x, 4), round(y, 4)) == powers:
            st["side"] = "bloons" if st["side"] == "shop" else "shop"
    g.on_click = on_click
    h.shop_layout = "top"
    r = h.start_round()
    assert r and r.reason == "sent" and st["side"] == "shop" and h.shop_layout is None
    assert g.calls == [("click", *powers), ("click", *send), ("click", *powers)]
    # Sandbox: a send marks the round active for good (round-only abilities), and can repeat
    assert h.round_active
    # already on the bloon panel: no toggle before sending
    g.calls.clear()
    st["side"] = "bloons"
    assert h.start_round() and g.calls == [("click", *send), ("click", *powers)]
    # the toggle didn't take: refuse before sending
    g.calls.clear()
    st["side"], g.on_click = "shop", None
    assert h.start_round().reason == "no_panel" and ("click", *send) not in g.calls
    # FF off after a send: one Space turns it on; already on: left alone
    g.calls.clear()
    st["side"], g.on_click = "bloons", on_click
    btn = {"b": "ff_off"}
    h.read_button = lambda frame: btn["b"]
    g.on_key = lambda game, k: btn.update(b="ff_on" if btn["b"] == "ff_off" else "ff_off")
    assert h.start_round() and g.calls[-1] == ("key", "space") and btn["b"] == "ff_on" and h.fast_forward
    g.calls.clear()
    assert h.start_round() and ("key", "space") not in g.calls and btn["b"] == "ff_on"


def test_harness_start_round_ff_needs_agreeing_reads(tmp_path):
    """One misread ff_off frame right after the start mustn't toggle FF (it was already on)."""
    g, h = _round_harness(tmp_path)
    reads = ["idle", "idle", "ff_off", "ff_on", "ff_on"]
    h.read_button = lambda frame: reads.pop(0) if reads else "ff_on"
    r = h.start_round()
    assert r and r.reason == "started" and g.calls == [("key", "space")] and h.fast_forward is True


# --- abilities (PLAN 4.5) -------------------------------------------------------------------
def test_hero_abilities_data():
    import json
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "bloons/data"
    ha = json.loads((d / "hero_abilities.json").read_text())
    heroes = json.loads((d / "heroes.json").read_text())["heroes"]
    assert set(ha["heroes"]) == set(heroes) and ha["_meta"]["game_version"]
    for k, rows in ha["heroes"].items():
        for a in rows or []:
            assert 1 <= a["level"] <= 20 and a["name"], k
            assert a["name"].split()[0] in heroes[k]["levels"][a["level"] - 1], (k, a["name"])


def test_harness_ability_slots(tmp_path):
    from bloons.harness import BloonsHarness, HERO_TID
    g = FakeGame()
    now = [0.0]
    h = BloonsHarness(g, "monkey_meadow", hero="quincy", sleep=lambda s: None,
                      spots_path=tmp_path / "s.json", mode="sandbox")
    h.clock = lambda: now[0]
    a, b = h.register("boomerang_monkey", (0.3, 0.3), 220), h.register("boomerang_monkey", (0.5, 0.2), 220)
    for t in (a, b):
        for _ in range(4):
            h.record_upgrade(t, 1, 0)
    # two Turbo Charges share one bar slot; new abilities start on cooldown
    assert [(x.slot, x.name, x.ready) for x in h.abilities] == [(0, "Turbo Charge", False)]
    assert h.use_ability(0).reason == "cooldown"
    now[0] = 50.0
    assert h.use_ability(0).reason == "round_only"      # Turbo Charge isn't usable between rounds
    h.round_active = True
    h.icon_diff = lambda x, y: 30.0                      # the icon changes: it fired
    r = h.use_ability(0)
    assert r and r.value == "Turbo Charge" and g.calls[-1] == ("key", "1")
    assert h.abilities[0].ready                          # the second instance is still ready
    assert h.use_ability(0) and not h.abilities[0].ready
    now[0] = 96.0
    assert h.abilities[0].ready
    # no change on the icon: unconfirmed, and both instances back off 3 s
    h.icon_diff = lambda x, y: 2.0
    assert h.use_ability(0).reason == "unconfirmed" and not h.abilities[0].ready
    now[0] = 99.5
    assert h.abilities[0].ready
    # tier 5 on the same path replaces that tower's Turbo Charge with Perma Charge (new slot)
    h.record_upgrade(a, 1, 0)
    assert [(x.name, x.tid) for x in h.abilities] == [("Turbo Charge", b), ("Perma Charge", a)]
    # selling drops the instance; an empty slot disappears and the next one moves up
    h.unregister(b)
    assert [x.name for x in h.abilities] == ["Perma Charge"]
    # hero abilities arrive with levels (Quincy: Rapid Shot at 3); slot 1's key '2' is confirmed
    h.register_hero((0.6, 0.6), 540)
    for lv in (2, 3):
        h.hero.level = lv - 1
        h._gain_hero_abilities(lv)
    assert [x.name for x in h.abilities] == ["Perma Charge", "Rapid Shot"] and h.abilities[1].tid == HERO_TID
    now[0] = 1000.0
    h.icon_diff = lambda x, y: 30.0
    assert h.place_ability_target(1, 0.4, 0.4)
    assert g.calls[-2:] == [("key", "2"), ("click", 0.4, 0.4)]
    assert h.use_ability(5).reason == "no_ability"


def test_harness_use_ability_of(tmp_path):
    from bloons.harness import BloonsHarness
    from bloons.ui import panel
    g = FakeGame()
    now = [0.0]
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    h.clock = lambda: now[0]
    h.read_panel = lambda frame: "left"
    h.icon_diff = lambda x, y: 30.0
    a = h.register("boomerang_monkey", (0.7, 0.3), 220)
    assert h.use_ability_of(a).reason == "no_ability" and h.use_ability_of(42).reason == "no_tower"
    for _ in range(4):
        h.record_upgrade(a, 1, 0)
    now[0], h.round_active = 100.0, True
    r = h.use_ability_of(a)
    assert r and r.value == "Turbo Charge"
    left = panel("left")
    assert g.calls[-3:] == [("click", 0.7, 0.3), ("click", *[round(v, 4) for v in left["ability_icon"]]),
                            ("click", *[round(v, 4) for v in left["close"]])]
    assert h.use_ability_of(a).reason == "cooldown"


# --- targeting and target points (PLAN 4.6) -------------------------------------------------
def _targeting_harness(tmp_path, kind, modes, xy=(0.7, 0.45)):
    """A harness with one `kind` tower whose panel (left) cycles `modes` on Tab."""
    from bloons.harness import BloonsHarness
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    st = {"i": 0, "panel": None, "clicks_left": 0}
    tid = h.register(kind, xy, 0)
    h.read_targeting = lambda side: modes[st["i"]] if st["panel"] else ""
    h.read_panel = lambda frame: st["panel"]

    def on_key(game, k):
        if k == "tab" and st["panel"]:
            st["i"] = (st["i"] + 1) % len(modes)
    g.on_key = on_key
    return g, h, tid, st


def test_harness_set_targeting(tmp_path):
    from bloons.ui import panel
    g, h, tid, st = _targeting_harness(tmp_path, "dart_monkey", ["FIRST", "LAST", "CLOSE", "STRONG"])
    close = ("click", *[round(v, 4) for v in panel("left")["close"]])

    def on_click(game, x, y):
        p = (round(x, 4), round(y, 4))
        st["panel"] = None if ("click", *p) == close else ("left" if p == (0.7, 0.45) else st["panel"])
    g.on_click = on_click
    r = h.set_targeting(tid, "strong")
    assert r and r.value == "strong" and h.registry[tid].targeting == "strong"
    assert [c for c in g.calls if c[0] == "key"] == [("key", "tab")] * 3 and g.calls[-1] == close
    g.calls.clear()
    assert h.set_targeting(tid, "strong") and not [c for c in g.calls if c[0] == "key"]   # already set
    # a mode the tower doesn't offer: one full cycle, then refuse
    g.calls.clear()
    assert h.set_targeting(tid, "patrol_points").reason == "no_such_mode"
    assert len([c for c in g.calls if c[0] == "key"]) == 4 and g.calls[-1] == close
    # OCR-tolerant labels: 'FOLLOW MOUSe' is follow_touch
    g2, h2, t2, st2 = _targeting_harness(tmp_path, "heli_pilot", ["FOLLOW MOUSe", "LOCK IN PLACE", "PATROL POINTS"])
    st2["panel"] = "left"
    h2.close_panel = lambda side: None
    assert h2.set_targeting(t2, "follow_touch") and h2.set_flight_path(t2, "lock_in_place")
    assert h.set_targeting(99, "first").reason == "no_tower"


def test_harness_target_points(tmp_path):
    from bloons.ui import panel
    g, h, tid, st = _targeting_harness(tmp_path, "heli_pilot", ["FOLLOW MOUSE", "LOCK IN PLACE", "PATROL POINTS"])
    left = panel("left")
    close = (round(left["close"][0], 4), round(left["close"][1], 4))
    lb = left["targeting_label"]
    button = (round((lb[0] + lb[2]) / 2, 4), round((lb[1] + lb[3]) / 2, 4))

    def on_click(game, x, y):
        p = (round(x, 4), round(y, 4))
        if p == close:
            st["panel"] = None
        elif p == button and st["panel"]:
            st["panel"], st["clicks_left"] = None, (1 if st["i"] == 1 else 2)
        elif st["clicks_left"]:
            st["clicks_left"] -= 1
            if not st["clicks_left"]:
                st["panel"] = "left"
        elif p == (0.7, 0.45):
            st["panel"] = "left"
    g.on_click = on_click
    r = h.set_target_point(tid, 0.45, 0.6)
    assert r and r.value == [(0.45, 0.6)] and h.registry[tid].targeting == "lock_in_place"
    assert ("click", *button) in g.calls and g.calls[-1] == ("click", *close) and st["panel"] is None
    r = h.set_patrol_points(tid, [(0.3, 0.15), (0.6, 0.85)])
    assert r and h.registry[tid].targeting == "patrol_points"
    assert h.set_patrol_points(tid, [(0.3, 0.15)]).reason == "bad_points"
    # the panel never comes back: cancel point mode, unconfirmed
    g.calls.clear()
    st["i"], st["clicks_left"] = 1, 0
    g.on_click = lambda game, x, y: st.update(panel="left") if (round(x, 4), round(y, 4)) == (0.7, 0.45) else (
        st.update(panel=None) if (round(x, 4), round(y, 4)) in (button, close) else None)
    assert h.set_target_point(tid, 0.45, 0.6).reason == "unconfirmed"
    # towers without a point mode
    dart = h.register("dart_monkey", (0.2, 0.2), 0)
    assert h.set_target_point(dart, 0.5, 0.5).reason == "no_target_point"
    assert h.set_patrol_points(dart, [(0, 0), (1, 1)]).reason == "no_patrol"


# --- prompts (PLAN 4.8) ---------------------------------------------------------------------
def test_harness_handle_prompts(tmp_path):
    from bloons.harness import BloonsHarness
    from bloons.ui import UI
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    pt = lambda sec, k: tuple(round(v, 4) for v in UI[sec][k])
    screens = ["restart_confirm", "pause", "game"]
    h.read_kind = lambda frame: screens[0]
    g.on_click = lambda game, x, y: screens.pop(0)
    assert h.handle_prompts() == ["restart_confirm", "pause"]
    assert [c[1:] for c in g.calls] == [pt("menus", "restart_confirm_cancel"), pt("pause", "continue")]
    # outcomes are reported, not clicked
    g.calls.clear()
    screens[:] = ["victory"]
    assert h.handle_prompts() == [] and g.calls == []
    # per-call override: click Next on victory
    screens[:] = ["victory", "collection_event", "map_select"]
    assert h.handle_prompts({"victory": "menus.victory_next"}) == ["victory", "collection_event"]
    # a prompt that never goes away stops after max_steps
    g.on_click = None
    screens[:] = ["pause"]
    assert len(h.handle_prompts()) == UI["prompts"]["max_steps"]


# --- new game / restart / home (PLAN 4.9) ---------------------------------------------------
def _menu_harness(tmp_path, start, save_exists=False):
    """A harness on a scripted menu graph: clicks on known UI points move between screens."""
    from bloons.harness import BloonsHarness
    from bloons.ui import UI
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    now = [0.0]

    def tick(s):
        now[0] += s
    h.sleep, h.clock = tick, lambda: now[0]
    st = {"kind": start, "pages": 0, "slot": None, "diff": None, "mode": None}
    P = lambda sec, k: tuple(round(v, 4) for v in UI[sec][k])
    ms = UI["map_select"]

    def on_click(game, x, y):
        p, k = (round(x, 4), round(y, 4)), st["kind"]
        if k == "game" and p == P("buttons", "settings"):
            st["kind"] = "pause"
        elif k == "pause" and p == P("pause", "home"):
            st["kind"] = "home"
        elif k == "pause" and p == P("pause", "restart"):
            st["kind"] = "restart_confirm"
        elif k == "restart_confirm" and p == P("menus", "restart_confirm_restart"):
            st["kind"] = "game"
        elif k == "home" and p == P("menus", "home_play"):
            st["kind"], st["pages"] = "map_select", st.get("last_page", 3)   # reopens on the last map's page
        elif k == "map_select" and p in [tuple(round(v, 4) for v in s) for s in ms["slots"]]:
            st["slot"] = (st["pages"], [tuple(round(v, 4) for v in s) for s in ms["slots"]].index(p))
            st["last_page"] = st["pages"]
            st["kind"] = "difficulty_select"
        elif k == "difficulty_select" and p == P("menus", "difficulty_hard"):
            st["diff"], st["kind"] = "hard", "mode_select"
        elif k == "difficulty_select" and p == P("menus", "difficulty_easy"):
            st["diff"], st["kind"] = "easy", "mode_select"
        elif k == "mode_select" and p in (P("menus", "mode_standard"), P("menus", "mode_sandbox")):
            st["mode"] = "standard" if p == P("menus", "mode_standard") else "sandbox"
            st["kind"] = "overwrite_save" if save_exists else "game"
        elif k == "overwrite_save" and p == P("menus", "overwrite_save_cancel"):
            st["kind"] = "mode_select"
        elif k == "overwrite_save" and p == P("menus", "overwrite_save_ok"):
            st["kind"] = "game"
        elif k in ("map_select", "difficulty_select") and p == P("menus", "back"):
            st["kind"] = {"map_select": "home", "difficulty_select": "map_select"}[k]
    pages = [["MOnKeY MeADOW", "IN THE LOOP", "SKULLTWEAK", "THREE MINES 'ROUND", "SPA PITS", "TINKERTON"],
             ["TREE STUMP", "TOWN CENTER", "A", "B", "C", "D"], ["E"] * 6, ["F"] * 6,
             ["LOGS", "08 END OF THE ROAD", "", "", "", ""], ["LOST CREVASSE"] + ["G"] * 5]

    def click2(game, x, y):
        if st["kind"] == "map_select" and (round(x, 4), round(y, 4)) == P("map_select", "category_beginner"):
            st["pages"] = 1                                   # lands in Beginner, not always page 0
        elif st["kind"] == "map_select" and (round(x, 4), round(y, 4)) == tuple(round(v, 4) for v in ms["next_page"]):
            st["pages"] = (st["pages"] + 1) % len(pages)        # wraps like the game
        elif st["kind"] == "map_select" and (round(x, 4), round(y, 4)) == tuple(round(v, 4) for v in ms["prev_page"]):
            st["pages"] = (st["pages"] - 1) % len(pages)
        else:
            on_click(game, x, y)
    g.on_click = click2
    h.read_kind = lambda frame: st["kind"]
    h.read_map_names = lambda: pages[st["pages"]] if st["kind"] == "map_select" else [""] * 6
    return g, h, st


def test_harness_new_game(tmp_path):
    g, h, st = _menu_harness(tmp_path, "game")
    h.round_active = True
    r = h.new_game("logs", "hard", "standard")
    assert r and r.value == ("logs", "hard", "standard") and st["slot"] == (4, 0) and st["diff"] == "hard"
    assert h.map_name == "logs" and h.map["_meta"] and h.difficulty == "hard" and not h.round_active
    assert h.new_game("nowhere").reason == "unknown_map" and h.new_game(difficulty="impoppable").reason == "unknown_difficulty"
    # from a half-finished menu path: back out to home first
    g, h, st = _menu_harness(tmp_path, "difficulty_select")
    assert h.new_game("monkey_meadow", "easy", "sandbox") and st["mode"] == "sandbox" and h.mode == "sandbox"


def test_harness_new_game_save_exists(tmp_path):
    g, h, st = _menu_harness(tmp_path, "home", save_exists=True)
    assert h.new_game("tree_stump", "easy", "standard").reason == "save_exists" and st["kind"] == "mode_select"
    assert h.map_name == "monkey_meadow"                    # unchanged on failure
    g, h, st = _menu_harness(tmp_path, "home", save_exists=True)
    assert h.new_game("tree_stump", "easy", "standard", overwrite_save=True) and h.map_name == "tree_stump"


def test_harness_pick_map_fallbacks(tmp_path):
    from bloons.ui import UI
    # the name 'LOGS' never OCRs, but 'END OF THE ROAD' places the page: click the data slot
    g, h, st = _menu_harness(tmp_path, "home")
    real = h.read_map_names
    h.read_map_names = lambda: ["" if n == "LOGS" else n for n in real()]
    r = h.new_game("logs", "easy", "standard")
    assert r and st["slot"] == (4, 0) and ("click_slot", 0) in h.pick_trace
    # a page flip lost mid-slide is sent again
    g, h, st = _menu_harness(tmp_path, "home")
    click, dropped = g.on_click, []
    def lossy(game, x, y):
        if (round(x, 4), round(y, 4)) == tuple(round(v, 4) for v in UI["map_select"]["next_page"]) and not dropped:
            dropped.append(1)
            return
        click(game, x, y)
    g.on_click = lossy
    assert h.new_game("tree_stump", "easy", "standard") and st["slot"] == (1, 0)
    assert ("flip_lost", "next_page") in h.pick_trace


def test_harness_restart_and_home(tmp_path):
    g, h, st = _menu_harness(tmp_path, "game")
    tid = h.register("dart_monkey", (0.3, 0.3), 170)
    assert h.restart() and st["kind"] == "game" and tid not in h.registry
    assert h.go_home() and st["kind"] == "home"
    assert h.go_home().reason == "home"                      # already there
    assert h.restart().reason == "not_in_game:home"


def test_harness_open_pause_closes_panel(tmp_path):
    """Live 2026-11-09: with the right panel open its close ✕ sits where the gear is, so the
    gear click only closed the panel and go_home / restart / new_game gave no_pause_menu."""
    g, h, st = _menu_harness(tmp_path, "game")
    st["panel"] = "right"
    h.read_panel = lambda frame: st["panel"]
    menu_click = g.on_click

    def on_click(game, x, y):
        if st["panel"]:                       # any click with the panel up just closes it
            st["panel"] = None
        else:
            menu_click(game, x, y)
    g.on_click = on_click
    g.on_key = lambda game, k: st.update(panel=None)     # close_panel's hotkey, if it has one
    assert h.go_home() and st["kind"] == "home"
    # the one panel read missed it: the second gear click still gets there
    st["kind"], st["panel"] = "game", "right"
    h.read_panel = lambda frame: None
    assert h.restart() and st["kind"] == "game"


def test_play_repl(tmp_path):
    """tools/play.py: commands reach the harness; bad input prints usage and doesn't stop the loop."""
    import io
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import play
    import pytest
    from bloons.hud import Hud
    assert play.resolve("dart", ["dart_monkey", "dartling_gunner"]) == "dart_monkey"   # whole word wins
    assert play.resolve("ace", ["monkey_ace", "dart_monkey"]) == "monkey_ace"
    with pytest.raises(play.UsageError):
        play.resolve("monkey", ["monkey_ace", "monkey_sub"])
    g, h = _round_harness(tmp_path)
    h.tint = lambda base, held, x, y: "ok"

    def spend(game, x, y):
        if len(game.calls) > 1 and game.calls[-2] == ("move", x, y) and game.hud.money >= 170:
            game.hud = Hud(200, game.hud.money - 170, 1, 40)
    g.on_click = spend
    out = io.StringIO()
    p = play.Play(h, out=out)
    for line in ("hud", "place dart .1 .5", "place dart", "place monkey .1 .5", "up 1 sideways",
                 "state", "observe", "py h.money", "nonsense", "prompts", "forget", "state"):
        assert not p.onecmd(line)
    text = out.getvalue()
    assert "$650" in text and "ok  placed cost=170 value=1\n     $480" in text
    assert "dart_monkey" in text and "usage: place <tower> <x> <y>" in text and "free cells: land_r6" in text
    assert "could mean" in text and "path is top" in text and "unknown command 'nonsense'" in text
    assert "(no towers placed through the harness)" in text and not h.registry and "handled: " in text
    assert p.onecmd("quit")


def test_harness_observe(tmp_path):
    from bloons.hud import Hud
    from bloons.state import Screen
    g, h = _round_harness(tmp_path)
    now = [100.0]
    h.clock = lambda: now[0]
    seq = []
    h.read_screen = lambda frame: seq.pop(0) if seq else Screen("game", "idle")
    s = h.observe()
    assert (s.screen, s.money, s.lives, s.round, s.max_round) == ("game", 650, 200, 1, 40)
    assert not s.round_active and not s.round_ended and s.outcome is None and s.towers == [] and s.hero is None
    assert s.round_composition["counts"] == {"red": 20} and s.path.shape[1] == 2
    free = {c: m.ok.sum() for c, m in s.placeable.items()}
    # a tower: TowerState, upgrade options, masks lose cells around it
    tid = h.register("dart_monkey", (0.25, 0.5), 170)
    h.record_upgrade(tid, 0, 120)
    s = h.observe()
    t = s.towers[0]
    assert t.tiers == (1, 0, 0) and t.range_units == 32 and t.sell_value == 203 and t.range_norm[0] > 0
    # the range ellipse: y uses the ground's y scale (0.866 of x in px); no measurement yet
    assert t.range_measured is None and abs(t.range_norm[1] / h.unit[1] - 32 * 0.866) < 1e-6
    assert [o.locked for o in s.upgrade_options[tid]] == [None, None, None] and s.upgrade_options[tid][0].cost == 170
    assert all(s.placeable[c].ok.sum() < free[c] for c in free)
    assert not s.placeable["land_r6"].placeable(0.25, 0.5)
    # round edge: idle -> running (2 frames to confirm) -> idle fires round_ended exactly once
    seq[:] = [Screen("game", "ff_off")] * 2 + [Screen("game", "idle")] * 3
    states = [h.observe() for _ in range(5)]
    assert [x.round_active for x in states] == [False, True, True, False, False]
    assert [x.round_ended for x in states] == [False, False, False, True, False] and h.round_active is False
    assert [x.next_round for x in states] == [1, 1, 1, 2, 2] and states[-1].round_composition["round"] == 2
    g.hud = Hud(200, 650, 5, 40)                      # mid-round, the HUD resyncs the count
    seq[:] = [Screen("game", "ff_on")] * 2
    assert [h.observe().rounds_done for _ in range(2)] == [1, 4]
    # outcome needs two victory frames, then sticks; the HUD isn't read off-game
    seq[:] = [Screen("victory")] * 2 + [Screen("pause")]
    assert h.observe().outcome is None and h.observe().outcome == "win" and h.observe().outcome == "win"
    assert h.rounds_done == 5
    # ability estimate: just gained -> cooling with frac 1, ready after its cooldown
    h.hero_choice = "benjamin"
    h.register_hero((0.75, 0.25), 1020)
    h._gain_hero_abilities(3)
    a = h.observe().abilities[0]
    assert a.name == "Biohack" and not a.ready and a.cooldown_frac == 1.0
    now[0] += 61
    a = h.observe().abilities[0]
    assert a.ready and a.cooldown_frac == 0.0
    assert h.mask_classes_of("monkey_ace") == ["land_ace"] and h.mask_classes_of("benjamin") == ["land_r7"]
    h.new_episode()
    assert h.observe().outcome is None
    # Sandbox: the always-FF button doesn't make a round 'active'
    g, h = _round_harness(tmp_path, mode="sandbox")
    h.read_screen = lambda frame: Screen("game", "ff_off")
    assert [h.observe().round_active for _ in range(3)] == [False] * 3 and h.observe().round_composition is None


# --- bloon detector (PLAN 5.2) --------------------------------------------------------------
BLOON_FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/bloons"


def _bloon_frames(names):
    import cv2
    return [cv2.imread(str(BLOON_FIX / f"{n}.png")) for n in names]


def test_bloon_detector_fixtures():
    """Sandbox captures on Monkey Meadow, 3 bloons of one type each (pixels outside the path
    band blacked out). Covers a bloon on a swaying grass tuft (black_05), a bloon cut in two
    by it (yellow_02) and Halloween bats over the track (blue_09: at most one stray)."""
    import json
    from bloons import detect
    det = detect.BloonDetector(json.loads((Path(__file__).resolve().parents[1] / "bloons/data/maps/monkey_meadow.json").read_text()))
    assert det.detect(_bloon_frames(["red_04"])[0]) == []              # no background yet
    runs = {"a": ["red_04", "green_05", "yellow_02", "pink_04", "white_03", "black_05"],
            "b": ["purple_04", "lead_05", "zebra_04", "rainbow_05", "ceramic_06"]}
    for run, names in runs.items():
        assert det.set_background(_bloon_frames([f"empty_{run}_{i}" for i in range(4)]))
        for name, frame in zip(names, _bloon_frames(names)):
            b = det.detect(frame)
            assert detect.summarize(b) == {name.split("_")[0]: 3}, (name, b)
            assert all(0 <= x.x <= 1 and 0 <= x.y <= 1 and x.count == 1 for x in b)
    det.set_background(_bloon_frames([f"empty_a_{i}" for i in range(4)]))
    s = detect.summarize(det.detect(_bloon_frames(["blue_09"])[0]))
    assert s.get("blue") == 3 and sum(s.values()) <= 4
    # the background itself has no bloons
    assert det.detect(_bloon_frames(["empty_a_0"])[0]) == []


def test_bloon_detector_live_standard():
    """Live Monkey Meadow Standard frames with the map's stored background: a stream of
    overlapping red + blue bloons in round 3 (one blob, split by per-pixel colour; the round
    is 25 red + 5 blue) and a round-1 frame where a stray dark fragment used to count."""
    import json
    from bloons import detect
    det = detect.BloonDetector(json.loads((Path(__file__).resolve().parents[1] / "bloons/data/maps/monkey_meadow.json").read_text()))
    det.set_background_arrays(*detect.load_background("monkey_meadow"))
    r3, r1 = _bloon_frames(["live_r3_198", "live_r1_072"])
    assert detect.summarize(det.detect(r3)) == {"red": 15, "blue": 4}
    assert detect.summarize(det.detect(r1)) == {"red": 10}


def test_bloon_counting_rules():
    from bloons import detect
    assert detect.blob_count(5000, 4900) == 1 and detect.blob_count(7500, 4900) == 1
    assert detect.blob_count(9800, 4900) == 2 and detect.blob_count(15000, 4900) == 3
    det = detect.BloonDetector.__new__(detect.BloonDetector)
    det.single, det.unit = {"yellow": 6848, "white": 2144}, (0.0027, 0.005)
    # a fragment next to a partial blob of the same bloon merges; a lone small bloon stays
    raw = [[0.358, 0.178, "yellow", 2500.0, 0.7, None], [0.357, 0.22, "yellow", 1170.0, 1.5, None],
           [0.60, 0.50, "white", 2100.0, 0.2, None]]
    out = det._merge_fragments(raw)
    assert len(out) == 2 and sorted(round(b[3]) for b in out) == [2100, 3670]
    # two whole bloons side by side don't merge
    assert len(det._merge_fragments([[0.3, 0.2, "yellow", 6800.0, 0.3, None], [0.31, 0.2, "yellow", 6900.0, 0.3, None]])) == 2


def test_harness_observe_bloons(tmp_path):
    """observe(): the map's stored background (data/maps/monkey_meadow.bg.png) is loaded;
    idle Standard frames give bloons == [] without detecting (pre-round track markers);
    during a round the detector runs; our towers are left out of the band."""
    from bloons.state import Screen
    g, h = _round_harness(tmp_path)
    assert h.detector is not None and h.detector.bg is not None
    frames = {"cur": _bloon_frames(["red_04"])[0]}

    class F:
        def content_bgr(self):
            return frames["cur"]
    g.grab = lambda: F()
    seq = []
    h.read_screen = lambda frame: seq.pop(0) if seq else Screen("game", "idle")
    assert h.observe().bloons == []                          # idle: not even looked at
    seq[:] = [Screen("game", "ff_off")] * 3
    h.observe()
    s = h.observe()
    assert s.round_active and [b.kind for b in s.bloons] == ["red"] * 3
    b0 = s.bloons[0]                                         # a tower on top of one bloon hides it
    h.register("dart_monkey", (b0.x, b0.y - 0.02), 170)
    seq[:] = [Screen("game", "ff_off")]
    assert len(h.observe().bloons) == 2
    seq[:] = [Screen("pause")]
    assert h.observe().bloons is None                        # off the game screen


# --- upgrade panel check (PLAN 5.3) ---------------------------------------------------------
def _panel_fixtures():
    import json
    return json.loads((FIX / "panel" / "panel.json").read_text())["frames"]


def _panel_ocr(side, texts):
    """A fake `ocr(box)` answering the path_prices boxes with the live OCR text."""
    from bloons.ui import panel
    boxes = panel(side)["path_prices"]
    return lambda box: texts.get(str(boxes.index(box)), "") if box in boxes else ""


def test_panel_parse_price():
    from bloons.panel import parse_price
    assert parse_price("$1,530") == 1530 and parse_price("$85") == 85 and parse_price("$12,750") == 12750
    assert parse_price("") is None and parse_price("MAX UPGRADES") is None


def test_panel_read_fixtures():
    """Live Dart Monkey panels (Sandbox 0-0-0 .. 5-2-0, a second dart at 4-0-0 with tier 5
    taken, a Standard frame with Sharp Shots unaffordable, a left-side panel): state from the
    button colour, price from the (recorded) OCR, tier from the pips."""
    import cv2
    from bloons.panel import read_paths
    for stem, fx in _panel_fixtures().items():
        img = cv2.imread(str(FIX / "panel" / f"{stem}.jpg"))
        rows = read_paths(img, fx["side"], _panel_ocr(fx["side"], fx["ocr"]))
        assert [[r.state, r.price, r.tier] for r in rows.values()] == fx["truth"], stem


def test_panel_compare_with_guards():
    """Every Sandbox frame agrees with the price table + guards at the tiers the harness
    recorded; a wrong registry (tiers or price table) shows up as a disagreement."""
    import cv2
    from bloons.harness import BloonsHarness
    from bloons.panel import compare, read_paths
    h = BloonsHarness(FakeGame(), "monkey_meadow", sleep=lambda s: None)
    h.hud.update(FakeGame().hud)
    fx = _panel_fixtures()
    first = h.register("dart_monkey", (0.1, 0.8), 170)
    for stem in [s for s in fx if s.startswith("dart_")]:
        h.registry[first].tiers = tuple(int(c) for c in stem.split("_")[1])
        img = cv2.imread(str(FIX / "panel" / f"{stem}.jpg"))
        rows = read_paths(img, "right", _panel_ocr("right", fx[stem]["ocr"]))
        assert compare(h.registry[first], rows, h.guard_state()) == {}, stem
    # the second dart: path 1 at tier 4 shows MAX UPGRADES because the first one has tier 5
    second = h.register("dart_monkey", (0.22, 0.85), 170)
    h.registry[second].tiers = (4, 0, 0)
    img = cv2.imread(str(FIX / "panel" / "dart2_400_right.jpg"))
    rows = read_paths(img, "right", _panel_ocr("right", fx["dart2_400_right"]["ocr"]))
    assert compare(h.registry[second], rows, h.guard_state()) == {}
    h.registry[first].tiers = (4, 2, 0)                     # no tier 5 any more: path 1 should be open
    assert compare(h.registry[second], rows, h.guard_state()) == {0: ["state max != open"]}
    h.registry[first].tiers = (5, 2, 0)
    h.registry[second].tiers = (4, 1, 0)                    # a registry with the wrong tiers
    assert compare(h.registry[second], rows, h.guard_state()) == {
        1: ["price 85 != 160", "tier 0 != 1"], 2: ["state buy != closed"]}
    # Standard frame, $119: the unaffordable row still counts as open at the right price
    img = cv2.imread(str(FIX / "panel" / "std_000_poor_right.jpg"))
    rows = read_paths(img, "right", _panel_ocr("right", fx["std_000_poor_right"]["ocr"]))
    h.registry[first].tiers = (0, 0, 0)
    assert compare(h.registry[first], rows, h.guard_state()) == {}


def test_harness_check_upgrades_and_upgrade_reads_panel(tmp_path):
    from bloons import guards
    from bloons.harness import BloonsHarness
    from bloons.hud import Hud
    from bloons.panel import PathRow
    g = FakeGame()
    g.hud = Hud(200, 5000, 1, 40)
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json")
    h.refresh_hud()
    h.read_panel = lambda frame: "right"
    tid = h.register("dart_monkey", (0.2, 0.5), 170)
    t = h.registry[tid]
    price = guards.upgrade_cost(t, 0, h.guard_state())
    rows = {0: PathRow("buy", price, 0), 1: PathRow("buy", 85, 0), 2: PathRow("buy", 75, 0)}
    h.read_paths = lambda side, paths=(0, 1, 2), prices=True: {p: rows[p] for p in paths}
    assert h.observe().upgrade_options[tid][0].checked is None
    r = h.check_upgrades(tid)
    assert r and r.reason == "match" and r.value[1] == {}
    assert g.calls[-1] == ("click", *[round(v, 4) for v in __import__("bloons.ui", fromlist=["panel"]).panel("right")["close"]])
    assert [o.checked for o in h.observe().upgrade_options[tid]] == [True, True, True]
    rows[1] = PathRow("buy", 90, 0)
    r = h.check_upgrades(tid)
    assert not r and r.reason == "mismatch" and r.value[1] == {1: ["price 90 != 85"]}
    assert [o.checked for o in h.observe().upgrade_options[tid]] == [True, False, True]
    # upgrade(): the panel's price is what gets confirmed and recorded
    rows[0] = PathRow("buy", price + 5, 0)
    g.on_key = lambda game, k: k == "," and setattr(game, "hud", Hud(200, game.hud.money - price - 5, 1, 40))
    r = h.upgrade(tid, 0)
    assert r and r.cost == price + 5 and t.spent == 170 + price + 5 and t.tiers == (1, 0, 0)
    assert h.price_mismatches == [("dart_monkey", (0, 0, 0), 0, price + 5, price)]
    assert h.observe().upgrade_options[tid][0].checked is None      # new tiers: not read yet
    # a locked / unaffordable row isn't pressed
    g.on_key = None
    rows[1] = PathRow("closed", None, 0)
    n = len(g.calls)
    r = h.upgrade(tid, 1)
    assert not r and r.reason == "panel_locked" and ("key", ".") not in g.calls[n:]
    rows[1] = PathRow("poor", 85, 0)
    assert h.upgrade(tid, 1).reason == "panel_no_money"
    # an unreadable panel (overlay, black frame) is ignored: the guard's price is used
    rows[1] = PathRow("unknown", None, 0)
    assert h.upgrade(tid, 1).reason == "unconfirmed"                # FakeGame money didn't move
    assert h.check_upgrades(99).reason == "no_tower"
    # the HUD missed the money drop, but the row's pips went up after the press: confirmed
    rows[1] = PathRow("buy", 85, 0)
    g.on_key = lambda game, k: k == "." and rows.__setitem__(1, PathRow("buy", 160, 1))
    r = h.upgrade(tid, 1)
    assert r and r.cost == 85 and t.tiers == (1, 1, 0)
    rows[1] = PathRow("buy", 160, 1)
    g.on_key = None                                                # nothing changes: still unconfirmed
    assert h.upgrade(tid, 1).reason == "unconfirmed" and t.tiers == (1, 1, 0)


# --- 5.4 ability bar --------------------------------------------------------------------------
BAR_FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/abilitybar"


def _bar_frame(stem, meta):
    import cv2
    import numpy as np
    crop = cv2.imread(str(BAR_FIX / f"{stem}.jpg"))
    (x0, y0), (w, h) = meta["offset"], meta["frame"]
    img = np.zeros((h, w, 3), np.uint8)
    img[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return img


def test_abilitybar_read_fixtures():
    # 10 live strips, 3 slots each: ring fraction within 0.03 of the fitted cooldown line,
    # no ring on ready icons (also with Fan Club's stray orange pixels at ~105 deg)
    from bloons.abilitybar import read_bar
    meta = json.loads((BAR_FIX / "bar.json").read_text())
    for stem, truth in meta["frames"].items():
        reads = read_bar(_bar_frame(stem, meta), 3)
        assert all(r.seen for r in reads), stem
        for r, want in zip(reads, truth):
            assert r.ring == (want > 0), (stem, r, want)
            assert abs(r.frac - want) <= 0.03, (stem, r, want)


def test_abilitybar_unseen_and_shift():
    import numpy as np
    from bloons.abilitybar import read_bar
    reads = read_bar(np.zeros((1594, 2940, 3), np.uint8), 2)      # black: nothing to read
    assert [(r.seen, r.frac) for r in reads] == [(False, None), (False, None)]
    assert read_bar(None, 2) == [] and read_bar(np.zeros((10, 10, 3), np.uint8), 0) == []
    # with the panel on the left the bar moves right: the unshifted slot 0 sees no ring there
    meta = json.loads((BAR_FIX / "bar.json").read_text())
    img = _bar_frame("0117.72_post3", meta)
    assert read_bar(img, 2)[1].frac == 1.0 and not read_bar(img, 2, "left")[1].ring


def test_abilitybar_filter():
    from bloons.abilitybar import BarFilter, SlotRead
    f = BarFilter(jump_frac=0.15, confirm_s=2.0)
    assert f.update([SlotRead(0, True, 0.8)], 0.0)[0].frac == 0.8
    # the ring vanishes mid-cooldown (Sandbox round start): held until it lasts 2 s
    assert f.update([SlotRead(0, False, 0.0)], 0.5)[0].frac == 0.8
    assert f.update([SlotRead(0, True, 0.75)], 1.5)[0].frac == 0.75
    assert f.update([SlotRead(0, False, 0.0)], 2.0)[0].frac == 0.75
    assert f.update([SlotRead(0, False, 0.0)], 3.4)[0].frac == 0.75
    assert f.update([SlotRead(0, False, 0.0)], 4.1)[0].frac == 0.0       # lasted: believed
    # sparse reads: a ring read long ago may have run out since -> believed at once
    f.update([SlotRead(0, True, 0.6)], 20.0)
    assert f.update([SlotRead(0, False, 0.0)], 25.0)[0].frac == 0.0
    # with the cooldown known: held while > 1 s should be left, reporting the extrapolation
    f = BarFilter(confirm_s=2.0, slack_s=1.0)
    f.update([SlotRead(0, True, 0.12)], 40.0, [50.0])                   # 6 s left
    assert f.update([SlotRead(0, False, 0.0)], 42.0, [50.0])[0].frac == 0.08   # live case, 43.0
    assert f.update([SlotRead(0, True, 0.06)], 43.0, [50.0])[0].frac == 0.06
    assert not f.update([SlotRead(0, False, 0.0)], 45.5, [50.0])[0].ring   # 0.5 s left: finish
    # a natural finish (small arc -> none) is taken at once; unseen reads pass through
    f.update([SlotRead(0, True, 0.05)], 10.0)
    assert not f.update([SlotRead(0, False, 0.0)], 10.5)[0].ring
    assert f.update([SlotRead(0, False, None, False)], 11.0)[0].seen is False


def test_harness_sync_abilities(tmp_path):
    import numpy as np
    from bloons.abilitybar import SlotRead
    from bloons.harness import BloonsHarness
    g = FakeGame()
    now = [100.0]
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json", mode="sandbox")
    h.clock = lambda: now[0]
    t = h.register("boomerang_monkey", (0.3, 0.3), 220)
    for _ in range(4):
        h.record_upgrade(t, 1, 0)                 # Turbo Charge, 45 s, starts on cooldown
    frame = np.zeros((10, 10, 3), np.uint8)
    reads = {"r": [SlotRead(0, False, 0.0)]}
    h.read_bar = lambda content, n, side=None: reads["r"][:n]
    assert not h.abilities[0].ready
    assert h.sync_abilities(frame) and h.abilities[0].ready          # the bar says ready: it is
    reads["r"] = [SlotRead(0, True, 0.5)]
    h.sync_abilities(frame)
    assert not h.abilities[0].ready and h.ability_slots[0]["inst"][(t, 1)] == 100.0 + 22.5
    reads["r"] = [SlotRead(0, False, None, False)]                    # unseen: estimate kept
    h.sync_abilities(frame)
    assert h.ability_slots[0]["inst"][(t, 1)] == 122.5
    assert h.sync_abilities(None) is None
    # use_ability reads the bar before its guard: a ready read lets it fire early
    now[0] = 103.0              # 19.5 s should be left: the filter holds the ring-less read back
    reads["r"] = [SlotRead(0, False, 0.0)]
    assert h.sync_abilities(frame)[0].frac == round(19.5 / 45, 3)
    now[0] = 123.0                                   # run out by now: believed at once
    h.round_active = True
    reads["r"] = [SlotRead(0, False, 0.0)]
    h.icon_diff = lambda x, y: 30.0      # the crop changes (icon coming off cooldown) but no ring:
    assert h.use_ability(0).reason == "unconfirmed" and g.calls[-1] == ("key", "1")   # not a use
    reads["r"] = [SlotRead(0, False, 0.0)]
    g.on_key = lambda game, k: reads.update(r=[SlotRead(0, True, 1.0)])   # the press draws a full ring
    r = h.use_ability(0)
    assert r and r.reason == "used" and not h.abilities[0].ready


# --- 5.5 range circles ------------------------------------------------------------------------
RANGE_FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/range"


def _range_exclude(side):
    from bloons import ui
    from bloons.ui import UI
    rc, p = UI["range_circle"], ui.panel(side)
    (ax, ay), (hx, hy) = p["ability_icon"], rc["ability_icon_half"]
    return [tuple(b) for b in rc["exclude_always"]] + [tuple(p["panel"]), (ax - hx, ay - hy, ax + hx, ay + hy)]


def test_range_fixtures():
    # 4 live circles: a crosspath Dart 3-0-2 (additive 52.8, not 55.2), Tack (its sprite
    # doesn't change), Glue 0-3-0 in the corner (its tall sprite stops ~40 % of the rays),
    # Ice 3-0-1. Within 0.3 units of the table.
    import cv2
    import numpy as np
    from bloons import rangecircle
    meta = json.loads((RANGE_FIX / "range.json").read_text())
    W, H = meta["frame"]
    unit = rangecircle.ground_unit(json.loads((RANGE_FIX.parents[1] / "maps/monkey_meadow.json").read_text())["scale"]["unit_norm"])
    for name, c in meta["cases"].items():
        fr = {}
        for k in ("sel", "des"):
            f = np.zeros((H, W, 3), np.uint8)
            x0, y0, x1, y1 = c["box"]
            f[y0:y1, x0:x1] = cv2.imread(str(RANGE_FIX / f"{name}_{k}.jpg"))
            fr[k] = f
        r = rangecircle.measure(fr["sel"], fr["des"], c["xy"], unit, _range_exclude(c["side"]))
        assert r.units is not None and abs(r.units - c["range"]) <= 0.3, (name, r)
        assert r.spread < 0.5, (name, r)


def test_range_synthetic():
    # an ellipse disc (ry = 0.866 rx in px) on a textured background, a sprite outline that
    # changes, a foliage blob that moves; plus: identical frames -> unreadable
    import numpy as np
    from bloons import rangecircle
    rng = np.random.default_rng(0)
    H, W = 800, 1400
    des = rng.integers(60, 200, (H, W, 3), dtype=np.uint8)
    sel = des.copy()
    unit = (0.01, 0.01 * W / H * 0.866)                 # 14 px per unit along x on the ground
    cx, cy, rad = 500, 400, 20.0                        # 20 units: 280 px x, 242 px y
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((xx - cx) / (rad * unit[0] * W)) ** 2 + ((yy - cy) / (rad * unit[1] * H)) ** 2 <= 1
    sel[disc] = (sel[disc] * 0.7).astype(np.uint8)
    ring = np.abs(np.hypot(xx - cx, yy - cy) - 30) < 2  # sprite outline highlight
    sel[ring] = 255
    sel[(np.hypot(xx - cx, yy - cy) < 28)] = des[(np.hypot(xx - cx, yy - cy) < 28)]   # sprite unchanged
    blob = (np.abs(xx - 900) < 40) & (np.abs(yy - 150) < 40)
    sel[blob] = 0
    r = rangecircle.measure(sel, des, (cx / W, cy / H), unit)
    assert abs(r.units - rad) < 0.3 and r.rays > 100, r
    assert rangecircle.measure(des, des, (cx / W, cy / H), unit).units is None
    # a panel box over the right half: fewer rays, same answer
    r2 = rangecircle.measure(sel, des, (cx / W, cy / H), unit, [(0.4, 0.0, 1.0, 1.0)])
    assert abs(r2.units - rad) < 0.3 and r2.rays < r.rays


def test_harness_measure_range(tmp_path):
    from bloons.harness import BloonsHarness
    from bloons.rangecircle import RangeRead
    g = FakeGame()
    h = BloonsHarness(g, "monkey_meadow", sleep=lambda s: None, spots_path=tmp_path / "s.json", mode="sandbox")
    assert h.measure_range(7).reason == "no_tower"
    tid = h.register("dart_monkey", (0.3, 0.3), 170)
    h.read_panel = lambda frame: None
    assert h.measure_range(tid).reason == "no_panel"
    h.read_panel = lambda frame: "right"
    seen = {}

    def fake(sel, des, centre, unit, exclude):
        seen.update(centre=centre, unit=unit, n=len(exclude))
        return RangeRead(seen.get("value"), 120, 0.1)
    h.read_range = fake
    assert h.measure_range(tid).reason == "unreadable" and tid not in h.range_checks
    seen["value"] = 31.9
    r = h.measure_range(tid)
    assert r and r.reason == "match" and r.value == 31.9 and h.range_checks[tid] == ((0, 0, 0), 31.9)
    assert seen["centre"] == (0.3, 0.3) and seen["unit"] == h.ground_unit and seen["n"] >= 6
    # a crosspath read is kept per tiers; far from the table -> mismatch, recorded
    h.record_upgrade(tid, 2, 0)
    seen["value"] = 45.0
    r = h.measure_range(tid)
    assert not r and r.reason == "mismatch" and h.range_mismatches == [("dart_monkey", (0, 0, 1), 45.0, 40)]
    assert h.range_checks[tid] == ((0, 0, 1), 45.0)


# --- observation vector + grid image (PLAN 5.6) ---------------------------------------------
def test_obs_spec_matches_data():
    """data/obs.json's one-hot lists cover towers.json / heroes.json / rounds.json."""
    from bloons import obsvec
    spec = obsvec.load_spec()
    data = Path(__file__).resolve().parents[1] / "bloons/data"
    towers = json.loads((data / "towers.json").read_text())["towers"]
    heroes = json.loads((data / "heroes.json").read_text())["heroes"]
    rounds = json.loads((data / "rounds.json").read_text())["rounds"]
    assert set(spec["tower_kinds"]) == set(towers) and set(spec["heroes"]) == set(heroes)
    assert set().union(*(r["counts"] for r in rounds)) <= set(spec["bloon_types"])
    assert set(spec["maps"]) >= {p.name[:-5] for p in (data / "maps").glob("*.json") if p.name.count(".") == 1}
    assert len(set(spec["tower_kinds"])) == len(spec["tower_kinds"])


def test_obs_vector_and_grid(tmp_path):
    """to_vector: fixed length whatever is on the map, segments as layout() says; towers keep
    their slot after an earlier one is sold; grid_image: path / placeable / towers / ranges /
    bloons channels line up with what's placed."""
    import numpy as np
    import pytest
    from bloons import obsvec
    from bloons.detect import Bloon
    from bloons.state import Screen
    g, h = _round_harness(tmp_path)
    h.read_screen = lambda frame: Screen("game", "idle")
    spec = obsvec.load_spec()
    parts = {n: (a, b) for n, a, b in obsvec.layout()}
    s0 = h.observe()
    v0 = s0.to_vector()
    assert v0.dtype == np.float32 and v0.shape == (obsvec.layout()[-1][2],) and np.isfinite(v0).all()
    seg = lambda v, n: v[slice(*parts[n])]
    assert seg(v0, "map")[spec["maps"].index("monkey_meadow")] == 1 and seg(v0, "map").sum() == 1
    assert seg(v0, "money")[1] == 1 and seg(v0, "lives")[0] == 1.0          # 650 / 200 lives read
    assert seg(v0, "composition_counts")[spec["bloon_types"].index("red")] > 0 and seg(v0, "slot0").sum() == 0
    # two towers, sell the first: the second stays in slot 1
    a = h.register("dart_monkey", (0.25, 0.5), 170)
    b = h.register("tack_shooter", (0.6, 0.3), 260)
    h.record_upgrade(b, 0, 150)
    h.unregister(a)
    s = h.observe()
    assert s.slots[:2] == [None, b] and s.towers[0].slot == 1 and s.towers[0].footprint_units == 6
    v = s.to_vector()
    assert v.shape == v0.shape and seg(v, "slot0").sum() == 0
    s1 = seg(v, "slot1")
    assert s1[0] == 1 and s1[1 + spec["tower_kinds"].index("tack_shooter")] == 1
    k = 1 + len(spec["tower_kinds"])
    assert tuple(s1[k:k + 2]) == pytest.approx((0.6, 0.3)) and s1[k + 2] == pytest.approx(0.2)
    # a new tower takes the freed slot 0
    c = h.register("dart_monkey", (0.3, 0.7), 170)
    assert h.slots[:2] == [c, b]
    # grid: channel names, the path channel follows the track, towers / ranges near them
    img = h.observe().grid_image()
    names = obsvec.grid_channels()
    gg = spec["grid"]
    assert img.shape == (len(names), gg["ny"], gg["nx"]) and img.dtype == np.float32
    assert 0 < img[names.index("path")].mean() < 0.5 and img.min() >= 0 and img.max() <= 1
    cell = lambda x, y: (int((y - gg["box"][1]) / (gg["box"][3] - gg["box"][1]) * gg["ny"]),
                         int((x - gg["box"][0]) / (gg["box"][2] - gg["box"][0]) * gg["nx"]))
    j, i = cell(0.6, 0.3)
    assert img[names.index("towers"), j, i] > 0 and img[names.index("ranges"), j, i] > 0
    empty = s0.grid_image()
    pr6 = names.index("placeable_land_r6")
    assert img[pr6, j, i] < 0.25 and img[pr6, j, i] < empty[pr6, j, i]  # mostly covered by our tower
    assert img[names.index("bloons")].sum() == 0
    xc, yc = obsvec.cell_centre(i, j)
    assert cell(xc, yc) == (j, i)
    # bloons: counts in the vector, density in the grid, lead = progress along the path
    st = h.observe()
    p = st.path[len(st.path) // 2]
    st.bloons = [Bloon(float(p[0]), float(p[1]), "blue", 3, 2000, 0.1)]
    v = st.to_vector()
    assert seg(v, "bloons_counts")[spec["bloon_types"].index("blue")] > 0
    assert 0.3 < seg(v, "bloons_lead")[0] < 0.7 and seg(v, "bloons_lead")[2] == 1
    assert st.grid_image()[names.index("bloons")].sum() > 0
    assert obsvec.path_progress(st.path, [tuple(st.path[0]), tuple(st.path[-1])]) == pytest.approx([0, 1])


# --- review sweep 2026-11-06: round edges either Auto Start setting, live round checks, hero XP ---
def test_round_tracker_auto_start_with_idle_blip():
    """Auto Start on shows idle for ~1 s between rounds (RESEARCH 1.5). Whether the tracker sees
    that blip or not, each round fires round_ended once, and the HUD catching up late doesn't
    fire it again."""
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "idle", 1), ("game", "ff_on", 1), ("game", "ff_on", 1),
                    ("game", "idle", 1), ("game", "idle", 1),        # blip seen: round 1 ended
                    ("game", "ff_on", 1), ("game", "ff_on", 1),      # round 2 running, HUD lags
                    ("game", "ff_on", 2), ("game", "ff_on", 2),      # HUD catches up: no refire
                    ("game", "ff_on", 3), ("game", "ff_on", 3)])     # blip missed: round 2 ended
    assert [i for i, s in enumerate(out) if s.round_ended] == [4, 9]


def test_harness_start_round_after_unobserved_round_end(tmp_path):
    """round_active is only as fresh as the last observe(): start_round reads the button."""
    g, h = _round_harness(tmp_path)
    h.read_button = lambda frame: "idle"
    h.round_active = True                     # a round ran and ended without an observe()
    g.on_key = lambda game, k: setattr(h, "read_button", lambda frame: "ff_on")
    r = h.start_round()
    assert r and r.reason == "started" and g.calls == [("key", "space")]


def test_harness_round_only_ability_reads_button(tmp_path):
    g, h = _round_harness(tmp_path)
    tid = h.register("boomerang_monkey", (0.3, 0.5), 300)
    h._gain_ability(tid, "Turbo Charge", 45, between_rounds=False, path=1)
    h.ability_slots[0]["inst"][(tid, 1)] = -1.0     # ready
    h.read_bar = lambda content, n, side=None: []
    h.round_active = True                     # stale: the round has ended since
    h.read_button = lambda frame: "idle"
    assert h.use_ability(0).reason == "round_only" and not h.round_active and g.calls == []


def test_harness_hero_round_xp(tmp_path):
    """observe() credits the hero round-end XP and levels it up like the game, with abilities."""
    from bloons.hud import Hud
    from bloons.state import Screen
    g, h = _round_harness(tmp_path)
    h.hero_choice = "quincy"
    h.register_hero((0.75, 0.25), 540)
    seq = []
    h.read_screen = lambda frame: seq.pop(0) if seq else Screen("game", "idle")
    h.observe()
    for r in range(1, 8):                     # rounds 1-7: 40+60+...+160 = 700 XP
        g.hud = Hud(200, 650, r, 40)
        seq[:] = [Screen("game", "ff_on")] * 2 + [Screen("game", "idle")] * 2
        assert sum(h.observe().round_ended for _ in range(4)) == 1
    # Quincy: 180 to level 2, 460 more to level 3 (640) -> level 3 with 60 XP into it
    assert (h.hero.level, h.hero.xp_into_level) == (3, 60)
    assert "Rapid Shot" in [a["name"] for a in h.ability_slots]
    g2 = h.guard_state()
    from bloons import guards
    assert guards.hero_level_price(g2) == int((1000 - 60) * (1 - guards._training_discount(h.account)) + 0.5)


def test_harness_ability_drop_resets_bar_filter(tmp_path):
    from bloons.abilitybar import SlotRead
    g, h = _round_harness(tmp_path)
    a = h.register("boomerang_monkey", (0.3, 0.5), 300)
    b = h.register("boomerang_monkey", (0.5, 0.5), 300)
    h._gain_ability(a, "Turbo Charge", 45, between_rounds=False, path=1)
    h._gain_ability(b, "Other", 45, path=1)
    h.bar_filter.last[1] = (SlotRead(1, True, 0.9), 0.0)
    h.unregister(a)                           # slot 1 becomes slot 0
    assert h.bar_filter.last == {}


def test_harness_confirmed_spend_updates_money_now(tmp_path):
    """After a confirmed purchase the guards see the new money at once (HudFilter.accept)."""
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    h.refresh_hud()
    assert h.money == 650
    g.hud = Hud(200, 480, 1, 40)
    assert h._money_moved(650, -170) and h.money == 480


def test_harness_heli_tier_keeps_both_abilities(tmp_path):
    g, h = _round_harness(tmp_path)
    t = h.register("heli_pilot", (0.3, 0.5), 1000)
    for _ in range(4):
        h.record_upgrade(t, 1, 100)
    assert [a["name"] for a in h.ability_slots] == ["Redeploy", "Support Drop"]
    h.record_upgrade(t, 1, 100)
    assert [a["name"] for a in h.ability_slots] == ["Redeploy", "Support Drop", "Special Poperations"]


def test_harness_money_before_ignores_one_misread(tmp_path):
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    clock = [0.0]
    h.clock = lambda: clock[0]
    h.refresh_hud()                                        # filtered $650
    clock[0] = 10.0                                        # not fresh: _money_before reads again
    reads = [Hud(200, 50, 1, 40), Hud(200, 650, 1, 40)]    # '$650' clipped to '$50', then right
    g.read_hud = lambda frame=None: reads.pop(0) if reads else Hud(200, 650, 1, 40)
    assert h._money_before() == 650
    reads[:] = [Hud(200, 900, 1, 40), Hud(200, 905, 1, 40)]   # a real jump (round bonus): two reads agree
    clock[0] = 20.0
    assert h._money_before() == 905 and h.money == 905


# --- review sweep 3 (2026-11-06): debounced button reads, bounded money confirm, carried abilities ---
def test_harness_start_round_ignores_one_idle_misread(tmp_path):
    """Mid-round, one frame misread as idle must not make start_round press Space (it would
    toggle FF and still report 'started')."""
    g, h = _round_harness(tmp_path)
    reads = ["idle", "ff_off", "ff_off"]
    h.read_button = lambda frame: reads.pop(0) if reads else "ff_off"
    h.round_active = True
    assert h.start_round().reason == "round_active" and g.calls == []


def test_harness_sync_round_needs_two_reads(tmp_path):
    g, h = _round_harness(tmp_path)
    h.round_active = False
    reads = ["ff_off", "idle", "idle"]                   # one-frame misread, then agreement
    h.read_button = lambda frame: reads.pop(0) if reads else "idle"
    h._sync_round(object())
    assert h.round_active is False


def test_harness_money_moved_rejects_dropped_digit(tmp_path):
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    reads = [Hud(200, 357, 1, 40), Hud(200, 1187, 1, 40)]   # '$1,357' clipped, then the real spend
    g.read_hud = lambda frame=None: reads.pop(0) if reads else Hud(200, 1187, 1, 40)
    assert h._money_moved(1357, -170) and h.money == 1187


def test_harness_upgrade_keeps_carried_ability(tmp_path):
    g, h = _round_harness(tmp_path)
    t = h.register("heli_pilot", (0.3, 0.5), 1000)
    u = h.register("heli_pilot", (0.5, 0.5), 1000)
    for _ in range(4):
        h.record_upgrade(t, 1, 100)
        h.record_upgrade(u, 1, 100)
    h.ability_slots[0]["inst"][(t, 1)] = -1.0           # Redeploy ready
    h.record_upgrade(t, 1, 100)                          # 0-5-0 keeps Redeploy (now 15 s)
    a = h.ability_slots[0]
    assert a["name"] == "Redeploy" and a["inst"][(t, 1)] == -1.0
    assert h._cd(a, (t, 1)) == 15.0 and h._cd(a, (u, 1)) == 45.0   # the 0-4-0 heli keeps 45 s


def test_round_tracker_hud_jump_ends_each_round():
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 5), ("game", "ff_on", 5), ("game", "ff_on", 7)])
    assert out[-1].round_ended and out[-1].ended_rounds == (5, 6) and out[-1].ended_round == 6
    out = _feed(t, [("game", "ff_on", 40)])               # implausible jump: only the last round
    assert out[-1].ended_rounds == (39,)


def test_round_tracker_defeat_continue_ends_lost_round_once():
    """Continue after Defeat replays the lost round, which never finished: it ends once."""
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 29), ("game", "ff_on", 29), ("game", "ff_on", 30),   # 29 ended
                    ("defeat", None, 30), ("defeat", None, 30),
                    ("game", "ff_on", 30), ("game", "ff_on", 30),                         # Continue
                    ("game", "idle", 30), ("game", "idle", 30)])
    assert [s.ended_round for s in out if s.round_ended] == [29, 30]


def test_round_tracker_hud_one_behind_is_not_a_replay():
    """Auto Start off: round 6 ends before the filtered HUD shows 6; the HUD still saying 5
    afterwards mustn't make the next round count as 5 again."""
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 5), ("game", "ff_on", 5), ("game", "idle", 5), ("game", "idle", 5),
                    ("game", "ff_on", 5), ("game", "ff_on", 5), ("game", "idle", 5), ("game", "idle", 5),
                    ("game", "ff_on", 5), ("game", "ff_on", 5), ("game", "idle", 7), ("game", "idle", 7)])
    assert [s.ended_round for s in out if s.round_ended] == [5, 6, 7]


def test_harness_round_xp_credited_once(tmp_path):
    """A round end with no known round number (HUD never read) doesn't credit a round twice."""
    from bloons.state import RoundState
    g, h = _round_harness(tmp_path)
    h.hero_choice = "quincy"
    h.register_hero((0.75, 0.25), 540)
    h._hero_prev = True
    got = []
    h._hero_round_xp = got.append
    h.rounds_done = h._xp_round = 3
    rs = RoundState("game", False, True, None, None)
    h.tracker.update = lambda screen, hud=None: rs
    h.observe()
    h.rounds_done = 3                                    # stale count: the same guess again
    h.observe()
    assert got == [4]


# --- review sweep 4 (2026-11-06) ---
def test_harness_money_moved_confirms_bigger_move_on_two_reads(tmp_path):
    """Hero levels and sells are estimates: a bigger real move confirms once two reads agree,
    a lone dropped digit doesn't."""
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    reads = [Hud(200, 857, 1, 40), Hud(200, 860, 1, 40)]   # charged $500, guard said $50
    g.read_hud = lambda frame=None: reads.pop(0) if reads else Hud(200, 860, 1, 40)
    assert h._money_moved(1357, -50, max_move=600) and h.money in (857, 860)   # the filter's pick
    reads[:] = [Hud(200, 57, 1, 40)] * 4                  # the same misread repeated: beyond max_move
    assert not h._money_moved(1357, -300)
    reads[:] = [Hud(200, 357, 1, 40)] + [Hud(200, None, 1, 40)] * 3   # one misread only
    assert not h._money_moved(1357, -170)


def test_harness_money_before_falls_back_on_miss(tmp_path):
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    h.refresh_hud()
    g.read_hud = lambda frame=None: Hud(200, None, 1, 40)
    assert h._money_before() == 650


def test_harness_start_round_refuses_on_disagreeing_reads(tmp_path):
    g, h = _round_harness(tmp_path)
    reads = ["ff_on", "idle", "ff_on", "idle"]
    h.read_button = lambda frame: reads.pop(0) if reads else None
    h.round_active = False                               # stale
    assert h.start_round().reason == "no_button" and g.calls == []


def test_harness_sync_round_one_read_when_tracker_agrees(tmp_path):
    g, h = _round_harness(tmp_path)
    h.tracker._button = "ff_on"
    h._obs_t = h.clock()                                  # observe() just ran
    n = []
    h.read_button = lambda frame: n.append(1) or "ff_on"
    h._sync_round(object())
    assert h.round_active is True and len(n) == 1


def test_round_tracker_lagging_hud_round_still_ends():
    """Auto Start on: the next round starts while the HUD still shows the finished one, and the
    HUD isn't read again before the idle blip: that round's end still fires."""
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 4), ("game", "ff_on", 4),
                    ("game", "idle", 4), ("game", "idle", 4),        # round 4 ended
                    ("game", "ff_on", 4), ("game", "ff_on", 4),      # round 5, HUD lags
                    ("game", "idle", None), ("game", "idle", None)])  # HUD unread, blip
    assert [s.ended_round for s in out if s.round_ended] == [4, 5]


def test_round_tracker_big_hud_drop_is_not_a_replay():
    from bloons.state import RoundTracker
    t = RoundTracker()
    _feed(t, [("game", "ff_on", 29), ("game", "ff_on", 29), ("game", "ff_on", 30)])   # 29 ended
    out = _feed(t, [("game", "ff_on", 3), ("game", "ff_on", 30)])                    # misread 3
    assert not any(s.round_ended for s in out)


# --- review sweep 5 (2026-11-06) ---
def test_harness_sync_round_rereads_when_tracker_is_stale(tmp_path):
    g, h = _round_harness(tmp_path)
    clock = [100.0]
    h.clock = lambda: clock[0]
    h.tracker._button, h._obs_t = "ff_off", 0.0           # last observe() long ago
    reads = ["ff_off", "idle", "idle"]                    # one stale-matching misread, then the truth
    seen = []
    h.read_button = lambda frame: seen.append(1) or (reads.pop(0) if reads else "idle")
    h._sync_round(object())
    assert h.round_active is False and len(seen) == 3     # the first read isn't repeated on its frame


def test_harness_hero_placed_after_round_end_gets_no_xp(tmp_path):
    from bloons.state import RoundState
    g, h = _round_harness(tmp_path)
    got = []
    h._hero_round_xp = got.append
    rs = [RoundState("game", True, False, None, None)]
    h.tracker.update = lambda screen, hud=None: rs[0]
    h.observe()                                           # round 5 running, no hero yet
    h.hero_choice = "quincy"
    h.register_hero((0.75, 0.25), 540)                    # placed after round 5 ended, before observe
    rs[0] = RoundState("game", False, True, None, None, 5, 40, (5,))
    h.observe()
    assert got == [] and h.rounds_done == 5
    rs[0] = RoundState("game", False, True, None, None, 6, 40, (6,))
    h.observe()
    assert got == [6]


def test_harness_rounds_done_follows_lag_corrected_round(tmp_path):
    """Auto Start on: after round 5 ends the HUD still says 5 while round 6 runs; rounds_done stays 5."""
    from bloons.hud import Hud
    from bloons.state import Screen
    g, h = _round_harness(tmp_path)
    seq = [Screen("game", "ff_on")] * 2 + [Screen("game", "idle")] * 2 + [Screen("game", "ff_on")] * 2
    h.read_screen = lambda frame: seq.pop(0) if seq else Screen("game", "ff_on")
    g.hud = Hud(200, 650, 5, 40)
    for _ in range(6):
        st = h.observe()
    assert st.round_active and h.rounds_done == 5


# --- review sweep 6 (2026-11-06): repeated misreads, hero price either way, jump rollback ---
def test_harness_money_before_rejects_repeated_drop_misread(tmp_path):
    """Two quick reads of a still frame can repeat one misread: a drop we didn't cause stands
    only after an unconfirmed purchase."""
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    clock = [0.0]
    h.clock = lambda: clock[0]
    g.read_hud = lambda frame=None: Hud(200, 1357, 1, 40)
    h.refresh_hud()
    clock[0] = 10.0
    g.read_hud = lambda frame=None: Hud(200, 357, 1, 40)
    assert h._money_before() == 1357 and h.money == 1357
    h._money_unsure = True                                 # a purchase may have gone through
    assert h._money_before() == 357 and not h._money_unsure


def test_harness_money_moved_books_measured_hero_price(tmp_path):
    from bloons.hud import Hud
    g, h = _round_harness(tmp_path)
    reads = [Hud(200, 1057, 1, 40), Hud(200, 1058, 1, 40)]   # charged $300, estimate $1,000
    g.read_hud = lambda frame=None: reads.pop(0) if reads else Hud(200, 1058, 1, 40)
    assert h._money_moved(1357, -1000, max_move=1500, min_move=1) == 299
    g.read_hud = lambda frame=None: Hud(200, 1357, 1, 40)    # nothing moved
    assert h._money_moved(1357, -1000, min_move=1) == 0 and h._money_unsure


def test_harness_hero_placed_mid_round_earns_that_round(tmp_path):
    from bloons.result import ActionResult
    g, h = _round_harness(tmp_path)
    h.hero_choice = "quincy"
    h.refresh_hud()
    h._place = lambda *a, **k: ActionResult(True, "placed")
    h.read_button = lambda frame: "ff_on"
    assert h.place_hero(0.75, 0.25) and h._hero_prev is True


def test_harness_between_rounds_ability_skips_round_sync(tmp_path):
    g, h = _round_harness(tmp_path)
    tid = h.register("boomerang_monkey", (0.3, 0.5), 300)
    h._gain_ability(tid, "Anytime", 45, between_rounds=True, path=1)
    synced = []
    h._sync_round = lambda content: synced.append(1)
    h.read_bar = lambda content, n, side=None: []
    h.use_ability(0)
    assert synced == []


def test_round_tracker_jump_misread_rolls_back():
    """The HUD briefly shows 9 during round 5, then 5 again: rounds 5-8 haven't ended."""
    from bloons.state import RoundTracker
    t = RoundTracker()
    out = _feed(t, [("game", "ff_on", 5), ("game", "ff_on", 5), ("game", "ff_on", 9),
                    ("game", "ff_on", 5), ("game", "idle", 5), ("game", "idle", 5)])
    assert out[2].ended_rounds == (5, 6, 7, 8)
    assert out[-1].round_ended and out[-1].ended_round == 5      # round 5's real end still fires
