"""Unattended XP farm (docs/max.md B3). Steam BTD6 on Windows, game running full screen.

    python tools/farm.py [--map monkey_meadow] [--difficulty easy] [--hero quincy]
                         [--targets boomerang_monkey,bomb_shooter] [--games N]

Each game: new game -> bloons.farm.play_game (defence + all money into the targets) -> back
home -> spend each target's XP on unlocks (cheapest next tier first). Appends one line per
game to logs/farm.jsonl. Recovery: prompts, Escape, go home; the window gone -> relaunch via
Steam; 3 failed recoveries in a row -> screenshot, wait 5 min, retry.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloons import farm, xp  # noqa: E402
from bloons.harness import BloonsHarness  # noqa: E402
from bloons.platform import Game, GameNotFound  # noqa: E402

LOGS = ROOT / "logs"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "farm.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def leave_game(h, outcome):
    """From victory / defeat / a game back to the home screen."""
    for _ in range(8):
        kind = h.screen_kind()
        if kind == "home":
            return True
        if kind == "victory":
            h._click_ui("menus.victory_next")
            h._click_ui("menus.victory_home")
            continue
        r = h.go_home()
        if r:
            return True
        if not h.handle_prompts():
            h.game.key("escape")
            h.sleep(1.0)
    return False


def spend_xp(h, kinds):
    """Home -> Monkeys -> each target's XP screen: unlock what's affordable. -> {kind: [(path, cost)]}."""
    out = {}
    for kind in kinds:
        if not open_xp_screen(h, kind):
            log(f"xp: couldn't open {kind}'s upgrade screen")
            continue
        out[kind] = xp.unlock_all_affordable(h, log=lambda m, k=kind: log(f"xp {k}: {m}"))
        out[kind + "_state"] = xp.read(h.content()).__dict__
        h._click_ui("xp_screen.back")
        h._click_ui("xp_screen.back")
    return out


def open_xp_screen(h, kind):
    """Home -> Monkeys menu -> the tower's cell (ui.json monkeys_menu)."""
    mm = h.ui.get("monkeys_menu")
    if not mm or kind not in mm.get("cells", {}):
        return False
    if h.screen_kind() != "home" and not h.go_home():
        return False
    h._click_ui("monkeys_menu.open")
    h.game.click(*mm["cells"][kind])
    h.sleep(1.5)
    return h.screen_kind() == "xp_screen"


def recover(h, fails):
    try:
        if not h.game.reattach(timeout=2.0):
            log("window gone: relaunching through Steam")
            h.game.relaunch()
            h.sleep(30)
        if leave_game(h, None):
            return 0
    except Exception:
        log("recover: " + traceback.format_exc(limit=2))
    fails += 1
    if fails >= 3:
        f = h.game.grab()
        if f is not None:
            f.save_png(LOGS / f"stuck_{time.strftime('%Y%m%d_%H%M%S')}.png")
        log("3 failed recoveries: waiting 5 min")
        time.sleep(300)
        return 0
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="monkey_meadow")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--hero", default="quincy")
    ap.add_argument("--targets", default="boomerang_monkey,bomb_shooter")
    ap.add_argument("--defense", default="")
    ap.add_argument("--games", type=int, default=10 ** 6)
    a = ap.parse_args(argv)
    LOGS.mkdir(exist_ok=True)
    targets = [t for t in a.targets.split(",") if t]
    defense = [t for t in a.defense.split(",") if t]
    h = BloonsHarness(Game(), a.map, difficulty=a.difficulty, hero=a.hero)
    fails = 0
    for n in range(a.games):
        t0 = time.time()
        rec = {"game": n, "map": a.map, "difficulty": a.difficulty, "targets": targets}
        try:
            r = h.new_game(a.map, a.difficulty, "standard", overwrite_save=True)
            if not r:
                raise RuntimeError(f"new_game: {r.reason}")
            sp = farm.Spender(h, targets, defense, log=log)
            rec["outcome"] = farm.play_game(h, sp, log=log)
            rec["round"] = h.observe().round
            rec["towers"] = [(t.kind, t.tiers, t.spent) for t in h.registry.values()]
            leave_game(h, rec["outcome"])
            rec["xp"] = spend_xp(h, targets)
            fails = 0
        except GameNotFound:
            rec["error"] = "game_not_found"
            fails = recover(h, fails)
        except Exception as e:
            rec["error"] = repr(e)
            log(traceback.format_exc())
            fails = recover(h, fails)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        with open(LOGS / "farm.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"game {n}: {rec.get('outcome')} in {rec['minutes']} min {rec.get('error', '')}")


if __name__ == "__main__":
    main()
