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
from bloons import cli  # noqa: E402
cli.pre_parse()   # --platform mac|win; must precede every other bloons import

from bloons import farm, xp  # noqa: E402
from bloons.harness import BloonsHarness  # noqa: E402
from bloons.platform import Game, GameNotFound  # noqa: E402

LOGS = ROOT / "logs"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "farm.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def leave_game(h):
    """Back to the home screen from wherever the game ended (go_home knows victory, defeat,
    prompts); Escape as a last resort."""
    for _ in range(4):
        if h.go_home():
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
        out[kind + "_state"] = xp.resolve_tier5(h, xp.read_settled(h)).__dict__
        h._click_ui("xp_screen.back")
    return out


def open_xp_screen(h, kind):
    """Home -> Monkeys -> the tower's tab and cell (ui.json monkeys_menu) -> its XP screen."""
    mm = h.ui["monkeys_menu"]
    cell = mm["cells"].get(kind)
    if cell is None or not h.go_home():
        return False
    h._click_ui("monkeys_menu.open")
    h.game.click(*mm["tabs"][cell["tab"]])
    h.sleep(1.0)
    h.game.click(*cell["xy"])
    h.sleep(1.5)
    return h.screen_kind() == "xp_screen"


def tiers_left(progress, kind):
    """Tiers still locked (0 = fully unlocked, its XP is useless)."""
    return 15 - sum(progress[kind].get("unlocked", []))


def plan(h, a, duds=()):
    """Survey the account (logs/xp_progress.json) and pick targets / defence. `duds`: targets
    that got nothing placed in a game (skipped until the farm restarts)."""
    progress = farm.survey(h, log=log)
    if not any(p.get("unlocked_tower") for p in progress.values()):
        raise RuntimeError("survey saw nothing (game not on the home screen?)")
    (LOGS / "xp_progress.json").write_text(json.dumps(progress, indent=1))
    done = {k: sum(p.get("unlocked", [])) for k, p in progress.items() if p.get("unlocked_tower")}
    defense = ([t for t in a.defense.split(",") if t] if a.defense != "auto" else
               [k for k, _ in sorted(done.items(), key=lambda kv: -kv[1])   # strongest towers that
                if tiers_left(progress, k) and farm.placeable_here(h, k)][:2])  # still need XP: maxed
                                                                             # ones would waste it
    targets = ([t for t in a.targets.split(",") if t] if a.targets != "auto" else
               farm.choose_targets(h, progress, n=3, exclude=list(defense) + list(duds)))
    log(f"plan: targets {targets}, defence {defense}; tiers unlocked {done}")
    return targets, defense


def recover(h, fails):
    try:
        if not h.game.reattach(timeout=2.0):
            log("window gone: relaunching through Steam")
            h.game.relaunch()
            h.sleep(30)
            h.game.click(0.5, 0.9)       # title screen START (ui-free: only seen after a launch)
            h.sleep(10)
        if leave_game(h):
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
    cli.add_argument(ap)
    ap.add_argument("--map", default="monkey_meadow")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--hero", default="quincy")
    ap.add_argument("--targets", default="auto", help="comma list, or auto (survey + choose_targets)")
    ap.add_argument("--defense", default="auto", help="comma list, or auto (the 2 most unlocked towers)")
    ap.add_argument("--survey-every", type=int, default=5)
    ap.add_argument("--games", type=int, default=10 ** 6)
    a = ap.parse_args(argv)
    if cli.calib.PLATFORM != "win":
        sys.exit("tools/farm.py is Steam-only (XP farm on Dom's Steam account, docs/max.md Part B): "
                 "run it with --platform win on the Windows PC")
    log(cli.describe())
    LOGS.mkdir(exist_ok=True)
    h = BloonsHarness(Game(), a.map, difficulty=a.difficulty, hero=a.hero)
    targets = [t for t in a.targets.split(",") if t and t != "auto"]
    defense = [t for t in a.defense.split(",") if t and t != "auto"]
    fails = 0
    duds, replan = set(), False
    for n in range(a.games):
        if (a.targets == "auto" or a.defense == "auto") and (n % a.survey_every == 0 or replan):
            replan = False
            try:
                targets, defense = plan(h, a, duds)
            except Exception:
                log("plan: " + traceback.format_exc(limit=3))
                fails = recover(h, fails)
                replan = True
                continue
            if not targets:
                log("nothing left to farm on this map: done")
                break
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
            placed = {k for k, _, _ in rec["towers"]}
            new_duds = [k for k in targets if k not in placed] if (rec["round"] or 0) >= 10 else []
            if new_duds and a.targets == "auto":
                log(f"targets never placed: {new_duds}: replanning without them")
                duds.update(new_duds)
                replan = True
            leave_game(h)
            rec["xp"] = spend_xp(h, list(dict.fromkeys(targets + defense)))
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
