"""Drive the harness by hand. PLAN step 4.10.

  python tools/play.py [--map monkey_meadow] [--difficulty easy] [--mode standard] [--hero quincy]
                       [--spots data/maps/<map>.spots.json] [--no-account]      (Mac, game running)

A command prompt over BloonsHarness: every action prints its ActionResult and the HUD. The
harness only knows about towers placed through it, so start from a fresh game (`new`), or
run `forget` after restarting one by hand. Tower and hero names take any unique part
('dart', 'ace', 'sub'). Coordinates are normalized content coords (0-1). `cursor` prints
where the real mouse is, so hover a spot and read it off. `help <command>` explains one.
"""
import argparse
import cmd
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bloons.harness import HERO_TID, BloonsHarness, load_data   # noqa: E402

PATHS = {"top": 0, "mid": 1, "middle": 1, "bot": 2, "bottom": 2, "0": 0, "1": 1, "2": 2}


class UsageError(ValueError):
    pass


def resolve(name, choices):
    """A unique choice matching `name`: exact, else the only one with it as a whole word, else
    the only one starting with it, else the only one containing it ('dart' -> 'dart_monkey'
    rather than 'dartling_gunner', 'ace' -> 'monkey_ace', 'heli' -> 'heli_pilot')."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if n in choices:
        return n
    for test in (lambda c: n in c.split("_"), lambda c: c.startswith(n), lambda c: n in c):
        hits = [c for c in choices if test(c)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise UsageError(f"{name!r} could mean {', '.join(sorted(hits))}")
    raise UsageError(f"unknown {name!r}")


def num(s):
    try:
        return float(s)
    except ValueError:
        raise UsageError(f"not a number: {s!r}") from None


def tid_of(s):
    if s.lower() in ("hero", "h"):
        return HERO_TID
    try:
        return int(s)
    except ValueError:
        raise UsageError(f"not a tower id: {s!r}") from None


def points(args):
    if len(args) % 2:
        raise UsageError("points come in x y pairs")
    return [(num(args[i]), num(args[i + 1])) for i in range(0, len(args), 2)]


class Play(cmd.Cmd):
    intro = "Bloons harness REPL. 'help' lists commands, 'state' shows what the harness knows, 'quit' leaves."
    prompt = "bloons> "

    def __init__(self, harness, out=sys.stdout):
        super().__init__(stdout=out)
        self.h = harness

    # --- plumbing ---------------------------------------------------------------------------
    def say(self, *parts):
        print(*parts, file=self.stdout)

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except UsageError as e:
            self.say(f"usage: {e}")
        except KeyboardInterrupt:
            self.say("interrupted")
        except Exception as e:                     # a REPL shouldn't die on one bad command
            self.say(f"error: {type(e).__name__}: {e}")
        return False

    def emptyline(self):
        return False

    def default(self, line):
        self.say(f"unknown command {line.split()[0]!r} ('help' lists them)")

    def report(self, r):
        """Print an ActionResult and a fresh HUD read."""
        extra = "".join(f" {k}={v!r}" for k, v in (("cost", r.cost), ("value", r.value)) if v is not None)
        self.say(f"{'ok ' if r.ok else 'NO '} {r.reason}{extra}")
        self.h.refresh_hud()
        self.say(f"     {self.hud_line()}")

    def hud_line(self):
        hud = self.h.hud.current
        rnd = f"{hud.round}/{hud.rounds_total}" if hud.round is not None else "?"
        return f"${hud.money if hud.money is not None else '?'}  lives {hud.lives if hud.lives is not None else '?'}  round {rnd}"

    @staticmethod
    def args(line, n=None, at_least=None, usage=""):
        a = shlex.split(line)
        if (n is not None and len(a) != n) or (at_least is not None and len(a) < at_least):
            raise UsageError(usage)
        return a

    # --- looking ----------------------------------------------------------------------------
    def do_hud(self, line):
        """hud: read the HUD now."""
        self.h.refresh_hud()
        self.say(self.hud_line())

    def do_screen(self, line):
        """screen: which screen the game shows (game, pause, map_select, victory, ...)."""
        self.say(self.h.screen_kind())

    def do_state(self, line):
        """state: map / mode, HUD, the towers and hero the harness placed, the ability bar."""
        h = self.h
        self.say(f"{h.map_name}  {h.difficulty} {h.mode}  hero choice {h.hero_choice or '-'}  "
                 f"round_active {h.round_active}  ff {h.fast_forward}")
        self.say(self.hud_line())
        for t in h.registry.values():
            tgt = f"  targeting {t.targeting}" if t.targeting else ""
            self.say(f"  {t.tid:>3}  {t.kind:<18} {'-'.join(map(str, t.tiers))}  at ({t.xy[0]:.3f}, {t.xy[1]:.3f})"
                     f"  spent ${t.spent}{tgt}")
        if h.hero:
            self.say(f"  hero {h.hero.name} level {h.hero.level} at ({h.hero.xy[0]:.3f}, {h.hero.xy[1]:.3f})")
        if not h.registry and not h.hero:
            self.say("  (no towers placed through the harness)")
        for a in h.abilities:
            self.say(f"  ability slot {a.slot}: {a.name} ({'ready' if a.ready else 'cooling'}, tower {a.tid})")

    def do_actions(self, line):
        """actions: the legal actions right now with their costs (the RL action mask's source)."""
        acts = self.h.available_actions()
        for a in acts:
            self.say("  " + "  ".join(f"{k}={v}" for k, v in a.items()))
        if not acts:
            self.say("  (none; read the HUD first if money is unknown)")

    def do_observe(self, line):
        """observe: one GameState (HUD, round state, outcome, towers, upgrade options, abilities,
        placeable cells per mask class). Run it repeatedly to see round edges."""
        s = self.h.observe()
        self.say(f"screen {s.screen}  ${s.money}  lives {s.lives}  round {s.round}/{s.max_round}  "
                 f"active {s.round_active}  ended {s.round_ended}  ff {s.fast_forward}  outcome {s.outcome}")
        for t in s.towers:
            ups = "  ".join(f"{'x' if o.locked else ('$' + str(o.cost)) + ('' if o.affordable else '!')}"
                            for o in s.upgrade_options[t.tid])
            rng = f"range {t.range_units}{'' if t.range_exact else '~'}" if t.range_units is not None else "range ?"
            if t.range_measured is not None:
                rng += f" (seen {t.range_measured})"
            self.say(f"  {t.tid:>3}  {t.kind:<18} {'-'.join(map(str, t.tiers))}  {rng}  sell ${t.sell_value}  next [{ups}]")
        if s.hero:
            self.say(f"  hero {s.hero.name} level {s.hero.level}  sell ${s.hero.sell_value}")
        for a in s.abilities:
            self.say(f"  ability {a.slot}: {a.name} {'ready' if a.ready else f'cooling {a.cooldown_frac:.0%}'}")
        if s.round_composition:
            self.say(f"  round {s.next_round} (done {s.rounds_done}): {s.round_composition['counts']}  "
                     f"rbe {s.round_composition['rbe']}")
        if s.bloons is not None:
            from bloons.detect import summarize
            self.say(f"  bloons on screen: {summarize(s.bloons) or 'none'}")
        self.say("  free cells: " + "  ".join(f"{c} {int(m.ok.sum())}" for c, m in s.placeable.items()))

    def do_cursor(self, line):
        """cursor: the real mouse position in normalized coords (hover a spot, then run this)."""
        x, y = self.h.game.cursor()
        self.say(f"{x:.3f} {y:.3f}")

    def do_towers(self, line):
        """towers [part]: tower names (filtered by a part of the name)."""
        self.say("  ".join(k for k in self.h.towers if line.strip().lower() in k))

    # --- towers -----------------------------------------------------------------------------
    def do_place(self, line):
        """place <tower> <x> <y>: buy and place a tower ('place dart .4 .5'). Prints its tid."""
        a = self.args(line, 3, usage="place <tower> <x> <y>")
        self.report(self.h.place_tower(resolve(a[0], self.h.towers), num(a[1]), num(a[2])))

    def do_hero(self, line):
        """hero <x> <y>: place the hero chosen with --hero (or `pick`)."""
        a = self.args(line, 2, usage="hero <x> <y>")
        self.report(self.h.place_hero(num(a[0]), num(a[1])))

    def do_pick(self, line):
        """pick <hero>: tell the harness which hero this game uses (the one selected in the game)."""
        a = self.args(line, 1, usage="pick <hero>")
        self.h.hero_choice = resolve(a[0], self.h.heroes)
        self.say(f"hero choice: {self.h.hero_choice}")

    def do_sell(self, line):
        """sell <tid|hero>"""
        a = self.args(line, 1, usage="sell <tid|hero>")
        self.report(self.h.sell(tid_of(a[0])))

    def do_up(self, line):
        """up <tid> <top|mid|bot|0-2> [times]: buy the next tier on a path."""
        a = self.args(line, at_least=2, usage="up <tid> <top|mid|bot|0-2> [times]")
        if a[1].lower() not in PATHS:
            raise UsageError("path is top / mid / bot or 0 / 1 / 2")
        for _ in range(int(a[2]) if len(a) > 2 else 1):
            r = self.h.upgrade(tid_of(a[0]), PATHS[a[1].lower()])
            self.report(r)
            if not r:
                break

    def do_range(self, line):
        """range <tid>: measure a tower's range from its circle (select, read, close) and compare
        with the table; `observe` then shows it as 'seen'."""
        a = self.args(line, 1, usage="range <tid>")
        self.report(self.h.measure_range(tid_of(a[0])))

    def do_level(self, line):
        """level [times]: buy the hero's next level."""
        for _ in range(int(line) if line.strip() else 1):
            r = self.h.buy_hero_level()
            self.report(r)
            if not r:
                break

    def do_target(self, line):
        """target <tid> <mode>: targeting (first, last, close, strong, ...; Ace: circle,
        figure_infinite, figure_eight)."""
        a = self.args(line, 2, usage="target <tid> <mode>")
        self.report(self.h.set_targeting(tid_of(a[0]), a[1].lower()))

    def do_flight(self, line):
        """flight <tid> <circle|figure_infinite|figure_eight>: Monkey Ace flight path."""
        a = self.args(line, 2, usage="flight <tid> <pattern>")
        self.report(self.h.set_flight_path(tid_of(a[0]), a[1].lower()))

    def do_point(self, line):
        """point <tid> <x> <y>: Mortar target / Heli lock-in-place point."""
        a = self.args(line, 3, usage="point <tid> <x> <y>")
        self.report(self.h.set_target_point(tid_of(a[0]), num(a[1]), num(a[2])))

    def do_patrol(self, line):
        """patrol <tid> <x1> <y1> <x2> <y2>: Heli Pilot patrol points."""
        a = self.args(line, 5, usage="patrol <tid> <x1> <y1> <x2> <y2>")
        self.report(self.h.set_patrol_points(tid_of(a[0]), points(a[1:])))

    # --- abilities --------------------------------------------------------------------------
    def do_ability(self, line):
        """ability <slot> [x y]: use ability-bar slot (0-based, see `state`); x y for a target."""
        a = self.args(line, at_least=1, usage="ability <slot> [x y]")
        if len(a) == 3:
            self.report(self.h.place_ability_target(int(a[0]), num(a[1]), num(a[2])))
        elif len(a) == 1:
            self.report(self.h.use_ability(int(a[0])))
        else:
            raise UsageError("ability <slot> [x y]")

    def do_abilityof(self, line):
        """abilityof <tid|hero>: use a tower's ability from its panel."""
        a = self.args(line, 1, usage="abilityof <tid|hero>")
        self.report(self.h.use_ability_of(tid_of(a[0])))

    # --- rounds -----------------------------------------------------------------------------
    def do_go(self, line):
        """go: start the next round (Sandbox: send it from the bloon panel)."""
        self.report(self.h.start_round())

    def do_prompts(self, line):
        """prompts: clear popups (collect / continue / cancel). Victory and defeat are left alone."""
        done = self.h.handle_prompts()
        self.say(f"handled: {', '.join(done) if done else 'nothing'}  (now: {self.h.screen_kind()})")

    # --- games ------------------------------------------------------------------------------
    def do_new(self, line):
        """new [map] [easy|medium|hard] [standard|sandbox] [overwrite]: start a game from
        anywhere (defaults: the current ones). Without 'overwrite' a saved game is kept and
        the result is 'save_exists'."""
        a = [x.lower() for x in self.args(line)]
        overwrite = "overwrite" in a
        a = [x for x in a if x != "overwrite"]
        diff = next((x for x in a if x in ("easy", "medium", "hard")), None)
        mode = next((x for x in a if x in ("standard", "sandbox")), None)
        rest = [x for x in a if x not in (diff, mode)]
        maps = [p.stem for p in (Path(__file__).resolve().parents[1] / "bloons/data/maps").glob("*.json")
                if "." not in p.stem and not p.name.startswith("._")]      # not <map>.spots / .bg
        map_name = resolve(rest[0], maps) if rest else None
        self.report(self.h.new_game(map_name, diff, mode, overwrite_save=overwrite))

    def do_restart(self, line):
        """restart: restart this game (forgets its towers)."""
        self.report(self.h.restart())

    def do_home(self, line):
        """home: back to the home screen (leaving a game saves it)."""
        self.report(self.h.go_home())

    def do_forget(self, line):
        """forget: clear the harness's record of this game (after a restart done by hand)."""
        self.h.new_episode()
        self.h.refresh_hud()
        self.say(f"forgotten  {self.hud_line()}")

    # --- escape hatch -----------------------------------------------------------------------
    def do_py(self, line):
        """py <python>: evaluate with h (the harness) and g (the game) in scope ('py h.money')."""
        env = {"h": self.h, "g": self.h.game}
        try:
            self.say(repr(eval(line, env)))
        except SyntaxError:
            exec(line, env)

    def do_quit(self, line):
        """quit"""
        return True

    do_exit = do_quit
    do_EOF = do_quit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="monkey_meadow")
    ap.add_argument("--difficulty", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--mode", default="standard", choices=("standard", "sandbox"))
    ap.add_argument("--hero", default=None)
    ap.add_argument("--spots", default=None, help="learned-spots file (default: the map's own)")
    ap.add_argument("--no-account", action="store_true", help="ignore Monkey Knowledge in prices")
    a = ap.parse_args(argv)
    try:                                                    # any unique part, like the commands
        hero = resolve(a.hero, load_data("heroes.json")["heroes"]) if a.hero else None
    except UsageError as e:
        ap.error(f"--hero: {e}")
    from bloons.platform_mac import Game
    g = Game()
    h = BloonsHarness(g, a.map, a.difficulty, hero=hero, account=not a.no_account, spots_path=a.spots,
                      mode=a.mode)
    h.refresh_hud()
    Play(h).cmdloop()


if __name__ == "__main__":
    main()
