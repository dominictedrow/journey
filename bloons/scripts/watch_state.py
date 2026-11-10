"""Live check for step 1.5: watch a game and log what state.py / hud.py see. Sends NO input,
so Dom can play by hand (or a scripted run can drive) while this runs alongside.

    .venv/bin/python scripts/watch_state.py OUT_DIR [--hz 4]

RoundTracker handles Auto Start on or off by itself.

Every loop: grab -> read_screen + read_hud -> HudFilter -> RoundTracker. Prints a line
whenever the screen kind, button, panel, round, outcome or round_ended changes, and saves
that frame as OUT_DIR/NNNN_<kind>_<button>.png (full 2x frame, so it can become a fixture).
Also appends every loop as JSON lines to OUT_DIR/log.jsonl. Ctrl-C to stop; prints a
summary of the button / screen kinds seen and the lowest template score of each match.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloons.hud import HudFilter, parse_hud  # noqa: E402
from bloons.platform_mac import Game  # noqa: E402
from bloons.state import RoundTracker, read_screen  # noqa: E402
from bloons.vision import SPEC  # noqa: E402


def main():
    argv = sys.argv[1:]
    hz = 4.0
    if "--hz" in argv:
        i = argv.index("--hz")
        hz = float(argv[i + 1])
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    out = Path(args[0] if args else "watch_out")
    out.mkdir(parents=True, exist_ok=True)
    g, hf, rt = Game(), HudFilter(), RoundTracker()
    kinds, buttons, min_hit = Counter(), Counter(), {}
    last_key, n = None, 0
    print(f"watching pid {g.pid}, {hz} Hz, frames -> {out}/ (Ctrl-C to stop)")
    try:
        with open(out / "log.jsonl", "a") as log:
            while True:
                t0 = time.monotonic()
                frame = g.grab()
                if frame is None:
                    print("no frame (minimized or screen locked)")
                    time.sleep(1)
                    continue
                sc = read_screen(frame.content_bgr())
                raw = g.read_hud_raw(frame)
                hud = hf.update(parse_hud(raw))
                st = rt.update(sc, hud)
                kinds[sc.kind] += 1
                buttons[sc.button] += 1
                for name, score in sc.scores.items():
                    if score >= SPEC["templates"][name]["threshold"]:
                        min_hit[name] = min(min_hit.get(name, 1.0), score)
                rec = {"t": round(time.time(), 2), "screen": sc.kind, "button": sc.button, "panel": sc.panel,
                       "hud_raw": raw, "lives": hud.lives, "money": hud.money, "round": st.round,
                       "active": st.round_active, "ended": st.round_ended, "ff": st.fast_forward,
                       "outcome": st.outcome, "scores": sc.scores}
                log.write(json.dumps(rec) + "\n")
                key = (sc.kind, sc.button, sc.panel, st.round, st.outcome, st.round_ended)
                if key != last_key:
                    n += 1
                    name = f"{n:04d}_{sc.kind}_{sc.button}.png"
                    frame.save_png(out / name)
                    print(f"{time.strftime('%H:%M:%S')} {sc.kind:16} btn={sc.button!s:6} panel={sc.panel!s:5} "
                          f"round={st.round}/{st.rounds_total} lives={hud.lives} ${hud.money} "
                          f"active={st.round_active} ended={st.round_ended} ff={st.fast_forward} "
                          f"outcome={st.outcome}  -> {name}")
                    last_key = key
                time.sleep(max(0.0, 1 / hz - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    print("\nscreens:", dict(kinds))
    print("buttons:", dict(buttons))
    print("lowest score among matches (threshold 0.8):", {k: round(v, 3) for k, v in sorted(min_hit.items())})


if __name__ == "__main__":
    main()
