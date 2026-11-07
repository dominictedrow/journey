"""Live smoke test for bloons/platform_mac.py (step 1.1). Needs a BTD6+ game open.

    .venv/bin/python scripts/smoke_platform.py [OUT_DIR] [--input]

Without --input it only captures and reads the HUD (the game can stay in the background).
With --input it brings the game to the front, presses Q (Dart Monkey), saves a frame,
then clicks the placement cancel button. Nothing is bought.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloons.platform_mac import UI, Game  # noqa: E402

out = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "."))
g = Game()
print(f"window pid={g.pid} wid={g.wid} bounds={g.bounds} size_ok={g.size_ok()} front={g.is_front()}")

t0 = time.perf_counter(); f = g.grab(); t1 = time.perf_counter()
print(f"grab: {f.width}x{f.height} px, scale {f.scale:.1f}, {1000 * (t1 - t0):.1f} ms")
f.save_png(out / "smoke_frame.png")
t0 = time.perf_counter(); hud = g.read_hud(f); t1 = time.perf_counter()
print(f"HUD: {hud} ({1000 * (t1 - t0):.0f} ms for 3 fields; the first call loads the model)")
t0 = time.perf_counter(); hud = g.read_hud(f); t1 = time.perf_counter()
print(f"HUD again: {1000 * (t1 - t0):.0f} ms")

if "--input" in sys.argv:
    t0 = time.perf_counter(); ok = g.ensure_front(); t1 = time.perf_counter()
    print(f"ensure_front: {ok} in {1000 * (t1 - t0):.0f} ms")
    g.move(0.68, 0.18)
    g.key("q"); time.sleep(UI["timing_s"]["after_tower_key"])
    g.move(0.681, 0.18); time.sleep(0.2)
    g.grab().save_png(out / "smoke_held.png")
    g.click(*UI["buttons"]["cancel_placement"]); time.sleep(0.4)
    g.grab().save_png(out / "smoke_cancelled.png")
    print("input: pressed q, saved smoke_held.png, clicked cancel, saved smoke_cancelled.png")
