"""PLAN.md step 0.2 — throwaway Mac feasibility check. Not part of the harness.

Run from Terminal (needs Screen Recording + Accessibility for Terminal):
    python3 scripts/feasibility_mac.py
Open BTD6+ and start a game on any map before step 3.
"""
import subprocess
import time

import mss
import mss.tools
import Quartz as Q

KEY_Q = 12  # macOS virtual keycode for Q (ANSI layout)


def find_window():
    # All windows, not just on-screen ones: full screen puts the game in its own Space,
    # so it's off-screen while the terminal is in front. The game also owns several
    # 33px menu-bar strips; the largest window is the game view.
    wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionAll, Q.kCGNullWindowID)
    game = [w for w in wins
            if "bloons" in (w.get("kCGWindowOwnerName") or "").lower() and w.get("kCGWindowLayer") == 0]
    if not game:
        return None
    return max(game, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])


def click(x, y, post):
    for t in (Q.kCGEventMouseMoved, Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp):
        post(Q.CGEventCreateMouseEvent(None, t, (x, y), Q.kCGMouseButtonLeft))
        time.sleep(0.05)


def key(code, post):
    for down in (True, False):
        post(Q.CGEventCreateKeyboardEvent(None, code, down))
        time.sleep(0.05)


def activate(owner, wait=1.0):
    subprocess.run(["osascript", "-e", f'tell application "{owner}" to activate'], check=False)
    time.sleep(wait)


def grab(mon):
    with mss.MSS() as sct:
        return sct.grab(mon)


def changed(a, b, step=97):
    """Fraction of sampled BGRA bytes that differ noticeably between two grabs."""
    ra, rb = a.raw, b.raw
    idx = range(0, min(len(ra), len(rb)), step)
    return sum(abs(ra[i] - rb[i]) > 30 for i in idx) / max(len(idx), 1)


def ask(q):
    return input(f"{q} [y/n] ").strip().lower().startswith("y")


def main():
    w = find_window()
    if not w:
        raise SystemExit("No on-screen window with 'Bloons' in its owner name. Is the game open?")
    b, pid, owner = w["kCGWindowBounds"], w["kCGWindowOwnerPID"], w["kCGWindowOwnerName"]
    print(f"1. window: owner={owner!r} pid={pid} bounds(points)={dict(b)}")
    if not Q.CGPreflightScreenCaptureAccess():
        raise SystemExit("No Screen Recording permission for this terminal's app (e.g. Visual Studio Code). "
                         "Grant it in System Settings > Privacy & Security, then quit and reopen that app.")
    if not Q.CGPreflightPostEventAccess():
        Q.CGRequestPostEventAccess()  # shows the system prompt / adds the app to the list
        raise SystemExit("No Accessibility permission for this terminal's app, so synthetic input would be "
                         "silently dropped. Enable it in System Settings > Privacy & Security > Accessibility, "
                         "then quit and reopen that app.")

    activate(owner, 1.5)  # let the Space switch finish before capturing
    mon = {"left": int(b["X"]), "top": int(b["Y"]), "width": int(b["Width"]), "height": int(b["Height"])}
    img = grab(mon)
    mss.tools.to_png(img.rgb, img.size, output="frame.png")
    print(f"2. captured {img.size} px -> frame.png (pixels/points = {img.size[0] / b['Width']:.1f})")

    results = {}
    hid = lambda e: Q.CGEventPost(Q.kCGHIDEventTap, e)
    to_pid = lambda e: Q.CGEventPostToPid(pid, e)
    methods = (("HID tap", hid), ("PostToPid", to_pid))

    print("\n3. Key test. Have the game open in a round with nothing selected.")
    for name, post in methods:
        input(f"   [{name}] Press Enter, then hands off for 3 s (the script brings the game to the front)...")
        activate(owner)  # focus must move to the game AFTER Enter, or the key lands in the terminal
        before = grab(mon)
        key(KEY_Q, post)
        time.sleep(0.7)
        d = changed(before, grab(mon))
        results[f"Q key via {name} selected Dart Monkey"] = ask(
            f"   [{name}] screen changed {d:.1%}. Did Q select the Dart Monkey?")
        activate(owner, 0.5)
        key(53, post)  # Esc to deselect; harmless if ignored
        time.sleep(0.3)

    # A click on empty map does nothing visible, so test clicks by placing a monkey on grass.
    print("\n4. Click test. Each one places a Dart Monkey ($200 each on Medium).")
    spots = ((0.55, 0.20), (0.30, 0.30))  # grass on Monkey Meadow, as fractions of the window
    for (name, post), (fx, fy) in zip(methods, spots):
        input(f"   [{name}] Click into the game, press Q yourself so the Dart Monkey is selected, "
              "then come back here and press Enter. Hands off for 3 s...")
        activate(owner)
        before = grab(mon)
        click(b["X"] + fx * b["Width"], b["Y"] + fy * b["Height"], post)
        time.sleep(0.7)
        d = changed(before, grab(mon))
        results[f"click via {name} placed tower"] = ask(
            f"   [{name}] screen changed {d:.1%}. Was a Dart Monkey placed?")

    print("\nResults (paste into RESEARCH.md):")
    for k, v in results.items():
        print(f"- {k}: {'YES' if v else 'NO'}")
    if not any(results.values()):
        print("GATE FAILED: synthetic input ignored. Re-plan with the HID fallback (PLAN.md 0.2).")


if __name__ == "__main__":
    main()
