"""Ability bar reader (PLAN 5.4, portable): ready / cooldown fraction per bar icon from pixels.

While an ability cools down, its icon carries a thick red-orange ring (an annulus inside the
icon, fully on screen). Right after a use the ring is full; it shrinks **clockwise**, so what
is left is an arc that always ends at 12 o'clock, and its angle is the fraction of the
cooldown remaining (linear in time, measured live 2026-11-06: residual <= 0.017 over 45 s).
A ring-less icon is ready. A slot whose icon area is dark (mean V < icon_min_v: an
overlay, a black test frame) reads seen=False and says nothing. A full ring has a thin white tick at 12 o'clock, so the arc never
quite reaches 1: an arc that covers all but the tick reads 1.0.

The arc is read by sampling `n` angles on a few radii of the annulus and counting the
on-samples contiguous back from 12 o'clock (anticlockwise), which ignores stray orange
pixels elsewhere on the icon (Benjamin's headband, the Fan Club icon).

Layout lives in data/ui.json `ability_bar` (ring_y, ring_radii, ...), the colour band here;
slot x is `ui.ability_slot(i)`.

Seen once and not explained for sure: round-only abilities (Turbo Charge, Fan Club) lost
their ring for ~1.6 s just after a Sandbox round was sent, then came back where they were
(their cooldowns lagged ~2 s, Biohack's didn't: probably frozen between rounds).
`BarFilter` therefore doesn't believe a jump from mid-cooldown to ready until it has lasted
`confirm_s`.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import ui
from .ui import UI

# The ring's red-orange (OpenCV HSV, H 0-180): BGR ~(38, 125, 255) = H 12, S 217, V 255;
# icons, grass and leaves stay outside it (2026-11-06).
RING_HUE = (14, 172)        # hue <= 14 or >= 172 (wraps through red)
RING_MIN_S, RING_MIN_V = 140, 170


@dataclass(frozen=True)
class SlotRead:
    slot: int
    ring: bool          # a cooldown ring is visible
    frac: Optional[float]   # remaining cooldown fraction, 0 = ready (no ring), 1 = just used;
                            # None = no icon visible there (dark: overlay, black frame)
    seen: bool = True


def _cfg():
    return UI["ability_bar"]


def ring_frac(hsv, cx, cy, radii, n=None):
    """Remaining fraction of the ring centred at (cx, cy) px in an HSV image: on-samples
    contiguous back from 12 o'clock, / n. 0.0 when there's no ring there."""
    a = _cfg()
    n = n or a["ring_samples"]
    (h0, h1), smin, vmin = RING_HUE, RING_MIN_S, RING_MIN_V
    H, W = hsv.shape[:2]
    need = (len(radii) + 1) // 2
    run = 0
    for k in range(n - 1, -1, -1):          # anticlockwise from just before 12 o'clock
        ang = 2 * np.pi * k / n             # clockwise from 12 o'clock
        hits = 0
        for r in radii:
            x, y = int(round(cx + r * np.sin(ang))), int(round(cy - r * np.cos(ang)))
            if 0 <= x < W and 0 <= y < H:
                h, s, v = hsv[y, x]
                hits += bool((h <= h0 or h >= h1) and s >= smin and v >= vmin)
        if hits < need:
            break
        run += 1
    if run >= n - a["full_gap"]:            # all but the white tick at 12 o'clock
        return 1.0
    return run / n


def read_bar(img, n_slots, panel_side=None):
    """[SlotRead] for bar icons 0..n_slots-1 in a content frame (BGR, any scale). With the
    panel open on the left the bar is shifted (ui.ability_slot)."""
    a = _cfg()
    if img is None or n_slots <= 0:
        return []
    H, W = img.shape[:2]
    y0 = max(int((a["ring_y"] - 2 * max(a["ring_radii"])) * H), 0)
    hsv = cv2.cvtColor(img[y0:], cv2.COLOR_BGR2HSV)
    radii = [r * W for r in a["ring_radii"]]
    out = []
    rmax = max(radii)
    for i in range(n_slots):
        x, _ = ui.ability_slot(i, panel_side)
        cx, cy = x * W, a["ring_y"] * H - y0
        disk = hsv[max(int(cy - rmax), 0):int(cy + rmax), max(int(cx - rmax), 0):int(cx + rmax), 2]
        if disk.size == 0 or float(disk.mean()) < a["icon_min_v"]:
            out.append(SlotRead(i, False, None, False))
            continue
        f = ring_frac(hsv, cx, cy, radii)
        ring = f >= a["min_frac"]
        out.append(SlotRead(i, ring, round(f, 3) if ring else 0.0))
    return out


class BarFilter:
    """Smooths reads over time per slot. The ring can vanish mid-cooldown for a while (seen
    for ~1.6 s around a Sandbox round start, probably round-only abilities between rounds),
    so a ring-less read is held back while the slot should still be cooling down, until it
    has lasted `confirm_s`; meanwhile the slot reports the extrapolated fraction.

    "Should still be cooling down": with the slot's cooldown known (`cooldowns[slot]` in
    seconds), the last ring read run forward in time leaves more than `slack_s`; without
    it, the last ring read had more than `jump_frac` left and is younger than `confirm_s`."""

    def __init__(self, jump_frac=None, confirm_s=None, slack_s=None):
        a = _cfg()
        self.jump_frac = a["filter_jump_frac"] if jump_frac is None else jump_frac
        self.confirm_s = a["filter_confirm_s"] if confirm_s is None else confirm_s
        self.slack_s = a["filter_slack_s"] if slack_s is None else slack_s
        self.reset()

    def reset(self):
        self.last = {}          # slot -> (SlotRead accepted, when)
        self.gone_since = {}    # slot -> time a held-back ring-less read started

    def update(self, reads, now, cooldowns=None):
        out = []
        for r in reads:
            if not r.seen:
                out.append(r)
                continue
            prev, t_prev = self.last.get(r.slot, (None, None))
            held = None
            if not r.ring and prev is not None and prev.ring:
                cd = (cooldowns or {}).get(r.slot) if isinstance(cooldowns, dict) else (
                    cooldowns[r.slot] if cooldowns is not None and r.slot < len(cooldowns) else None)
                if cd:
                    left = prev.frac * cd - (now - t_prev)
                    if left > self.slack_s:
                        held = SlotRead(r.slot, True, round(left / cd, 3))
                elif prev.frac > self.jump_frac and now - t_prev < self.confirm_s:
                    held = prev
            if held is not None:
                t0 = self.gone_since.setdefault(r.slot, now)
                if now - t0 < self.confirm_s:
                    out.append(held)
                    continue
            self.gone_since.pop(r.slot, None)
            self.last[r.slot] = (r, now)
            out.append(r)
        for s in [s for s in self.last if s >= len(reads)]:   # slots that went away
            self.last.pop(s)
            self.gone_since.pop(s, None)
        return out
