"""HUD text -> numbers: lives, money, round. Portable (no OS calls); OCR itself lives in
platform_mac.py, which hands the raw strings here.

Two layers:
- `parse_*` / `parse_hud`: one OCR reading -> numbers, or None for anything that doesn't
  look like a clean HUD value. They are deliberately strict: a None is cheap, a wrong
  number reaches the agent.
- `HudFilter`: keeps the last accepted values across frames and only accepts a change
  once it has been read the same way `confirm` times in a row.

What the fixtures showed (RESEARCH.md, step 1.3):
- Vision (accurate, no language correction) reads clean HUDs exactly: '200', '$1,357', '39/40'.
- The round box can pick up the 'ROUND' label as a low-confidence junk line ('KOMID 1/40').
- With the upgrade panel open the HUD shifts: money then reads e.g. '20' (no '$', half the
  number cut off) and round reads junk ('TINYN'). Requiring the '$' and a well-formed
  'N/M' rejects both.
"""
import re
from dataclasses import dataclass
from typing import Optional

# Glyphs OCR can confuse with digits. Applied only to tokens that are otherwise numeric,
# so words like 'KOMID' are left alone.
_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                             "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
_NUMERIC_CHARS = set("0123456789,$/ ")

# Hard bounds; anything outside is a misread. Lives can grow past the 200 start
# (Banana Farm, abilities), money can get large late game, freeplay rounds go far past 100.
LIVES_MAX = 9_999
MONEY_MAX = 99_999_999
ROUND_MAX = 1_000

_MONEY_RE = re.compile(r"\$(\d{1,3}(?:[,.]\d{3})+|\d+)")   # Vision sometimes reads , as .
# Vision sometimes reads the '$' as a middle dot ('·9,998,181', step 4.3). Accepted only with
# well-formed thousands separators, which a clipped HUD reading ('20') never has.
_MONEY_DOT_RE = re.compile(r"[·•](\d{1,3}(?:[,.]\d{3})+)")
_ROUND_RE = re.compile(r"(\d{1,4})/(\d{1,4})")


def _tokens(text):
    """Split OCR text into whitespace tokens and fix digit confusables in numeric-looking ones."""
    out = []
    for tok in (text or "").split():
        if any(c.isdigit() for c in tok):
            fixed = tok.translate(_CONFUSABLE)
            if set(fixed) <= _NUMERIC_CHARS:
                tok = fixed
        out.append(tok)
    return out


def _single(matches):
    """The one candidate value, or None when there are zero or several different ones."""
    vals = set(matches)
    return vals.pop() if len(vals) == 1 else None


def parse_lives(text) -> Optional[int]:
    """'200' -> 200. A bare number of 1-4 digits; None for anything else."""
    vals = [int(t) for t in _tokens(text) if re.fullmatch(r"\d{1,4}", t)]
    v = _single(vals)
    return v if v is not None and 0 <= v <= LIVES_MAX else None


def parse_money(text) -> Optional[int]:
    """'$1,357' -> 1357. The '$' is required (without it the box is probably clipping a
    shifted HUD); thousands separators must be well-formed if present. Vision sometimes
    reads them as dots ('$9.999.609', seen in game): money never has cents, so a dot
    followed by exactly three digits is a separator too. A '$' read as '·' counts when the
    separators are well-formed ('·9,998,181')."""
    vals = []
    for t in _tokens(text):
        m = _MONEY_RE.fullmatch(t) or _MONEY_DOT_RE.fullmatch(t)
        if m:
            vals.append(int(re.sub(r"[,.]", "", m.group(1))))
    v = _single(vals)
    return v if v is not None and v <= MONEY_MAX else None


def parse_round(text) -> Optional[tuple]:
    """'39/40' -> (39, 40). Ignores junk tokens such as the OCR'd 'ROUND' label."""
    vals = []
    for t in _tokens(text):
        m = _ROUND_RE.fullmatch(t)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            if 1 <= cur <= total <= ROUND_MAX:
                vals.append((cur, total))
    return _single(vals)


@dataclass(frozen=True)
class Hud:
    lives: Optional[int] = None
    money: Optional[int] = None
    round: Optional[int] = None        # current round, 1-based
    rounds_total: Optional[int] = None  # e.g. 40 on Easy

    @property
    def complete(self):
        return None not in (self.lives, self.money, self.round, self.rounds_total)


def parse_hud(raw) -> Hud:
    """{'lives': '200', 'money': '$650', 'round': 'KOMID 1/40'} -> Hud(200, 650, 1, 40)."""
    r = parse_round(raw.get("round", ""))
    return Hud(parse_lives(raw.get("lives", "")), parse_money(raw.get("money", "")),
               r[0] if r else None, r[1] if r else None)


class HudFilter:
    """Smooths per-frame HUD readings over time.

    Each field holds its last accepted value. A different reading replaces it once
    `confirm` consecutive readings (default 2) agree with each other; the newest one is
    taken. For lives and round "agree" means equal. Money changes on almost every read
    during a round (pop income), so two reads rarely match exactly; for money "agree"
    means within MONEY_TOL of the previous reading (max $20 or 10%). A clipped or
    dropped digit ($1,357 -> $357) is far off, so it's still held back until it repeats.

    Found live (step 1.5): with exact matching only, money froze for whole rounds.

    A None reading (field unreadable this frame) never overwrites a value; it's counted in
    `misses` so callers can tell a stale HUD from a live one. The first value ever read
    is accepted immediately. Call `reset()` when a new game starts.
    """
    FIELDS = ("lives", "money", "round", "rounds_total")
    MONEY_TOL = (20, 0.10)   # (absolute $, fraction of the previous reading)

    def __init__(self, confirm=2):
        self.confirm = confirm
        self.reset()

    def reset(self):
        self.value = dict.fromkeys(self.FIELDS)
        self._pending = dict.fromkeys(self.FIELDS)  # (last reading, consecutive agreeing reads)
        self.misses = dict.fromkeys(self.FIELDS, 0)

    def agrees(self, field, a, b):
        """Two readings of `field` count as the same value (money: within MONEY_TOL)."""
        if field != "money":
            return a == b
        return abs(a - b) <= max(self.MONEY_TOL[0], self.MONEY_TOL[1] * max(a, b))

    def update(self, reading: Hud) -> Hud:
        for f in self.FIELDS:
            new = getattr(reading, f)
            if new is None:
                self.misses[f] += 1
                continue
            self.misses[f] = 0
            if self.value[f] is None or new == self.value[f]:
                self.value[f], self._pending[f] = new, None
                continue
            last, seen = self._pending[f] or (None, 0)
            seen = seen + 1 if last is not None and self.agrees(f, last, new) else 1
            if seen >= self.confirm:
                self.value[f] = new   # keep the run going: steadily rising money tracks every read
            self._pending[f] = (new, seen)
        return self.current

    def accept(self, field, value):
        """Take `value` for `field` now, without waiting for `confirm` reads: for a change the
        caller has confirmed another way (the harness saw the money drop by a purchase's price)."""
        self.value[field], self._pending[field], self.misses[field] = value, None, 0

    @property
    def current(self) -> Hud:
        return Hud(**self.value)
