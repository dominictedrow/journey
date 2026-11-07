"""Screen classification, round state and game outcome. Portable: built on vision.py
(templates) and hud.py (numbers); no OS calls.

- `read_screen(img)`: what one frame shows -> `Screen(kind, button, panel)`.
- `RoundTracker`: feeds successive `Screen`s + filtered `Hud`s and keeps the state the
  harness's observe() needs: `round_active`, `round_ended` (edge), `fast_forward`, `outcome`.

Screen kinds are checked in priority order, overlays first. Template matching ignores
brightness, so the play button still "matches" under the dimmed victory/pause overlays
(RESEARCH.md, step 1.4), and the defeat screen still shows behind the restart popup.
"""
from dataclasses import dataclass, field
from typing import Optional

from bloons.vision import Templates, to_work

# (kind, templates that must all match), highest priority first.
SCREENS = (
    ("restart_confirm", ("restart_confirm",)),
    ("overwrite_save", ("overwrite_save",)),
    ("victory", ("victory_banner",)),
    ("defeat", ("defeat_title",)),
    ("collection_event", ("collection_event",)),
    ("pause", ("pause_title",)),
    ("map_select", ("menu_map_select",)),
    ("mode_select", ("menu_mode_standard",)),
    ("difficulty_select", ("menu_difficulty",)),
    ("home", ("menu_home",)),
)
OUTCOMES = {"victory": "win", "defeat": "loss"}


@dataclass(frozen=True)
class Screen:
    kind: str                       # one of SCREENS' kinds, "game" or "unknown"
    button: Optional[str] = None    # in game: 'idle' / 'ff_off' / 'ff_on', None if unreadable
    panel: Optional[str] = None     # in game: upgrade panel side 'left' / 'right', or None
    scores: dict = field(default_factory=dict, compare=False)   # template -> score, for logs


def read_screen(img, templates: Optional[Templates] = None) -> Screen:
    """Classify one frame (content-area BGR, any size)."""
    t = templates or _templates()
    img = to_work(img)
    m = t.match_all(img)
    scores = {k: round(v.score, 3) for k, v in m.items()}
    for kind, needed in SCREENS:
        if all(m[n].found for n in needed):
            return Screen(kind, scores=scores)
    button = t.button_state(img)
    if button is None and not (m["btn_play"].found or m["btn_ff"].found):
        return Screen("unknown", scores=scores)
    # A button whose shape matches but whose colour fits no class (seen live: a brighter
    # glow, probably mouse hover) is still a game frame, just with an unknown button state.
    sell = m["sell_button"]
    panel = ("left" if sell.center[0] < 0.5 else "right") if sell.found else None
    return Screen("game", button, panel, scores)


_default = None


def _templates():
    global _default
    if _default is None:
        _default = Templates()
    return _default


@dataclass(frozen=True)
class RoundState:
    screen: str                     # Screen.kind of the latest frame
    round_active: Optional[bool]    # bloons on the track; None until first seen in game
    round_ended: bool               # True on exactly one update per finished round
    fast_forward: Optional[bool]    # None when the button isn't visible
    outcome: Optional[str]          # None / 'win' / 'loss'; sticky until reset()
    round: Optional[int] = None     # from the filtered HUD
    rounds_total: Optional[int] = None


class RoundTracker:
    """Round state over successive frames.

    - round_active: the play button shows fast-forward (ff_off / ff_on) -> True; the
      start-round arrow (idle) -> False. Pause, popups and in-game frames without a readable
      button keep the last value; victory/defeat set it False.
    - round_ended fires on exactly one update per finished round:
        auto_start=False: when the button goes running -> idle.
        auto_start=True: the button never goes idle, so when the HUD round number goes up
        while a round is running. Tell the tracker if the harness changes the setting.
      Not fired for the final round: outcome='win' covers that.
    - outcome: 'win' / 'loss' once the victory / defeat screen has been seen on `confirm`
      consecutive frames. Sticky until reset() (the restart popup over defeat keeps 'loss').
    - Button changes also need `confirm` consecutive frames, like HudFilter.
    Feed it filtered HUD values (hud.HudFilter), so the round number is already debounced.
    """

    def __init__(self, auto_start=False, confirm=2):
        self.auto_start = auto_start
        self.confirm = confirm
        self.reset()

    def reset(self):
        """Call when a new game starts."""
        self.active = None
        self.outcome = None
        self.round = self.rounds_total = None
        self._button = None            # accepted button state
        self._pending = (None, 0)      # (candidate button or outcome screen, times seen)
        self._last = None

    def _confirmed(self, value):
        cand, seen = self._pending
        seen = seen + 1 if cand == value else 1
        self._pending = (value, seen)
        return seen >= self.confirm

    def update(self, screen: Screen, hud=None) -> RoundState:
        ended = False
        prev_round = self.round
        if hud is not None:
            if hud.round is not None:
                self.round = hud.round
            if hud.rounds_total is not None:
                self.rounds_total = hud.rounds_total

        if screen.kind in OUTCOMES:
            if self.outcome is None and self._confirmed(screen.kind):
                self.outcome = OUTCOMES[screen.kind]
            if self.outcome is not None:
                self.active, self._button = False, None
        elif screen.kind == "game" and screen.button is not None:
            if screen.button == self._button:
                self._pending = (None, 0)
            elif self._button is None or self._confirmed(screen.button):
                was_active = self.active
                self._button = screen.button
                self.active = screen.button != "idle"
                ended = bool(was_active and not self.active and not self.auto_start)
            if (self.auto_start and self.active and prev_round is not None
                    and self.round is not None and self.round > prev_round):
                ended = True
        else:
            self._pending = (None, 0)

        ff = None if self._button is None else self._button == "ff_on"
        self._last = RoundState(screen.kind, self.active, ended, ff, self.outcome,
                                self.round, self.rounds_total)
        return self._last

    @property
    def current(self) -> Optional[RoundState]:
        return self._last
