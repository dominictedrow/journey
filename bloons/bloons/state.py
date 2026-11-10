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

from bloons.vision import Templates, _templates, to_work

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


@dataclass(frozen=True)
class RoundState:
    screen: str                     # Screen.kind of the latest frame
    round_active: Optional[bool]    # bloons on the track; None until first seen in game
    round_ended: bool               # True on exactly one update per finished round
    fast_forward: Optional[bool]    # None when the button isn't visible
    outcome: Optional[str]          # None / 'win' / 'loss'; sticky until reset()
    round: Optional[int] = None     # from the filtered HUD
    rounds_total: Optional[int] = None
    ended_rounds: tuple = ()            # with round_ended: every round that ended, oldest first
                                        # (several if the HUD jumped between updates); () if unknown

    @property
    def ended_round(self) -> Optional[int]:
        """With round_ended: the (last) round that ended, if known."""
        return self.ended_rounds[-1] if self.ended_rounds else None


MAX_ROUND_JUMP = 5     # HUD round rises above this between updates -> credit only the last


class RoundTracker:
    """Round state over successive frames.

    - round_active: the play button shows fast-forward (ff_off / ff_on) -> True; the
      start-round arrow (idle) -> False. Pause, popups and in-game frames without a readable
      button keep the last value; victory/defeat set it False.
    - round_ended fires on exactly one update per finished round, whatever the Auto Start
      setting: when the button goes running -> idle (Auto Start off; with it on the idle
      button shows for only ~1 s, RESEARCH 1.5), or when the HUD round number goes up while
      the button shows a running round (Auto Start on, idle missed). Each finished round
      fires once: the round number of the running stretch is remembered, so a HUD that
      catches up a frame late doesn't fire it again. A HUD jump of several rounds between
      updates (no update for a while with Auto Start on) ends each round in between, up to
      MAX_ROUND_JUMP; a bigger jump is more likely a misread, so only the last round counts.
      A running stretch whose HUD still shows a finished round (lagging) is taken as the
      next round. A HUD round below the last finished one is a lag or a misread, never a
      replay: a Defeat 'Continue' replays only the round that was lost, which never finished. Not fired for the final round: outcome='win' covers that.
    - outcome: 'win' / 'loss' once the victory / defeat screen has been seen on `confirm`
      consecutive frames. Sticky until reset() (the restart popup over defeat keeps 'loss').
    - Button changes also need `confirm` consecutive frames, like HudFilter.
    Feed it filtered HUD values (hud.HudFilter), so the round number is already debounced.
    """

    def __init__(self, confirm=2):
        self.confirm = confirm
        self.reset()

    def reset(self):
        """Call when a new game starts."""
        self.active = None
        self.outcome = None
        self.round = self.rounds_total = None
        self._button = None            # accepted button state
        self._pending = (None, 0)      # (candidate button or outcome screen, times seen)
        self._run_round = None         # HUD round of the running stretch (the one that will end)
        self._finished = None          # last round number round_ended fired for
        self._ended = []
        self._jumped = ()              # rounds the last HUD jump finished (undone if the HUD drops back)
        self._last = None

    def _finish(self, rnd):
        """round_ended for round `rnd` (None = number unknown) unless it already fired."""
        if rnd is not None and self._finished is not None and rnd <= self._finished:
            return False
        if rnd is not None:
            self._finished = rnd
            self._ended.append(rnd)
        return True

    def _running_round(self):
        """The round a stretch that just started is playing: the HUD's, unless that lags
        behind a round already finished (Auto Start on: the HUD updates after the button)."""
        if self._finished is not None and (self.round is None or self.round <= self._finished):
            return self._finished + 1
        return self.round

    @property
    def running_round(self):
        """The round the current running stretch is playing (lag-corrected), or None."""
        return self._run_round

    @property
    def button(self):
        """The accepted (confirmed) play button state, or None."""
        return self._button

    def _confirmed(self, value):
        cand, seen = self._pending
        seen = seen + 1 if cand == value else 1
        self._pending = (value, seen)
        return seen >= self.confirm

    def update(self, screen: Screen, hud=None) -> RoundState:
        ended = False
        self._ended = []
        prev_round = self.round
        if hud is not None:
            if hud.round is not None:
                self.round = hud.round
                if self._jumped and self._jumped[0] <= hud.round <= self._jumped[-1]:
                    # the HUD came back into rounds the last jump finished: the jump was a
                    # misread (5 read as 9), so those rounds haven't ended; let them end.
                    # (Below the jump's start it's a misread the other way: ignored.)
                    self._finished = hud.round - 1
                    if self.active:
                        self._run_round = hud.round
                    self._jumped = ()
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
                if self.active and not was_active:
                    self._run_round = self._running_round()
                elif was_active and not self.active:
                    ended = self._finish(self._run_round)
                    self._jumped = ()
            if (self.active and screen.button != "idle" and prev_round is not None
                    and self.round is not None and self.round > prev_round):
                first = prev_round if self.round - prev_round <= MAX_ROUND_JUMP else self.round - 1
                done = tuple(r for r in range(first, self.round) if self._finish(r))
                ended = bool(done) or ended
                self._jumped = done or self._jumped
                self._run_round = self.round
            elif self.active and self._run_round is None:
                self._run_round = self._running_round()
        else:
            self._pending = (None, 0)

        ff = None if self._button is None else self._button == "ff_on"
        got = tuple(self._ended) if ended else ()
        self._last = RoundState(screen.kind, self.active, ended, ff, self.outcome,
                                self.round, self.rounds_total, got)
        return self._last

    @property
    def current(self) -> Optional[RoundState]:
        return self._last
