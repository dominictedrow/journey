"""Joovier creator modes J-T01..J-T03 (research/sources/JOOVIER_TRANSCRIPT_FINDINGS.md).

Provenance INTERPRETATION (D-018). The source is a partial third-party transcript of one course,
with no chart playback. Every threshold here is ours, fixed before the first real-data run, and
documented in research/experiments/JOOVIER_MODE_RULES.md. These modes must never be called
creator-faithful.
The creator's volume cues are dropped: the CFD/FX files carry tick volume only (D-020).

Shared conventions:
- A "swing" is ``swing_pivots(k)``: known only k bars after the pivot bar.
- Higher-timeframe (HTF) levels are built in ``prepare`` by resampling this mode's own bars, and a
  value is used only once its HTF bar has closed (``htf_position``).
- Session: entries between ``entry_start`` and ``entry_end`` ET, flat at ``flat_time`` ET,
  weekdays only, as in REV01. Times apply to the decision time (bar end).
- Fills are always on real bars. Heikin-Ashi values are signals only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trader.data.resample import ET, resample_bars
from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, Provenance, Side
from trader.modes.indicators import (
    anchored_vwap,
    atr,
    heikin_ashi,
    htf_position,
    last_known,
    recent_pivots,
    swing_pivots,
    take,
)

SESSION = {"entry_start": "09:45", "entry_end": "15:30", "flat_time": "15:55"}  # S&P RTH


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class _SessionMode(Mode):
    """Session window, decision mask and day-flat exit shared by the J-T modes."""

    provenance = Provenance.INTERPRETATION
    instruments = ("USA500IDXUSD", "EURUSD")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.timeframe = self.params["timeframe"]  # per-variant timeframe

    def _prepare_session(self, bars: pd.DataFrame) -> None:
        p = self.params
        end_et = (bars.index + pd.Timedelta(self.timeframe)).tz_convert(ET)
        mins = np.asarray(end_et.hour * 60 + end_et.minute)
        weekday = np.asarray(end_et.dayofweek < 5)
        start, stop, flat = (_minutes(p[k]) for k in ("entry_start", "entry_end", "flat_time"))
        self.feat["entry_ok"] = weekday & (mins >= start) & (mins <= stop)
        self.feat["flat"] = weekday & (mins >= flat)
        self.feat["active"] = weekday & (mins >= start) & (mins <= flat)

    def decision_mask(self) -> np.ndarray:
        # Timestamp-only, as REV01: the entry window through the flat bar, plus the first bar
        # after flat_time as a safety net when bars around it are missing.
        active = self.feat["active"].copy()
        after_flat = self.feat["flat"] & ~active
        first_after = after_flat & ~np.r_[False, after_flat[:-1]]
        return active | first_after

    def _day_flat(self, i: int) -> bool:
        return bool(self.feat["flat"][i] or not self.feat["active"][i])

    def _sides(self) -> tuple[Side, ...]:
        return (Side.LONG, Side.SHORT) if self.params["allow_short"] else (Side.LONG,)


_SFX = {Side.LONG: "L", Side.SHORT: "S"}


class SupportResistanceRetest(_SessionMode):
    """J-T01: HTF support/resistance retest, then an LTF swing break.

    - Zones: the last ``n_zones`` confirmed ``zone_tf`` swing lows (support) and highs
      (resistance), each +/- ``zone_atr`` x HTF ATR at confirmation.
    - Touch: an LTF bar trades into a zone without closing through it. That arms the side for
      ``arm_bars`` bars; a later close through the far edge of the zone disarms it.
    - Entry: while armed, a close beyond the last confirmed LTF swing high (long) / low (short).
    - Stop at the extreme since the touch (the retest swing). Target at the nearest opposite zone
      level; skip unless reward >= ``min_rr`` x risk.

    Feature arrays for the short side are stored sign-flipped (x -> -x) so one code path serves
    both sides: "above" always means "in the trade's favour".
    """

    name = "jt01"
    version = "0.1.0"
    timeframe = "1min"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "timeframe": "1min",
            "zone_tf": "1h",
            "zone_k": 3,
            "n_zones": 3,
            "zone_atr": 0.25,
            "atr_period": 14,
            "swing_k": 3,
            "arm_bars": 60,
            "min_rr": 1.0,
            **SESSION,
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        self._prepare_session(bars)
        self._arm: dict[Side, tuple[int, float, float] | None] = {Side.LONG: None, Side.SHORT: None}

        htf = resample_bars(bars, p["zone_tf"])
        width = p["zone_atr"] * atr(htf, p["atr_period"])
        sh, sl = swing_pivots(htf, p["zone_k"])
        pos = htf_position(bars.index, pd.Timedelta(self.timeframe), htf["end"])

        def zones(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ok = np.isfinite(levels) & np.isfinite(width)
            lv, _ = recent_pivots(np.where(ok, levels, np.nan), p["n_zones"])
            wd, _ = recent_pivots(np.where(ok, width, np.nan), p["n_zones"])
            return take(lv, pos), take(wd, pos)

        sup, sup_w = zones(sl)
        res, res_w = zones(sh)
        lo, hi, c = (bars[k].to_numpy()[:, None] for k in ("low", "high", "close"))
        ltf_sh, ltf_sl = swing_pivots(bars, p["swing_k"])

        with np.errstate(invalid="ignore"):
            touch_l = (lo <= sup + sup_w) & (c >= sup - sup_w)
            touch_s = (hi >= res - res_w) & (c <= res + res_w)
            floor_l = np.where(touch_l, sup - sup_w, -np.inf).max(axis=1)
            ceil_s = np.where(touch_s, res + res_w, np.inf).min(axis=1)
            tgt_l = np.where(res > c, res, np.inf).min(axis=1)
            tgt_s = np.where(sup < c, sup, -np.inf).max(axis=1)

        def finite(x: np.ndarray) -> np.ndarray:
            return np.where(np.isfinite(x), x, np.nan)

        f = self.feat
        f["touchL"], f["touchS"] = touch_l.any(axis=1), touch_s.any(axis=1)
        f["floorL"], f["floorS"] = finite(floor_l), -finite(ceil_s)
        f["tgtL"], f["tgtS"] = finite(tgt_l), -finite(tgt_s)
        f["brkL"], f["brkS"] = last_known(ltf_sh), -last_known(ltf_sl)
        f["advL"], f["advS"] = lo[:, 0], -hi[:, 0]
        f["closeL"], f["closeS"] = c[:, 0], -c[:, 0]

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p, f, i = self.params, self.feat, ctx.i
        if ctx.position.side is not Side.FLAT:
            return [Exit("flat_time")] if self._day_flat(i) else []
        if self._day_flat(i):
            self._arm = {Side.LONG: None, Side.SHORT: None}
            return []
        enter: Enter | None = None
        for side in self._sides():
            sfx, s = _SFX[side], side.value
            sc, adv = f["close" + sfx][i], f["adv" + sfx][i]
            arm = self._arm[side]  # (expiry bar, signed zone floor, signed extreme since touch)
            if arm is not None and (i > arm[0] or sc < arm[1]):
                arm = None
            if arm is not None:
                arm = (arm[0], arm[1], min(arm[2], adv))
                brk, tgt = f["brk" + sfx][i], f["tgt" + sfx][i]
                if enter is None and f["entry_ok"][i] and math.isfinite(brk) and sc > brk:
                    risk = sc - arm[2]
                    if risk > 0 and math.isfinite(tgt) and tgt - sc >= p["min_rr"] * risk:
                        enter = Enter(side, stop=s * arm[2], target=s * tgt, reason="retest_break")
                        arm = None
            if f["touch" + sfx][i]:
                extreme = adv if arm is None else min(arm[2], adv)
                arm = (i + p["arm_bars"], f["floor" + sfx][i], extreme)
            self._arm[side] = arm
        if enter is not None:
            self._arm = {Side.LONG: None, Side.SHORT: None}
            return [enter]
        return []


class SqueezeBreakout(_SessionMode):
    """J-T02: converging HTF trendlines, broken with an LTF swing-break confirmation.

    - Pattern on ``pattern_tf``: the last two confirmed swing highs are falling and the last two
      swing lows rising (>= 2 rejections per line). Lines run through each pair in HTF bar
      units, are extrapolated to the forming HTF bar, must not have crossed, and the oldest
      pivot must be within ``max_pattern_bars`` HTF bars.
    - Long entry: LTF close above the upper line AND above the last confirmed LTF swing high (a
      line break alone is not enough). Shorts mirror. One trade per pattern.
    - Stop at the last confirmed LTF opposite swing. Target at the older (outer) pattern pivot,
      the prior swing; skip unless reward >= ``min_rr`` x risk.
    """

    name = "jt02"
    version = "0.1.0"
    timeframe = "5min"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "timeframe": "5min",
            "pattern_tf": "1h",
            "pivot_k": 3,
            "max_pattern_bars": 72,
            "swing_k": 3,
            "min_rr": 1.0,
            **SESSION,
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        self._prepare_session(bars)
        self._used = -1.0  # id of the last pattern traded

        k = p["pivot_k"]
        htf = resample_bars(bars, p["pattern_tf"])
        sh, sl = swing_pivots(htf, k)
        hv, hr = recent_pivots(sh, 2)
        lv, lr = recent_pivots(sl, 2)
        pos = htf_position(bars.index, pd.Timedelta(self.timeframe), htf["end"])
        h1, h2, l1, l2 = (take(a, pos) for a in (hv[:, 0], hv[:, 1], lv[:, 0], lv[:, 1]))
        # Pivot bars (confirmed k bars later), in HTF row units; -1 rows become NaN.
        rows = (hr[:, 0], hr[:, 1], lr[:, 0], lr[:, 1])
        hx1, hx2, lx1, lx2 = (take(np.where(r >= 0, r - k, np.nan), pos) for r in rows)
        x = (pos + 1).astype(float)  # the HTF bar still forming at this LTF bar
        c = bars["close"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            upper = h2 + (h2 - h1) / (hx2 - hx1) * (x - hx2)
            lower = l2 + (l2 - l1) / (lx2 - lx1) * (x - lx2)
            valid = (
                (h2 < h1) & (l2 > l1) & (upper > lower)
                & (x - np.fmin(hx1, lx1) <= p["max_pattern_bars"])
            )  # fmt: skip
        ltf_sh, ltf_sl = (last_known(a) for a in swing_pivots(bars, p["swing_k"]))

        f = self.feat
        f["pid"] = take(np.maximum(hr[:, 1], lr[:, 1]).astype(float), pos)
        with np.errstate(invalid="ignore"):
            f["sigL"] = valid & (c > upper) & (c > ltf_sh)
            f["sigS"] = valid & (c < lower) & (c < ltf_sl)
        f["stopL"], f["stopS"] = ltf_sl, ltf_sh
        f["tgtL"], f["tgtS"] = h1, l1

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p, f, i = self.params, self.feat, ctx.i
        if ctx.position.side is not Side.FLAT:
            return [Exit("flat_time")] if self._day_flat(i) else []
        if not f["entry_ok"][i] or f["pid"][i] == self._used:
            return []
        c = ctx.close[i]
        for side in self._sides():
            sfx, s = _SFX[side], side.value
            if not f["sig" + sfx][i]:
                continue
            stop, tgt = f["stop" + sfx][i], f["tgt" + sfx][i]
            risk, reward = s * (c - stop), s * (tgt - c)
            if risk > 0 and reward >= p["min_rr"] * risk:
                self._used = f["pid"][i]
                return [Enter(side, stop=stop, target=tgt, reason="squeeze_break")]
        return []


class VwapHeikinAshi(_SessionMode):
    """J-T03: VWAP-side Heikin-Ashi pullback continuation scalp.

    - Bias: long only while closes have stayed above the session VWAP (anchored daily at
      ``vwap_anchor`` ET) for >= ``min_side_bars`` bars, which skips the first pullback after a
      cross, and the last two confirmed swing highs and lows are both rising (structure).
    - Pullback: >= ``min_clean`` clean bearish HA bars (no upper wick) among the previous
      ``pullback_bars`` bars; they need not be consecutive.
    - Entry: the first bullish HA bar after a bearish one, clean (no lower wick).
    - Stop at the lowest real low of the pullback window. No target: exit on a clean opposite
      HA bar, a close back through VWAP, after ``max_bars``, or at the day flat. Shorts mirror.
    """

    name = "jt03"
    version = "0.1.0"
    timeframe = "1min"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "timeframe": "1min",
            "vwap_anchor": "09:30",
            "min_side_bars": 15,
            "swing_k": 3,
            "pullback_bars": 10,
            "min_clean": 2,
            "max_bars": 30,
            **SESSION,
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        self._prepare_session(bars)
        h, lo, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
        vwap = anchored_vwap(bars, p["vwap_anchor"])
        ha_open, ha_close = heikin_ashi(bars)
        bull, bear = ha_close > ha_open, ha_close < ha_open
        clean_bull = bull & (lo >= ha_open)  # bullish HA has no lower wick
        clean_bear = bear & (h <= ha_open)  # bearish HA has no upper wick

        with np.errstate(invalid="ignore"):
            vside = pd.Series(np.sign(c - vwap)).fillna(0.0)
        run = vside.groupby((vside != vside.shift()).cumsum()).cumcount().to_numpy() + 1
        sh, sl = swing_pivots(bars, p["swing_k"])
        hv, _ = recent_pivots(sh, 2)
        lv, _ = recent_pivots(sl, 2)
        up = (hv[:, 1] > hv[:, 0]) & (lv[:, 1] > lv[:, 0])
        down = (hv[:, 1] < hv[:, 0]) & (lv[:, 1] < lv[:, 0])

        nb = p["pullback_bars"]
        bears_before = pd.Series(clean_bear, dtype=float).shift(1).rolling(nb).sum().to_numpy()
        bulls_before = pd.Series(clean_bull, dtype=float).shift(1).rolling(nb).sum().to_numpy()
        prev_bear, prev_bull = np.r_[False, bear[:-1]], np.r_[False, bull[:-1]]
        vs = vside.to_numpy()
        bias_l = (vs > 0) & (run >= p["min_side_bars"]) & up
        bias_s = (vs < 0) & (run >= p["min_side_bars"]) & down

        f = self.feat
        with np.errstate(invalid="ignore"):
            f["sigL"] = bias_l & (bears_before >= p["min_clean"]) & clean_bull & prev_bear
            f["sigS"] = bias_s & (bulls_before >= p["min_clean"]) & clean_bear & prev_bull
            f["vwap_exitL"], f["vwap_exitS"] = c < vwap, c > vwap
        f["stopL"] = pd.Series(lo).rolling(nb + 1).min().to_numpy()
        f["stopS"] = pd.Series(h).rolling(nb + 1).max().to_numpy()
        f["ha_exitL"], f["ha_exitS"] = clean_bear, clean_bull

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p, f, i = self.params, self.feat, ctx.i
        side = ctx.position.side
        if side is not Side.FLAT:
            sfx = _SFX[side]
            if self._day_flat(i):
                return [Exit("flat_time")]
            if ctx.position.bars_held >= p["max_bars"]:
                return [Exit("time")]
            if f["ha_exit" + sfx][i]:
                return [Exit("ha_flip")]
            if f["vwap_exit" + sfx][i]:
                return [Exit("vwap")]
            return []
        if not f["entry_ok"][i]:
            return []
        c = ctx.close[i]
        for side in self._sides():
            sfx = _SFX[side]
            stop = f["stop" + sfx][i]
            if f["sig" + sfx][i] and math.isfinite(stop) and side.value * (c - stop) > 0:
                return [Enter(side, stop=stop, reason="ha_pullback")]
        return []
