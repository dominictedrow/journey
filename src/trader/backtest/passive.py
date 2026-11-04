"""EXEC01 passive-execution simulator (research/EXEC01_DESIGN.md, pre-registered in b1dcdfe).

Runs an unchanged signal mode (REV01) but enters with a resting limit order instead of a market
order, and optionally takes profit with a resting limit (variant E2). Backtest-only.

Broker quote model: per 1-minute bar, ``bid_x = mid_x - H`` and ``ask_x = mid_x + H`` where ``H``
is the scenario's half-spread for that minute (observed, or floored at a retail minimum).

Timing (no look-ahead):
- A decision is made at the end of a mode bar. The order is placed at the OPEN of the first
  1-minute bar at/after that time (minute ``f``), priced off that minute's opening quote: a buy
  limit at the bid, a sell limit at the ask. It rests from that open, so minute ``f`` itself
  can fill it, and it is cancelled after ``ttl`` (unfilled signals are logged).
- Fill rules: ``cross`` (primary): buy limit L fills when ask_low <= L, sell when bid_high >= L.
  ``through`` (sensitivity only): buy fills when bid_low <= L - through, sell when
  ask_high >= L + through. Limit fills are at L with no slippage.
- Stop = fill price -/+ the mode's stop distance. Stop-market: long triggers on bid_low <= stop,
  short on ask_high >= stop, fill at the bid/ask minus/plus slippage (gap: at the open). It is
  checked from the entry minute itself (worst case). The take-profit limit (E2) is checked
  only from the minute after entry, and a stop wins if both hit in one minute.
- Mode exits (time / flat / reverted) are market orders at the next minute's open quote. In E2
  the "reverted" exit is replaced by the take-profit limit; time and flat exits still apply.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trader.backtest.engine import TRADE_COLUMNS, daily_equity, trading_day_ids
from trader.data.resample import resample_bars
from trader.instruments import Instrument
from trader.modes.base import BarContext, Enter, Exit, Mode, PositionView, Side
from trader.risk import RiskConfig, RiskLayer

SIGNAL_COLUMNS = ["signal_time", "side", "limit", "filled", "fwd_r"]


@dataclass
class PassiveResult:
    trades: pd.DataFrame
    signals: pd.DataFrame
    daily_equity: pd.Series
    initial_equity: float
    rejections: dict[str, int] = field(default_factory=dict)


@dataclass
class _Pending:
    side: Side
    limit: float
    dist: float
    target: float | None
    expire: int  # first minute index at which the order is no longer live
    k: int
    units: float
    signal_row: int


@dataclass
class _Pos:
    side: Side
    units: float
    entry: float
    stop: float
    target: float | None
    entry_idx: int
    entry_k: int
    init_stop: float = float("nan")


class PassiveBacktester:
    def __init__(
        self,
        instrument: Instrument,
        risk: RiskConfig | None = None,
        initial_equity: float = 25_000.0,
        *,
        take_profit: bool = False,
        fill_rule: str = "cross",
        through: float = 0.00001,
        ttl: str = "5min",
        fwd_minutes: int = 60,
    ) -> None:
        if fill_rule not in ("cross", "through"):
            raise ValueError("fill_rule must be 'cross' or 'through'")
        self.inst = instrument
        self.risk = RiskLayer(risk or RiskConfig(), instrument)
        self.initial_equity = initial_equity
        self.take_profit = take_profit
        self.fill_rule = fill_rule
        self.through = through
        self.ttl = pd.Timedelta(ttl)
        self.fwd_minutes = fwd_minutes

    def run(self, mode: Mode, mid: pd.DataFrame, half_spread: np.ndarray) -> PassiveResult:
        inst, slip = self.inst, self.inst.slippage
        n = len(mid)
        H = np.asarray(half_spread, dtype=float)
        if H.shape != (n,) or not np.isfinite(H).all() or (H < 0).any():
            raise ValueError("half_spread must be a finite, non-negative array, one per bar")
        m_ns = mid.index.asi8
        mo, mh, ml, mc = (mid[c].to_numpy() for c in ("open", "high", "low", "close"))
        bid_lo, bid_hi, ask_lo, ask_hi = ml - H, mh - H, ml + H, mh + H
        bid_op, ask_op = mo - H, mo + H
        eps = inst.tick_size * 1e-3  # float tolerance: touching a price counts as reaching it

        htf = resample_bars(mid, mode.timeframe)
        htf_bars = htf.drop(columns="end")
        mode.prepare(htf_bars)
        h_arrays = {c: htf_bars[c].to_numpy() for c in htf_bars.columns}
        h_end = pd.DatetimeIndex(htf["end"]).asi8
        fill_idx = np.searchsorted(m_ns, h_end, side="left")
        z, sigma = mode.feat.get("z"), mode.feat.get("sigma")
        day_id = trading_day_ids(mid.index)

        cash = self.initial_equity
        pend: _Pending | None = None
        pos: _Pos | None = None
        trades: list[dict] = []
        signals: list[dict] = []
        rej: Counter[str] = Counter()
        cur = 0
        current_day, day_start_eq = None, cash

        def close(p: _Pos, idx: int, price: float, reason: str) -> None:
            nonlocal cash
            pnl = (price - p.entry) * p.side.value * p.units * inst.point_value
            risk_amt = abs(p.entry - p.init_stop) * p.units * inst.point_value
            cash += pnl
            raw_mid = mo[idx] if reason not in ("stop", "target") else price
            trades.append(
                {
                    "entry_time": mid.index[p.entry_idx], "exit_time": mid.index[idx],
                    "side": p.side.value, "units": p.units, "entry_price": p.entry,
                    "exit_price": price, "initial_stop": p.init_stop,
                    "pnl": pnl, "cost": abs(price - raw_mid) * p.units * inst.point_value,
                    "r_multiple": pnl / risk_amt if risk_amt > 0 else np.nan,
                    "entry_reason": "limit", "exit_reason": reason,
                    "entry_idx": p.entry_idx, "exit_idx": idx,
                }
            )  # fmt: skip

        def scan_position(lo: int, hi: int) -> bool:
            """Check stop (and TP) on minutes [lo, hi). Returns True if the position closed."""
            nonlocal pos
            p = pos
            if p is None or hi <= lo:
                return False
            if p.side is Side.LONG:
                stop_hit = bid_lo[lo:hi] <= p.stop + eps
                tp_hit = bid_hi[lo:hi] >= p.target - eps if p.target is not None else None
            else:
                stop_hit = ask_hi[lo:hi] >= p.stop - eps
                tp_hit = ask_lo[lo:hi] <= p.target + eps if p.target is not None else None
            if tp_hit is not None:
                tp_hit = tp_hit & (np.arange(lo, hi) > p.entry_idx)
                hit = stop_hit | tp_hit
            else:
                hit = stop_hit
            hits = np.flatnonzero(hit)
            if not hits.size:
                return False
            j = int(hits[0])
            idx = lo + j
            if stop_hit[j]:
                if p.side is Side.LONG:
                    px = min(bid_op[idx], p.stop) if idx > p.entry_idx else p.stop
                    px -= slip
                else:
                    px = max(ask_op[idx], p.stop) if idx > p.entry_idx else p.stop
                    px += slip
                close(p, idx, px, "stop")
            else:
                close(p, idx, float(p.target), "target")  # type: ignore[arg-type]
            pos = None
            return True

        def advance(to: int) -> None:
            nonlocal pend, pos, cur
            while cur < to:
                if pend is not None:
                    end = min(to, pend.expire)
                    sl = slice(cur, end)
                    if pend.side is Side.LONG:
                        ok = (
                            ask_lo[sl] <= pend.limit + eps
                            if self.fill_rule == "cross"
                            else bid_lo[sl] <= pend.limit - self.through + eps
                        )
                    else:
                        ok = (
                            bid_hi[sl] >= pend.limit - eps
                            if self.fill_rule == "cross"
                            else ask_hi[sl] >= pend.limit + self.through - eps
                        )
                    hits = np.flatnonzero(ok)
                    if hits.size:
                        j = cur + int(hits[0])
                        stop = pend.limit - pend.side.value * pend.dist
                        pos = _Pos(
                            pend.side, pend.units, pend.limit, stop, pend.target, j, pend.k, stop
                        )
                        signals[pend.signal_row]["filled"] = True
                        pend = None
                        scan_position(j, to)  # stop may hit in the fill minute itself
                        cur = to
                        continue
                    if end >= pend.expire:
                        pend = None
                    cur = end
                elif pos is not None:
                    scan_position(cur, to)
                    cur = to
                else:
                    cur = to

        def equity_at(idx: int) -> float:
            if pos is None or idx < 0:
                return cash
            return cash + (mc[idx] - pos.entry) * pos.side.value * pos.units * inst.point_value

        mask = mode.decision_mask()
        ks = np.flatnonzero(mask) if mask is not None else range(len(h_end))
        for k in ks:
            k = int(k)
            f = int(fill_idx[k])
            if f >= n:
                break
            advance(f)
            eq = equity_at(f - 1)
            if day_id[f] != current_day:
                current_day, day_start_eq = day_id[f], eq
            if pos is None:
                view = PositionView()
            else:
                view = PositionView(
                    side=pos.side, units=pos.units, entry_price=pos.entry, stop=pos.stop,
                    target=pos.target, entry_time=mid.index[pos.entry_idx],
                    bars_held=k - pos.entry_k,
                )  # fmt: skip
            ctx = BarContext(
                i=k, time=pd.Timestamp(h_end[k], tz="UTC"), index=htf_bars.index,
                position=view, equity=eq, **h_arrays,
            )  # fmt: skip
            for intent in mode.on_bar(ctx) or []:
                if isinstance(intent, Exit):
                    if pos is None or (self.take_profit and intent.reason == "reverted"):
                        continue
                    px = bid_op[f] - slip if pos.side is Side.LONG else ask_op[f] + slip
                    close(pos, f, px, intent.reason or "exit")
                    pos = None
                elif isinstance(intent, Enter):
                    if pos is not None or pend is not None:
                        rej["busy"] += 1
                        continue
                    if (m_ns[f] - h_end[k]) > pd.Timedelta("5min").value:
                        rej["stale_data"] += 1
                        continue
                    side = intent.side
                    limit = bid_op[f] if side is Side.LONG else ask_op[f]
                    dist = abs(h_arrays["close"][k] - float(intent.stop))  # type: ignore[arg-type]
                    target = None
                    if self.take_profit and z is not None and sigma is not None:
                        anchor = h_arrays["close"][k] - z[k] * sigma[k]
                        if np.isfinite(anchor) and (anchor - limit) * side.value > 0:
                            target = float(anchor)
                    entry = Enter(side, stop=limit - side.value * dist, target=target)
                    day_loss = max(0.0, (day_start_eq - eq) / day_start_eq)
                    d = self.risk.approve_entry(entry, limit, eq, day_loss)
                    if not d.approved:
                        rej[d.reason] += 1
                        continue
                    g = min(f + self.fwd_minutes, n - 1)
                    fwd = (mc[g] - mo[f]) * side.value / dist if dist > 0 else np.nan
                    signals.append(
                        {"signal_time": mid.index[f], "side": side.value, "limit": limit,
                         "filled": False, "fwd_r": fwd}
                    )  # fmt: skip
                    expire = int(np.searchsorted(m_ns, m_ns[f] + self.ttl.value, side="left"))
                    pend = _Pending(side, limit, dist, target, expire, k, d.units, len(signals) - 1)

        advance(n)
        if pos is not None:
            px = bid_op[n - 1] if pos.side is Side.LONG else ask_op[n - 1]
            close(pos, n - 1, px, "end_of_data")
        trade_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        return PassiveResult(
            trades=trade_df,
            signals=pd.DataFrame(signals, columns=SIGNAL_COLUMNS),
            daily_equity=daily_equity(trade_df, mid, day_id, inst, self.initial_equity),
            initial_equity=self.initial_equity,
            rejections=dict(rej),
        )
