"""Event-driven backtester.

Timing rules (no look-ahead):
- A mode's bar k is complete at its bucket end. The mode decides then, and orders fill at the OPEN
  of the first 1-minute bar starting at or after that time (next-bar fills).
- Stops and targets are checked on 1-minute bars even for higher-timeframe modes. A gap through
  the stop fills at the open. If one minute touches both stop and target, the stop wins.
- Entries are dropped (NO TRADE) when the next available bar is more than the mode's
  ``max_entry_delay`` (else the engine default)
  after the decision time, e.g. across weekends and data holes. Exits always execute.
- Costs: market/stop fills pay half-spread + slippage, limit (target) fills pay half-spread,
  plus commission per unit per side. Data prices are treated as mid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trader.data.resample import ET, resample_bars
from trader.instruments import Instrument
from trader.modes.base import BarContext, Enter, Exit, Mode, MoveStop, PositionView, Side
from trader.risk import RiskConfig, RiskLayer


@dataclass
class _Position:
    side: Side
    units: float
    entry_price: float
    stop: float
    initial_stop: float
    target: float | None
    entry_idx: int
    entry_k: int
    entry_commission: float
    entry_cost: float
    reason: str


@dataclass
class BacktestResult:
    mode: dict
    symbol: str
    cost_multiplier: float
    initial_equity: float
    trades: pd.DataFrame
    daily_equity: pd.Series
    rejections: dict[str, int] = field(default_factory=dict)
    decisions: int = 0


def trading_day_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer trading-day id per bar using the 17:00 ET roll (matches daily resampling)."""
    wall = index.tz_convert(ET).tz_localize(None)
    days = (wall + pd.Timedelta(hours=7)).normalize()
    return days.to_numpy(dtype="datetime64[D]").astype(np.int64)


DEFAULT_ENTRY_DELAY = pd.Timedelta("5min")

TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "units", "entry_price", "exit_price", "initial_stop",
    "pnl", "cost", "r_multiple", "entry_reason", "exit_reason", "entry_idx", "exit_idx",
]  # fmt: skip


class Backtester:
    def __init__(
        self,
        instrument: Instrument,
        risk: RiskConfig | None = None,
        initial_equity: float = 25_000.0,
        max_entry_delay: pd.Timedelta | None = None,
        cost_multiplier: float = 1.0,
    ) -> None:
        self.inst = instrument.scaled_costs(cost_multiplier)
        self.cost_multiplier = cost_multiplier
        self.risk = RiskLayer(risk or RiskConfig(), self.inst)
        self.initial_equity = initial_equity
        self.max_entry_delay = max_entry_delay

    # ---- fill helpers -------------------------------------------------------------------
    def _market_fill(self, price: float, buy: bool) -> float:
        adj = self.inst.half_spread + self.inst.slippage
        return price + adj if buy else price - adj

    def _limit_fill(self, price: float, buy: bool) -> float:
        return price + self.inst.half_spread if buy else price - self.inst.half_spread

    # ---- main loop ----------------------------------------------------------------------
    def run(self, mode: Mode, bars: pd.DataFrame) -> BacktestResult:
        inst = self.inst
        n = len(bars)
        m_ns = bars.index.asi8
        mo, mh, ml, mc = (bars[c].to_numpy() for c in ("open", "high", "low", "close"))

        htf = resample_bars(bars, mode.timeframe)
        htf_bars = htf.drop(columns="end")
        mode.prepare(htf_bars)
        h_arrays = {c: htf_bars[c].to_numpy() for c in htf_bars.columns}
        h_end = pd.DatetimeIndex(htf["end"]).asi8
        fill_idx = np.searchsorted(m_ns, h_end, side="left")
        day_id = trading_day_ids(bars.index)
        delay = self.max_entry_delay or mode.max_entry_delay or DEFAULT_ENTRY_DELAY
        delay_ns = pd.Timedelta(delay).value

        cash = self.initial_equity
        pos: _Position | None = None
        trades: list[dict] = []
        rejections: Counter[str] = Counter()
        cur = 0
        day_start_equity = cash
        current_day = None

        def close(p: _Position, idx: int, raw: float, price: float, reason: str) -> None:
            nonlocal cash
            commission = p.units * inst.commission_per_unit_side
            gross = (price - p.entry_price) * p.side.value * p.units * inst.point_value
            pnl = gross - p.entry_commission - commission
            cost = p.entry_cost + abs(price - raw) * p.units * inst.point_value + commission
            risk_amt = abs(p.entry_price - p.initial_stop) * p.units * inst.point_value
            cash += pnl
            trades.append(
                {
                    "entry_time": bars.index[p.entry_idx], "exit_time": bars.index[idx],
                    "side": p.side.value, "units": p.units, "entry_price": p.entry_price,
                    "exit_price": price, "initial_stop": p.initial_stop, "pnl": pnl, "cost": cost,
                    "r_multiple": pnl / risk_amt if risk_amt > 0 else np.nan,
                    "entry_reason": p.reason, "exit_reason": reason,
                    "entry_idx": p.entry_idx, "exit_idx": idx,
                }
            )  # fmt: skip

        def scan(to: int) -> None:
            """Check the open position's stop/target on minutes [cur, to)."""
            nonlocal pos, cur
            if pos is not None and to > cur:
                lo, hi = ml[cur:to], mh[cur:to]
                if pos.side is Side.LONG:
                    stop_hit = lo <= pos.stop
                    tgt_hit = hi >= pos.target if pos.target is not None else None
                else:
                    stop_hit = hi >= pos.stop
                    tgt_hit = lo <= pos.target if pos.target is not None else None
                hit = stop_hit if tgt_hit is None else (stop_hit | tgt_hit)
                hits = np.flatnonzero(hit)
                if hits.size:
                    j = int(hits[0])
                    idx = cur + j
                    exit_buy = pos.side is Side.SHORT
                    o = mo[idx]
                    if stop_hit[j]:
                        gapped = (o <= pos.stop) if pos.side is Side.LONG else (o >= pos.stop)
                        raw = o if gapped else pos.stop
                        close(pos, idx, raw, self._market_fill(raw, exit_buy), "stop")
                    else:
                        gapped = (o >= pos.target) if pos.side is Side.LONG else (o <= pos.target)
                        raw = o if gapped else pos.target
                        close(pos, idx, raw, self._limit_fill(raw, exit_buy), "target")
                    pos = None
            cur = max(cur, to)

        def equity_at(idx: int) -> float:
            if pos is None or idx < 0:
                return cash
            mtm = (mc[idx] - pos.entry_price) * pos.side.value * pos.units * inst.point_value
            return cash + mtm - pos.entry_commission

        decisions = 0
        for k in range(len(h_end)):
            f = int(fill_idx[k])
            if f >= n:
                break
            scan(f)
            eq = equity_at(f - 1)
            if day_id[f] != current_day:
                current_day = day_id[f]
                day_start_equity = eq

            if pos is None:
                view = PositionView()
            else:
                view = PositionView(
                    side=pos.side, units=pos.units, entry_price=pos.entry_price, stop=pos.stop,
                    target=pos.target, entry_time=bars.index[pos.entry_idx],
                    bars_held=k - pos.entry_k,
                )  # fmt: skip
            ctx = BarContext(
                i=k, time=pd.Timestamp(h_end[k], tz="UTC"), index=htf_bars.index,
                position=view, equity=eq, **h_arrays,
            )  # fmt: skip
            intents = mode.on_bar(ctx)
            decisions += 1
            if not intents:
                continue

            stale = (m_ns[f] - h_end[k]) > delay_ns
            for intent in intents:
                if isinstance(intent, MoveStop):
                    if pos is None:
                        continue
                    bad = self.risk.check_stop(pos.side, intent.stop, mc[f - 1])
                    if bad:
                        rejections[f"move_stop:{bad}"] += 1
                    else:
                        pos.stop = intent.stop
                elif isinstance(intent, Exit):
                    if pos is not None:
                        raw = mo[f]
                        close(pos, f, raw, self._market_fill(raw, pos.side is Side.SHORT),
                              intent.reason or "exit")  # fmt: skip
                        pos = None
                elif isinstance(intent, Enter):
                    if pos is not None:
                        rejections["already_in_position"] += 1
                        continue
                    if stale:
                        rejections["stale_data"] += 1
                        continue
                    raw = mo[f]
                    fill = self._market_fill(raw, intent.side is Side.LONG)
                    day_loss = max(0.0, (day_start_equity - eq) / day_start_equity)
                    decision = self.risk.approve_entry(intent, fill, eq, day_loss)
                    if not decision.approved:
                        rejections[decision.reason] += 1
                        continue
                    commission = decision.units * inst.commission_per_unit_side
                    pos = _Position(
                        side=intent.side, units=decision.units, entry_price=fill,
                        stop=float(intent.stop), initial_stop=float(intent.stop),
                        target=intent.target, entry_idx=f, entry_k=k,
                        entry_commission=commission,
                        entry_cost=abs(fill - raw) * decision.units * inst.point_value + commission,
                        reason=intent.reason,
                    )  # fmt: skip
                else:
                    raise TypeError(f"unknown intent {intent!r}")

        scan(n)
        if pos is not None:
            raw = mc[n - 1]
            close(pos, n - 1, raw, self._market_fill(raw, pos.side is Side.SHORT), "end_of_data")
            pos = None

        trade_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        return BacktestResult(
            mode=mode.describe(),
            symbol=inst.symbol,
            cost_multiplier=self.cost_multiplier,
            initial_equity=self.initial_equity,
            trades=trade_df,
            daily_equity=daily_equity(trade_df, bars, day_id, inst, self.initial_equity),
            rejections=dict(rejections),
            decisions=decisions,
        )


def daily_equity(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    day_id: np.ndarray,
    inst: Instrument,
    initial_equity: float,
) -> pd.Series:
    """Mark-to-market equity at the last bar of each trading day (17:00 ET roll)."""
    n = len(bars)
    if n == 0:
        return pd.Series(dtype=float)
    is_last = np.r_[day_id[1:] != day_id[:-1], True]
    d_idx = np.flatnonzero(is_last)
    mc = bars["close"].to_numpy()
    equity = np.full(len(d_idx), initial_equity, dtype=float)
    if len(trades):
        exit_idx = trades["exit_idx"].to_numpy()
        entry_idx = trades["entry_idx"].to_numpy()
        cum_pnl = np.r_[0.0, np.cumsum(trades["pnl"].to_numpy())]
        # Trades are sequential and non-overlapping, so exits are sorted.
        n_closed = np.searchsorted(exit_idx, d_idx, side="right")
        equity += cum_pnl[n_closed]
        # The trade after the last closed one is open at d if it has already been entered.
        nxt = n_closed
        open_mask = (nxt < len(trades)) & (entry_idx[np.minimum(nxt, len(trades) - 1)] <= d_idx)
        if open_mask.any():
            t = trades.iloc[nxt[open_mask]]
            commission_in = t["units"].to_numpy() * inst.commission_per_unit_side
            mtm = (
                (mc[d_idx[open_mask]] - t["entry_price"].to_numpy())
                * t["side"].to_numpy()
                * t["units"].to_numpy()
                * inst.point_value
            )
            equity[open_mask] += mtm - commission_in
    dates = pd.to_datetime(day_id[d_idx], unit="D")
    return pd.Series(equity, index=pd.DatetimeIndex(dates, name="trading_day"), name="equity")
