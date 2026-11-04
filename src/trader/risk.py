"""Independent risk veto and position sizing.

Defaults are the PROPOSED (not yet owner-approved) numbers in docs/planning/RISK_POLICY_PROPOSAL.md:
0.25% equity at stop per position, 1% daily loss breaker. Sizing formula follows that file:
units = floor(risk_budget / (stop_distance * point_value + round_trip_cost_per_unit)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trader.instruments import Instrument
from trader.modes.base import Enter, Side


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.0025
    daily_loss_limit: float = 0.01  # block new entries for the rest of the trading day
    min_stop_ticks: int = 4
    allow_short: bool = True


@dataclass(frozen=True)
class Decision:
    approved: bool
    units: float = 0.0
    reason: str = ""


def reject(reason: str) -> Decision:
    return Decision(False, 0.0, reason)


class RiskLayer:
    def __init__(self, config: RiskConfig, instrument: Instrument) -> None:
        self.config = config
        self.instrument = instrument

    def check_stop(self, side: Side, stop: float | None, price: float) -> str | None:
        """Rejection reason, or None if ``stop`` is a valid protective stop at ``price``."""
        if stop is None or not math.isfinite(stop):
            return "no_stop"
        distance = (price - stop) * side.value
        if distance <= 0:
            return "stop_wrong_side"
        if distance < self.config.min_stop_ticks * self.instrument.tick_size:
            return "stop_too_tight"
        return None

    def approve_entry(
        self, intent: Enter, fill_price: float, equity: float, day_loss_frac: float
    ) -> Decision:
        """Size an entry at the actual fill price, or veto it."""
        if intent.side is Side.FLAT:
            return reject("flat_entry")
        if intent.side is Side.SHORT and not self.config.allow_short:
            return reject("short_disabled")
        if day_loss_frac >= self.config.daily_loss_limit:
            return reject("daily_loss_breaker")
        if equity <= 0:
            return reject("no_equity")
        bad = self.check_stop(intent.side, intent.stop, fill_price)
        if bad:
            return reject(bad)
        if intent.target is not None and (intent.target - fill_price) * intent.side.value <= 0:
            return reject("target_wrong_side")

        inst = self.instrument
        distance = abs(fill_price - intent.stop)  # type: ignore[operator]
        per_unit_risk = distance * inst.point_value + inst.round_trip_cost_per_unit()
        raw = equity * self.config.risk_per_trade / per_unit_risk
        max_by_leverage = equity * inst.max_leverage / (fill_price * inst.point_value)
        units = math.floor(min(raw, max_by_leverage) / inst.unit_step) * inst.unit_step
        if units < inst.min_units:
            return reject("below_min_size")  # never round up (RISK_POLICY_PROPOSAL)
        return Decision(True, float(units), "ok")
