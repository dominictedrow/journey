"""Instrument contract specs and cost assumptions.

Cost numbers are ASSUMPTIONS until the owner supplies real broker figures
(see research/DATA_REQUIREMENTS.md, "Actual broker cost numbers").
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    # Account-currency P&L per 1.0 price move per unit (MES: $5/pt; EURUSD: $1 per unit).
    point_value: float
    min_units: float
    unit_step: float
    # Costs, applied symmetrically around the data price (treated as mid).
    half_spread: float  # price units, paid on every fill
    slippage: float  # price units, paid on market and stop fills (not on limit fills)
    commission_per_unit_side: float  # account currency per unit per side
    # Cap on notional exposure as a multiple of equity (margin proxy).
    max_leverage: float
    data_note: str = ""

    def scaled_costs(self, multiplier: float) -> Instrument:
        """Same instrument with every cost component multiplied (competition runs 1x/2x/3x)."""
        return replace(
            self,
            half_spread=self.half_spread * multiplier,
            slippage=self.slippage * multiplier,
            commission_per_unit_side=self.commission_per_unit_side * multiplier,
        )

    def round_trip_cost_per_unit(self) -> float:
        """Worst-case modeled round-trip cost (market in, stop out) per unit, account currency."""
        price_cost = 2 * (self.half_spread + self.slippage) * self.point_value
        return price_cost + 2 * self.commission_per_unit_side


# MES contract specs on the S&P 500 index CFD price series (USA500IDXUSD).
# ASSUMPTION: 1 tick spread, 1 tick slippage on market/stop fills, $0.62/side all-in commission.
MES = Instrument(
    symbol="USA500IDXUSD",
    tick_size=0.25,
    point_value=5.0,
    min_units=1,
    unit_step=1,
    half_spread=0.125,
    slippage=0.25,
    commission_per_unit_side=0.62,
    max_leverage=10.0,
    data_note="Index CFD price used as an ES/MES proxy; MES contract specs used for sizing/costs.",
)

# EUR/USD spot, units = EUR. ASSUMPTION: 1.0 pip spread, 0.2 pip slippage, no commission.
EURUSD = Instrument(
    symbol="EURUSD",
    tick_size=0.00001,
    point_value=1.0,
    min_units=1_000,
    unit_step=1_000,
    half_spread=0.00005,
    slippage=0.00002,
    commission_per_unit_side=0.0,
    max_leverage=20.0,
    data_note="Single-sided price (bid or mid, unconfirmed); spread is modeled, not observed.",
)

INSTRUMENTS: dict[str, Instrument] = {i.symbol: i for i in (MES, EURUSD)}


def get_instrument(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol]
    except KeyError as exc:
        raise KeyError(f"unknown instrument {symbol!r}; known: {sorted(INSTRUMENTS)}") from exc
