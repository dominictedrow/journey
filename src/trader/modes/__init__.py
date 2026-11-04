"""Mode registry: config refers to modes by ``name``."""

from __future__ import annotations

from trader.modes.base import Mode
from trader.modes.baselines import AlwaysLong, NoTrade, RandomEntry
from trader.modes.imom import IntradayMomentum
from trader.modes.joovier import SqueezeBreakout, SupportResistanceRetest, VwapHeikinAshi
from trader.modes.reversion import ShortTermReversion
from trader.modes.trend import TrendEnsemble
from trader.modes.tsm import TimeSeriesMomentum

REGISTRY: dict[str, type[Mode]] = {
    m.name: m
    for m in (
        NoTrade,
        AlwaysLong,
        RandomEntry,
        IntradayMomentum,
        TrendEnsemble,
        ShortTermReversion,
        SupportResistanceRetest,
        SqueezeBreakout,
        VwapHeikinAshi,
        TimeSeriesMomentum,
    )
}


def build_mode(name: str, **params) -> Mode:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown mode {name!r}; registered: {sorted(REGISTRY)}") from exc
    return cls(**params)


__all__ = ["REGISTRY", "Mode", "build_mode"]
