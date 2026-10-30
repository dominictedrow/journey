"""Strategy-mode contract (MASTER_PLAN v10).

A mode sees only COMPLETED bars of its own timeframe and returns intents. Every entry intent must
carry a protective stop (D-015); the risk layer rejects any that do not. Modes never size
positions or touch fills: sizing is the risk layer's job, fills are the engine's.

``prepare`` may precompute indicators vectorised over the whole frame, but each value at row i
must depend only on rows <= i. ``tests/test_no_lookahead.py`` checks this for every registered mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class Provenance(str, Enum):
    SOURCE = "SOURCE"  # rule stated by a primary source
    ADAPTATION = "ADAPTATION"  # source rule adapted to our market/timeframe
    INTERPRETATION = "INTERPRETATION"  # our reading of an ambiguous rule
    BASELINE = "BASELINE"  # control, not a strategy claim


class Side(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class Enter:
    side: Side
    stop: float | None
    target: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Exit:
    reason: str = ""


@dataclass(frozen=True)
class MoveStop:
    stop: float
    reason: str = ""


Intent = Enter | Exit | MoveStop


@dataclass(frozen=True)
class PositionView:
    side: Side = Side.FLAT
    units: float = 0.0
    entry_price: float = float("nan")
    stop: float = float("nan")
    target: float | None = None
    entry_time: pd.Timestamp | None = None
    bars_held: int = 0  # completed mode-timeframe bars since entry


@dataclass
class BarContext:
    """What a mode may see when bar ``i`` (its own timeframe) has just completed."""

    i: int
    time: pd.Timestamp  # decision time = end of bar i (UTC)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    index: pd.DatetimeIndex
    position: PositionView
    equity: float

    def history(self, field_name: str, n: int) -> np.ndarray:
        """Last ``n`` values of a bar field up to and including bar i."""
        arr = getattr(self, field_name)
        return arr[max(0, self.i - n + 1) : self.i + 1]


class Mode(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    provenance: ClassVar[Provenance]
    instruments: ClassVar[tuple[str, ...]] = ("*",)  # "*" = any instrument
    timeframe: ClassVar[str] = "1h"
    # Longest wait from decision to fill before an entry counts as stale. Daily modes need
    # enough to span the scheduled 17:00-18:00 ET futures halt. None = engine default (5min).
    max_entry_delay: ClassVar[str | None] = None

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = {**self.default_params(), **params}
        unknown = set(params) - set(self.default_params())
        if unknown:
            raise TypeError(f"{self.name}: unknown params {sorted(unknown)}")
        self.feat: dict[str, np.ndarray] = {}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    def supports(self, symbol: str) -> bool:
        return "*" in self.instruments or symbol in self.instruments

    def prepare(self, bars: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Optional causal precompute over this mode's timeframe bars."""

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Intent]: ...

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "provenance": self.provenance.value,
            "instruments": list(self.instruments),
            "timeframe": self.timeframe,
            "params": dict(self.params),
        }
