"""Rebalancer configuration and holdings (D-031). Plain TOML, validated strictly."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Rules:
    band_abs: float = 0.05  # rebalance when any weight drifts more than this from target
    min_trade_usd: float = 50.0  # skip smaller trades
    cash_target: float = 0.0  # fraction of the account kept in cash
    fractional: bool = True  # broker supports fractional shares
    max_price_age_days: int = 4  # older price -> NO TRADE
    max_daily_move: float = 0.20  # larger one-day move -> treat as bad data, NO TRADE
    drawdown_alert: float = 0.20  # warn when account is this far below its recorded peak


@dataclass(frozen=True)
class RebalanceConfig:
    targets: dict[str, float]
    rules: Rules = field(default_factory=Rules)

    @classmethod
    def from_toml(cls, path: Path) -> RebalanceConfig:
        raw = tomllib.loads(path.read_text())
        targets = {k.upper(): float(v) for k, v in raw.get("allocation", {}).items()}
        rules = Rules(**raw.get("rules", {}))
        cfg = cls(targets, rules)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        r = self.rules
        if not self.targets:
            raise ConfigError("allocation is empty")
        if any(not math.isfinite(w) or w < 0 for w in self.targets.values()):
            raise ConfigError("allocation weights must be finite and >= 0 (no shorting)")
        if not 0 <= r.cash_target < 1:
            raise ConfigError("cash_target must be in [0, 1)")
        total = sum(self.targets.values()) + r.cash_target
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"allocation + cash_target must sum to 1.0, got {total:.6f}")
        if not 0 < r.band_abs < 1:
            raise ConfigError("band_abs must be in (0, 1)")
        if r.min_trade_usd < 0 or r.max_price_age_days < 0 or r.max_daily_move <= 0:
            raise ConfigError("min_trade_usd, max_price_age_days must be >= 0; max_daily_move > 0")


@dataclass(frozen=True)
class Holdings:
    cash: float
    shares: dict[str, float]
    peak_value: float | None = None  # highest account value you've recorded, for the alert

    @classmethod
    def from_toml(cls, path: Path) -> Holdings:
        raw = tomllib.loads(path.read_text())
        cash = float(raw.get("cash", {}).get("usd", 0.0))
        shares = {k.upper(): float(v) for k, v in raw.get("positions", {}).items()}
        peak = raw.get("account", {}).get("peak_value")
        h = cls(cash, shares, float(peak) if peak is not None else None)
        h.validate()
        return h

    def validate(self) -> None:
        if not math.isfinite(self.cash) or self.cash < 0:
            raise ConfigError("cash must be finite and >= 0 (margin balances not supported)")
        bad = [k for k, v in self.shares.items() if not math.isfinite(v) or v < 0]
        if bad:
            raise ConfigError(f"positions must be >= 0 (no shorts): {bad}")
