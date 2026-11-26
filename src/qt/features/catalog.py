"""Every feature family, registered once from ``configs/features.yaml`` (component doc 03 §8).

:func:`register_all` is the single place the build, the matrix reader and the registry sync get
their specs from, grouped by how they are computed:

- ``price`` / ``fund`` / ``masks``: per-asset row functions of a ``PriceFrame``
- ``asset_tsz``: per-asset z of a price feature (ETF and crypto only)
- ``xs``: cross-sectional ranks and composites of a ``Panel`` (index members only)
- ``market``: functions of a ``MarketState`` (macro, cross-asset, events, crypto, regime,
  staleness)
- ``market_tsz``: z of a market feature over its own decision-time history
"""

from __future__ import annotations

from dataclasses import dataclass

from qt.common.config import FeaturesCfg
from qt.features import (
    cross_section,
    crypto,
    events,
    fundamentals,
    market,
    normalize,
    price,
    registry,
)
from qt.features import regime as regime_mod


@dataclass(frozen=True)
class Catalog:
    price: list[registry.FeatureSpec]
    fund: list[registry.FeatureSpec]
    masks: list[registry.FeatureSpec]
    asset_tsz: list[registry.FeatureSpec]
    xs: list[registry.FeatureSpec]
    market: list[registry.FeatureSpec]
    market_tsz: list[registry.FeatureSpec]

    def all(self) -> list[registry.FeatureSpec]:
        return (self.price + self.fund + self.masks + self.asset_tsz + self.xs + self.market
                + self.market_tsz)  # fmt: skip

    def keys(self) -> set[tuple[str, int]]:
        return {s.key for s in self.all()}


def _newest(specs: list[registry.FeatureSpec]) -> list[registry.FeatureSpec]:
    best: dict[str, registry.FeatureSpec] = {}
    for s in specs:
        if s.name not in best or s.version > best[s.name].version:
            best[s.name] = s
    return [s for s in specs if best[s.name] is s]


def register_all(cfg: FeaturesCfg) -> Catalog:
    """Register every family (idempotent) and return the current version of each feature.

    Older versions stay registered (pinned models may still need them); see ``registry.to_compute``.
    """
    pr = price.register(cfg.price)
    fu = fundamentals.register(cfg.fundamentals)
    ma = normalize.register_masks(cfg.normalize)
    xs = cross_section.register(cfg.cross_section, pr + fu)
    mk = (market.register(cfg.market) + events.register(cfg.market) + crypto.register(cfg.market)
          + regime_mod.register(cfg.market, cfg.regime))  # fmt: skip
    mk = mk + normalize.register_staleness(mk, cfg.normalize)
    n = cfg.normalize
    a_tsz = normalize.register_ts_z(_newest(pr), n, n.asset_groups)
    m_tsz = normalize.register_ts_z(_newest(mk), n, n.market_groups)
    return Catalog(price=_newest(pr), fund=_newest(fu), masks=ma, asset_tsz=a_tsz, xs=xs,
                   market=_newest(mk), market_tsz=m_tsz)  # fmt: skip
