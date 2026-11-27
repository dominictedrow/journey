"""Cross-sectional ranks and composite factors (component doc 03 section 5.2, Q8).

At one decision time T: each per-asset feature in ``cross_section.rank_groups`` becomes a
percentile rank in [0, 1] across the point-in-time index members at T (``<name>_xs``) and again
within the member's sector (``<name>_xs_sec``). Ties take the average rank, NaN stays out of the
rank and stays NaN. A rank is ``(average rank - 1) / (valid count - 1)``, and 0.5 for a single
valid value.

Composites (``comp_<name>``) are the mean of their components' universe ranks, where a ``-x``
component uses ``1 - rank(x)``: lower is better. A composite needs ``min_components`` of them.

Inputs are the per-asset values already computed for T, so these functions are pure: the
point-in-time work happened upstream (price/fundamentals ``ready_at``, membership as of T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qt.common.config import CrossSectionCfg
from qt.features import registry

GROUP_RANK = "xs_rank"
GROUP_SECTOR = "xs_sector_rank"
GROUP_COMPOSITE = "composite"


@dataclass
class Panel:
    """Per-asset values at T for the index members (index: asset_id)."""

    values: pd.DataFrame  # columns: base feature names
    sectors: pd.Series  # asset_id -> sector (None/NaN = unknown)
    cfg: CrossSectionCfg

    def __post_init__(self) -> None:
        self._cache: dict[str, pd.Series] = {}

    def rank(self, name: str) -> pd.Series:
        key = f"u:{name}"
        if key not in self._cache:
            self._cache[key] = pct_rank(self.values[name])
        return self._cache[key]

    def sector_rank(self, name: str) -> pd.Series:
        key = f"s:{name}"
        if key not in self._cache:
            x = self.values[name]
            sec = self.sectors.reindex(x.index)
            out = pd.Series(np.nan, index=x.index)
            for _s, idx in x.groupby(sec, dropna=True).groups.items():
                g = x.loc[idx]
                if g.notna().sum() >= self.cfg.min_sector_size:
                    out.loc[idx] = pct_rank(g)
            self._cache[key] = out
        return self._cache[key]

    def composite(self, components: list[str]) -> pd.Series:
        parts = []
        for c in components:
            r = self.rank(c.lstrip("-"))
            parts.append(1.0 - r if c.startswith("-") else r)
        m = pd.concat(parts, axis=1)
        n = m.notna().sum(axis=1)
        return m.mean(axis=1).where(n >= self.cfg.min_components)


def pct_rank(x: pd.Series) -> pd.Series:
    """Average-rank percentile in [0, 1]; NaN excluded and kept."""
    valid = x.notna()
    n = int(valid.sum())
    r = x.rank(method="average")
    if n == 0:
        return pd.Series(np.nan, index=x.index)
    if n == 1:
        return pd.Series(np.where(valid, 0.5, np.nan), index=x.index)
    return (r - 1.0) / (n - 1.0)


def register(cfg: CrossSectionCfg, base: list[registry.FeatureSpec]) -> list[registry.FeatureSpec]:
    """Register ranks for every ``base`` spec in the rank groups, plus the composites."""
    keys: list[tuple[str, int]] = []
    ranked: dict[str, registry.FeatureSpec] = {}
    for s in base:  # the newest version of each ranked base
        newer = s.name not in ranked or s.version > ranked[s.name].version
        if s.group in cfg.rank_groups and newer:
            ranked[s.name] = s
    for name, s in sorted(ranked.items()):
        v = s.version  # a rank takes its base's version: a base bump bumps its ranks
        registry.feature(
            f"{name}_xs", v, kind="asset", group=GROUP_RANK,
            formula=f"percentile rank (average ties, NaN kept) of {name} v{s.version} across "
            f"the {cfg.index_code} members at T", sources=(f"xs:{name}",),
        )(lambda p, name=name: p.rank(name))  # fmt: skip
        registry.feature(
            f"{name}_xs_sec", v, kind="asset", group=GROUP_SECTOR,
            formula=f"percentile rank of {name} v{s.version} within the member's sector "
            f"(NaN if < {cfg.min_sector_size} valid in the sector)", sources=(f"xs:{name}",),
        )(lambda p, name=name: p.sector_rank(name))  # fmt: skip
        keys += [(f"{name}_xs", v), (f"{name}_xs_sec", v)]
    for comp, parts in cfg.composites.items():
        missing = [c.lstrip("-") for c in parts if c.lstrip("-") not in ranked]
        if missing:
            raise registry.RegistryError(f"composite {comp}: no ranked feature {missing}")
        vers = ", ".join(f"{c} v{ranked[c.lstrip('-')].version}" for c in parts)
        # 1 + every component bump since v1, so bumping any component bumps the composite
        cv = 1 + sum(ranked[c.lstrip("-")].version - 1 for c in parts)
        registry.feature(
            f"comp_{comp}", cv, kind="asset", group=GROUP_COMPOSITE,
            formula=f"mean of universe ranks of [{vers}] ('-x' = 1 - rank); needs >= "
            f"{cfg.min_components}", sources=tuple(f"xs:{c.lstrip('-')}" for c in parts),
        )(lambda p, parts=parts: p.composite(parts))  # fmt: skip
        keys.append((f"comp_{comp}", cv))
    return [registry.get(*k) for k in keys]


def compute(panel: Panel, specs: list[registry.FeatureSpec]) -> pd.DataFrame:
    """Every spec over the panel: index asset_id, one column per spec key."""
    cols = pd.Index([s.key for s in specs], tupleize_cols=False)
    data = np.column_stack([s.fn(panel).reindex(panel.values.index).to_numpy(dtype=float)
                            for s in specs])  # fmt: skip
    return pd.DataFrame(data, index=panel.values.index, columns=cols)
