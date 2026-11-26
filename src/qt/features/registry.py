"""Feature registry and versioning (component doc 03 section 8, D-033 Q7).

Each feature is a function registered with :func:`feature`. ``(name, version)`` identifies one
exact computation: its definition (formula text, window, sources, group) is immutable once it is in
``feature_registry`` (migration 008 enforces that in the DB as well). Changing any of them means a
new version, written alongside the old one.

Which versions get computed each day: the newest version of every feature, plus any older version
that an active model still pins. An older version nobody pins is marked ``frozen``; its history
stays in ``features``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg

Kind = Literal["asset", "market"]

# Phase 7 creates the model pin table; until then nothing is pinned.
PIN_TABLE = "model_feature_pins"


class RegistryError(RuntimeError):
    """A registered definition disagrees with what the DB already holds for that version."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    kind: Kind
    group: str
    formula: str
    window: str | None
    sources: tuple[str, ...]
    fn: Callable[..., Any] = field(compare=False, repr=False)

    @property
    def key(self) -> tuple[str, int]:
        return (self.name, self.version)

    def definition(self) -> tuple[str, str | None, tuple[str, ...], str]:
        return (self.formula, self.window, self.sources, self.group)


_REGISTRY: dict[tuple[str, int], FeatureSpec] = {}


def feature(
    name: str,
    version: int,
    *,
    kind: Kind,
    group: str,
    formula: str,
    window: str | int | None = None,
    sources: Iterable[str] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` as feature ``name`` v``version``. ``formula`` is the human definition."""
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        raise ValueError(f"feature name {name!r} must be lower_snake_case")
    if version < 1:
        raise ValueError("feature versions start at 1")

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = FeatureSpec(
            name=name,
            version=version,
            kind=kind,
            group=group,
            formula=" ".join(formula.split()),
            window=None if window is None else str(window),
            sources=tuple(sources),
            fn=fn,
        )
        old = _REGISTRY.get(spec.key)
        # Re-registering the same definition (config-driven families registered again) is fine;
        # two different definitions under one (name, version) are not.
        if old is not None and old.fn is not fn and (old.definition(), old.kind) != (
            spec.definition(),
            spec.kind,
        ):
            raise RegistryError(f"feature {name} v{version} registered twice")
        _REGISTRY[spec.key] = spec
        return fn

    return wrap


def all_specs() -> list[FeatureSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: (s.kind, s.group, s.name, s.version))


def current_specs() -> list[FeatureSpec]:
    """The newest registered version of each feature."""
    newest: dict[str, FeatureSpec] = {}
    for s in _REGISTRY.values():
        if s.name not in newest or s.version > newest[s.name].version:
            newest[s.name] = s
    return sorted(newest.values(), key=lambda s: (s.kind, s.group, s.name))


def get(name: str, version: int) -> FeatureSpec:
    try:
        return _REGISTRY[(name, version)]
    except KeyError:
        raise KeyError(f"feature {name} v{version} is not registered") from None


def pinned_versions(conn: psycopg.Connection) -> set[tuple[str, int]]:
    """``(feature, version)`` pairs pinned by active models (empty until Phase 7 adds the table)."""
    exists = conn.execute("SELECT to_regclass(%s)", (PIN_TABLE,)).fetchone()
    if not exists or exists[0] is None:
        return set()
    rows = conn.execute(
        f"SELECT DISTINCT feature_name, feature_version FROM {PIN_TABLE} WHERE active"
    ).fetchall()
    return {(r[0], int(r[1])) for r in rows}


def to_compute(conn: psycopg.Connection) -> list[FeatureSpec]:
    """Current versions plus every pinned older version that is still registered in code."""
    want = {s.key: s for s in current_specs()}
    for key in pinned_versions(conn):
        if key in want:
            continue
        if key not in _REGISTRY:
            raise RegistryError(
                f"a model pins {key[0]} v{key[1]}, but that version is no longer in the code"
            )
        want[key] = _REGISTRY[key]
    return sorted(want.values(), key=lambda s: (s.kind, s.group, s.name, s.version))


def sync(conn: psycopg.Connection, *, changelog: dict[tuple[str, int], str] | None = None) -> int:
    """Insert unseen versions into ``feature_registry``; fail on any changed definition.

    Then freeze versions that are neither current nor pinned. Returns the number inserted.
    """
    rows = conn.execute(
        'SELECT feature_name, version, formula, "window", source_series, feature_group, status '
        "FROM feature_registry"
    ).fetchall()
    stored = {(r[0], int(r[1])): r for r in rows}
    mismatched = []
    inserted = 0
    for s in all_specs():
        row = stored.get(s.key)
        if row is None:
            conn.execute(
                'INSERT INTO feature_registry (feature_name, version, formula, "window", '
                "source_series, feature_group, changelog) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    s.name,
                    s.version,
                    s.formula,
                    s.window,
                    list(s.sources),
                    s.group,
                    (changelog or {}).get(s.key),
                ),
            )
            inserted += 1
        elif (row[2], row[3], tuple(row[4] or ()), row[5]) != s.definition():
            mismatched.append(f"{s.name} v{s.version}")
    if mismatched:
        raise RegistryError(
            "definition changed without a version bump: " + ", ".join(sorted(mismatched))
        )

    live = {s.key for s in to_compute(conn)}
    for key, row in stored.items():
        if key not in live and row[6] == "active":
            conn.execute(
                "UPDATE feature_registry SET status = 'frozen' "
                "WHERE feature_name = %s AND version = %s",
                key,
            )
    return inserted


def _clear_for_tests() -> dict[tuple[str, int], FeatureSpec]:
    """Swap out the registry (tests only); returns the old contents for restoring."""
    global _REGISTRY
    old, _REGISTRY = _REGISTRY, {}
    return old


def _restore_for_tests(saved: dict[tuple[str, int], FeatureSpec]) -> None:
    global _REGISTRY
    _REGISTRY = saved
