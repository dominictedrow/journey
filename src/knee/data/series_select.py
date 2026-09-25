"""Series typing and slot assignment (S02 step 3).

Every series gets a type (plane, fluid flag, contrast class). Track A reads a fixed set of 5 slots:

    sag-fluid, sag-other, cor-fluid, cor-other, axial (fluid preferred)

Track B's bag reads *every* series; the slot is just one more tag on it.

"fluid" = the CSV `Fluid_Sensitive` flag, which in practice means **fat-suppressed** (it equals Fat_Suppression on all
24,371 train rows, and a non-FS T2 is 0 on the test stub; docs/data_audit_test.md §8). If the CSV flag is missing
(it exists at test time per the Data page, so this is insurance), the header fat-sat detection is used instead.
The contrast class comes from TR/TE/TI (series_meta.contrast_class).
"""
from __future__ import annotations

SLOTS = ["sag_fluid", "sag_other", "cor_fluid", "cor_other", "axial"]
PLANES = ["Sagittal", "Coronal", "Axial"]
CONTRASTS = ["PD", "T2", "T1", "STIR", "GRE", "UNK"]

# Within a slot, lower rank wins. HP: a placeholder preference (PD/T2 FS are the workhorse knee sequences; a 3D GRE
# has very thin slices, so a 2D series covers the joint better per slice). Ties break on more slices, then SeriesNumber.
_FLUID_RANK = {"PD": 0, "T2": 1, "STIR": 2, "GRE": 3, "UNK": 4, "T1": 5}
_OTHER_RANK = {"PD": 0, "T1": 1, "T2": 2, "GRE": 3, "UNK": 4, "STIR": 5}


def slot_candidates(plane: str, fluid: bool) -> list[str]:
    """Slots a series of this type may fill, in order of preference."""
    if plane == "Sagittal":
        return ["sag_fluid" if fluid else "sag_other"]
    if plane == "Coronal":
        return ["cor_fluid" if fluid else "cor_other"]
    if plane == "Axial":
        return ["axial"]
    return []


def _rank(s: dict, slot: str) -> tuple:
    fluid_first = slot == "axial"
    contrast_rank = (_FLUID_RANK if s["fluid"] else _OTHER_RANK).get(s["contrast_class"], 9)
    return (0 if (s["fluid"] or not fluid_first) else 1,  # axial: prefer a fluid series
            contrast_rank, -int(s["n_slices"]), int(s.get("series_number") or 1_000_000))


def assign_slots(series: list[dict]) -> dict[str, str]:
    """series: dicts with series_uid, plane, fluid (bool), contrast_class, n_slices, series_number.
    Returns {slot: series_uid} for the filled slots (at most one series per slot)."""
    out: dict[str, str] = {}
    for slot in SLOTS:
        cands = [s for s in series if s["n_slices"] > 0 and slot in slot_candidates(s["plane"], s["fluid"])]
        if cands:
            out[slot] = min(cands, key=lambda s: _rank(s, slot))["series_uid"]
    return out
