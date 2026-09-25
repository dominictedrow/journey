"""Series-level metadata derived from DICOM headers: contrast class, fat suppression, scanner/site proxy.

The CSV gives one bit per series (Fluid_Sensitive == Fat_Suppression on every train row), so the real contrast
has to come from the header. All thresholds here are placeholders from public notebooks ([101], [KITO], [RR]);
S01's audit measures how they line up with the CSV flag. UNVERIFIED as final rules; S02 may refine them.
"""
from __future__ import annotations

import hashlib
import math
import re

# ScanOptions tokens that mean fat suppression. Matched as whole tokens: GE's 'SAT_GEMS' is spatial saturation,
# not fat-sat, so substring matching is wrong ([CACHE]). UNVERIFIED completeness per vendor.
FAT_SAT_SCAN_OPTIONS = {"FS", "SFS", "FAT", "FATSAT", "FAT_SAT", "SPIR", "SPAIR", "CHESS", "WE"}
# Series-description tokens (after lower-casing and splitting on non-alphanumerics).
FAT_SAT_DESC_TOKENS = {"fs", "fatsat", "fat", "spir", "spair", "stir", "tirm", "we", "dixon", "chess", "sfs", "fsat"}

T1_MAX_TR_MS = 800.0   # TR below this → T1-weighted (HP; [101])
T2_MIN_TE_MS = 60.0    # long-TR series with TE at or above this → T2, below → PD (HP; [101])


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _tokens(s) -> set[str]:
    if s is None:
        return set()
    return {t for t in re.split(r"[^0-9a-z]+", str(s).lower()) if t}


def fat_sat_from_header(scan_options, series_description) -> bool:
    opts = {t.strip().upper() for t in re.split(r"[\\,\s]+", str(scan_options or "")) if t.strip()}
    return bool(opts & FAT_SAT_SCAN_OPTIONS) or bool(_tokens(series_description) & FAT_SAT_DESC_TOKENS)


def contrast_class(tr, te, ti, scanning_sequence, series_description=None) -> str:
    """One of T1, PD, T2, STIR, GRE, UNK. STIR = inversion recovery with a short TI (fat nulling)."""
    tr, te, ti = _num(tr), _num(te), _num(ti)
    seq = str(scanning_sequence or "").upper()
    desc = _tokens(series_description)
    if ("IR" in seq and ti is not None and 0 < ti < 300) or ({"stir", "tirm"} & desc):
        return "STIR"
    if "GR" in seq and "SE" not in seq:
        return "GRE"
    if tr is None or te is None:
        # Some vendors (Toshiba in the test stub) drop TR/TE; fall back to the description ('SG PD FatSat', 'CO T2W FS').
        for cls, toks in (("PD", {"pd", "pdw", "dp"}), ("T2", {"t2", "t2w"}), ("T1", {"t1", "t1w"})):
            if desc & toks:
                return cls
        return "UNK"
    if tr < T1_MAX_TR_MS:
        return "T1"
    return "T2" if te >= T2_MIN_TE_MS else "PD"


def normalize_manufacturer(m) -> str:
    s = str(m or "").lower()
    for key, name in (("siemens", "Siemens"), ("philips", "Philips"), ("toshiba", "Toshiba"),
                      ("canon", "Canon"), ("hitachi", "Hitachi"), ("fujifilm", "Fujifilm")):
        if key in s:
            return name
    if s.startswith("ge") or "general electric" in s:
        return "GE"
    return "Unknown" if not s.strip() else s.strip()


def scanner_fingerprint(manufacturer, model, field_strength, software) -> str:
    """Scanner identity used as a site proxy for grouped CV. There is no InstitutionName tag, so this is the
    closest thing to 'site' the headers allow ([KITO]: 112 groups over 4,407 studies)."""
    fs = _num(field_strength)
    fs_s = f"{fs:.1f}T" if fs is not None else "?T"
    return "|".join([normalize_manufacturer(manufacturer), str(model or "?").strip(), fs_s, str(software or "?").strip()])


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]
