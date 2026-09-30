"""The one preprocessing entry point: DICOM study folder → canonical uint8 volumes + metadata.

Used identically by the cache builder (train) and the Kaggle inference notebook (test), which is what makes
train/infer parity testable (tests/test_data.py::test_parity_cache_equals_direct).

Per series:
  1. read every file once (pixel bytes deferred, so memory stays small until a slice is decoded)
  2. keep the majority orientation; split off duplicate positions (several stacks/echoes in one series)
  3. order slices along the plane's canonical normal (IPP·n, never filename/InstanceNumber)
  4. decode → modality LUT → centre crop crop_mm (native px) → resize to the output scale
  5. per-series percentile window → uint8, pad to out_px × out_px
  6. in-plane reorientation to the canonical display; mirror left knees to right-knee anatomy
Per study: laterality (tag → geometry) and the slot assignment.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from knee.data import crop_resize as CR
from knee.data import geometry as G
from knee.data import normalize as N
from knee.data import series_meta as SM
from knee.data.dicom_io import decode_pixels, list_dicoms, read_slice, slice_geometry
from knee.data.series_select import assign_slots

log = logging.getLogger(__name__)

PREPROC_VERSION = "v1"
_DUP_TOL_MM = 0.05       # slices closer than this along the normal share a position
_ORIENT_TOL = 0.995      # |cos| between a slice normal and the series' majority normal (~5.7°)


@dataclass(frozen=True)
class PreprocParams:
    crop_mm: float = CR.CROP_MM
    out_px: int = CR.OUT_PX
    pct_lo: float = N.PCT_LO
    pct_hi: float = N.PCT_HI
    lat_dead_zone_mm: float = 20.0  # S01 part 2 (docs/data_audit.md §9): tag agreement 48% for |x| ≤ 20 mm, 98–100% above
    canonicalize_laterality: bool = True
    version: str = PREPROC_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessedSeries:
    series_uid: str
    pixels: np.ndarray       # (n, out_px, out_px) uint8, canonical orientation and slice order
    positions_mm: np.ndarray  # (n,) float32, ascending along the canonical normal (after the laterality mirror)
    valid: np.ndarray        # (n,) bool, False where the slice failed to decode (its pixels are 0)
    meta: dict = field(default_factory=dict)


@dataclass
class _SeriesHeaders:
    series_uid: str
    files: list               # SliceFile, already filtered and in canonical order
    positions: np.ndarray
    iop: np.ndarray
    plane: str
    meta: dict


def _first(values):
    for v in values:
        if v is not None:
            return v
    return None


def _mode(values):
    vals = [v for v in values if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _read_series(series_dir: Path, series_uid: str, plane_csv: str | None) -> _SeriesHeaders | None:
    paths = list_dicoms(series_dir)
    files = []
    for p in paths:
        s = read_slice(p, with_pixels=True)
        if s is None:
            continue
        ipp, iop, ps = slice_geometry(s.header)
        if ipp is None or iop is None or ps is None:
            log.warning("no geometry in %s; slice skipped", p)
            continue
        s.header["_ipp"], s.header["_iop"], s.header["_ps"] = ipp, iop, ps
        files.append(s)
    n_files = len(paths)
    if not files:
        log.warning("series %s: no usable slices", series_uid)
        return None

    # 1) majority orientation: keep the slices whose normal is parallel to the most common one.
    normals = np.stack([G.slice_normal(f.header["_iop"]) for f in files])
    ref = normals[np.argmax([np.sum(np.abs(normals @ n) > _ORIENT_TOL) for n in normals])]
    keep = np.abs(normals @ ref) > _ORIENT_TOL
    n_off_orient = int((~keep).sum())
    files = [f for f, k in zip(files, keep) if k]
    iop = files[0].header["_iop"]
    plane = G.plane_from_iop(iop)

    # 2) duplicate positions → several stacks/echoes interleaved. Keep the largest (EchoNumbers, AcquisitionNumber)
    #    group, then drop any remaining duplicates (lowest InstanceNumber wins).
    pos = G.canonical_positions(np.stack([f.header["_ipp"] for f in files]), iop, plane)
    n_dup_dropped = 0
    if len(pos) > 1 and np.min(np.diff(np.sort(pos))) < _DUP_TOL_MM:
        groups = Counter((f.header.get("EchoNumbers"), f.header.get("AcquisitionNumber")) for f in files)
        if len(groups) > 1:
            best = groups.most_common(1)[0][0]
            sel = [(f.header.get("EchoNumbers"), f.header.get("AcquisitionNumber")) == best for f in files]
            n_dup_dropped += len(files) - int(sum(sel))
            files = [f for f, k in zip(files, sel) if k]
            pos = pos[np.asarray(sel)]
        order = sorted(range(len(files)), key=lambda i: (round(pos[i] / _DUP_TOL_MM), _inst(files[i])))
        kept, last = [], None
        for i in order:
            b = round(pos[i] / _DUP_TOL_MM)
            if b != last:
                kept.append(i)
                last = b
        n_dup_dropped += len(files) - len(kept)
        files = [files[i] for i in kept]
        pos = pos[kept]

    # 3) canonical slice order
    order = np.argsort(pos, kind="stable")
    files = [files[i] for i in order]
    pos = pos[order]

    h = [f.header for f in files]
    tr, te, ti = _first(x["RepetitionTime"] for x in h), _first(x["EchoTime"] for x in h), \
        _first(x["InversionTime"] for x in h)
    desc = _first(x["SeriesDescription"] for x in h)
    mid = files[len(files) // 2].header
    center = G.image_center(mid["_ipp"], iop, int(mid["Rows"]), int(mid["Columns"]), mid["_ps"])
    sp = G.spacing_stats(pos)
    meta = {
        "series_uid": series_uid,
        "plane": plane,
        "plane_csv": plane_csv,
        "obliquity_deg": G.obliquity_deg(iop),
        "contrast_class": SM.contrast_class(tr, te, ti, _first(x["ScanningSequence"] for x in h), desc),
        "fat_sat_header": SM.fat_sat_from_header(_first(x["ScanOptions"] for x in h), desc),
        "tr": SM._num(tr), "te": SM._num(te), "ti": SM._num(ti),
        "field_strength": SM._num(_first(x["MagneticFieldStrength"] for x in h)),
        "manufacturer": SM.normalize_manufacturer(_first(x["Manufacturer"] for x in h)),
        "fingerprint": SM.scanner_fingerprint(_first(x["Manufacturer"] for x in h),
                                              _first(x["ManufacturerModelName"] for x in h),
                                              _first(x["MagneticFieldStrength"] for x in h),
                                              _first(x["SoftwareVersions"] for x in h)),
        "series_description": desc,
        "series_number": SM._num(_first(x["SeriesNumber"] for x in h)),
        "laterality_tag": _mode(str(x["Laterality"]).strip().upper() for x in h if x["Laterality"]),
        "center_x": float(center[0]),
        "n_files": n_files,
        "n_off_orientation": n_off_orient,
        "n_dup_dropped": n_dup_dropped,
        "slice_spacing_mm": sp["median_mm"],
        "slice_gap_ratio": sp["max_gap_ratio"],
        "pixel_spacing_mm": float(np.median([x["_ps"][0] for x in h])),
    }
    return _SeriesHeaders(series_uid, files, pos, iop, plane, meta)


def _inst(f) -> int:
    try:
        return int(f.header.get("InstanceNumber"))
    except (TypeError, ValueError):
        return 1 << 30


def _render_series(sh: _SeriesHeaders, mirror: bool, p: PreprocParams) -> ProcessedSeries:
    resized, valid = [], []
    for f in sh.files:
        img = decode_pixels(f.ds, f.path)
        f.ds = None  # free the pixel bytes as soon as the slice is decoded
        if img is None:
            resized.append(None)
            valid.append(False)
            continue
        crop = CR.crop_mm_native(img, f.header["_ps"], p.crop_mm)
        resized.append(CR.resize_crop(crop, f.header["_ps"], p.crop_mm, p.out_px))
        valid.append(True)
    lo, hi = N.series_window([r for r in resized if r is not None], p.pct_lo, p.pct_hi)

    ops = G.canonical_inplane_ops(sh.iop, sh.plane)
    mirror_cols = mirror and sh.plane in ("Axial", "Coronal")
    out = np.zeros((len(resized), p.out_px, p.out_px), dtype=np.uint8)
    for i, r in enumerate(resized):
        if r is not None:
            out[i] = G.apply_inplane_ops(CR.pad_center(N.to_uint8(r, lo, hi), p.out_px, 0), ops, mirror_cols)
    pos = sh.positions.astype(np.float32)
    valid_a = np.asarray(valid, dtype=bool)
    if mirror and sh.plane == "Sagittal":  # x → -x: reverse the slice order, keep positions ascending
        out, pos, valid_a = out[::-1].copy(), (-pos[::-1]).copy(), valid_a[::-1].copy()
    meta = {**sh.meta, "n_slices": int(len(out)), "n_decode_fail": int((~valid_a).sum()),
            "window_lo": lo, "window_hi": hi, "transpose": ops[0], "flip_rows": ops[1],
            "flip_cols": bool(ops[2] != mirror_cols), "mirrored": bool(mirror)}
    return ProcessedSeries(sh.series_uid, out, pos, valid_a, meta)


def preprocess_study(study_dir: str | os.PathLike, series_info: dict[str, dict] | None = None,
                     params: PreprocParams = PreprocParams(), series_uids: list[str] | None = None
                     ) -> tuple[list[ProcessedSeries], dict]:
    """Process every series of one study folder.

    series_info: {SeriesInstanceUID: {"fluid": 0/1, "plane": "Sagittal"...}} from train_series.csv / test_series.csv.
    series_uids: restrict to these series (default: every sub-folder).
    Returns (series in folder-name order, study meta with laterality and slots). Never raises for bad data: a series
    that can't be read is left out and logged.
    """
    study_dir = Path(study_dir)
    series_info = series_info or {}
    dirs = sorted(e.name for e in os.scandir(study_dir) if e.is_dir())
    if series_uids is not None:
        dirs = [d for d in dirs if d in set(series_uids)]
    headers = []
    for uid in dirs:
        try:
            sh = _read_series(study_dir / uid, uid, (series_info.get(uid) or {}).get("plane"))
        except Exception as e:  # noqa: BLE001 - one bad series must not kill the study
            log.exception("series %s failed in header pass: %s", uid, e)
            sh = None
        if sh is not None:
            headers.append(sh)

    lat, lat_src = G.resolve_study_laterality([s.meta["laterality_tag"] for s in headers],
                                              [s.meta["center_x"] for s in headers], params.lat_dead_zone_mm)
    mirror = params.canonicalize_laterality and lat == "L"
    out = []
    for sh in headers:
        try:
            ps = _render_series(sh, mirror, params)
        except Exception as e:  # noqa: BLE001
            log.exception("series %s failed in render: %s", sh.series_uid, e)
            continue
        info = series_info.get(sh.series_uid) or {}
        csv_fluid = info.get("fluid")
        ps.meta["fluid_csv"] = None if csv_fluid is None else int(csv_fluid)
        ps.meta["fluid"] = bool(csv_fluid) if csv_fluid is not None else bool(ps.meta["fat_sat_header"])
        out.append(ps)

    slots = assign_slots([{"series_uid": s.series_uid, "plane": s.meta["plane"], "fluid": s.meta["fluid"],
                           "contrast_class": s.meta["contrast_class"], "n_slices": int(s.valid.sum()),
                           "series_number": s.meta["series_number"]} for s in out])
    slot_of = {uid: slot for slot, uid in slots.items()}
    for s in out:
        s.meta["slot"] = slot_of.get(s.series_uid)
    study_meta = {"study_uid": study_dir.name, "laterality": lat, "laterality_source": lat_src,
                  "mirrored": bool(mirror), "n_series": len(out), "slots": slots}
    return out, study_meta


def series_info_from_csv(df) -> dict[str, dict]:
    """train_series.csv / test_series.csv rows → the series_info mapping preprocess_study expects."""
    return {str(r.SeriesInstanceUID): {"fluid": int(r.Fluid_Sensitive), "plane": str(r.Anatomical_Plane)}
            for r in df.itertuples(index=False)}
