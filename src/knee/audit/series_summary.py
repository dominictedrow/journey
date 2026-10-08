"""S01 steps B4, B7–B9: turn the per-file header scan into per-series and per-study tables.

Per series: file/error counts, geometry (matrix, spacing, FOV, thickness, IPP-derived spacing, obliquity,
in-plane axis letters), slice-order checks (IPP vs InstanceNumber vs filename), contrast class, scanner
fingerprint, laterality (tag + geometry). Per study: laterality consensus and the site proxy.

Series never span shards, so this runs shard by shard and stays small in RAM.

    python -m knee.audit.series_summary --split train
Outputs in <audit_out>/: series_<split>.parquet, study_<split>.parquet, tags_<split>.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from knee.data import geometry as G
from knee.data.series_meta import contrast_class, fat_sat_from_header, normalize_manufacturer, scanner_fingerprint
from knee.paths import load_paths


def _mode(s: pd.Series):
    s = s.dropna()
    s = s[s.astype(str).str.strip() != ""]
    return s.mode().iloc[0] if len(s) else None


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _order_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of consecutive pairs (in order a) that are also increasing in b; 1 = same order, 0 = reversed."""
    if len(a) < 2:
        return np.nan
    idx = np.argsort(a, kind="stable")
    return float(np.mean(np.diff(b[idx]) > 0))


def summarize_series(g: pd.DataFrame) -> dict:
    ok = g[g.error.isna()]
    r: dict = {"n_files": len(g), "n_errors": int(g.error.notna().sum()),
               "n_truncated": int(g.truncated.fillna(False).astype(bool).sum())}
    if ok.empty:
        return r
    first = ok.iloc[0]
    r.update({
        "StudyInstanceUID": first.StudyInstanceUID,
        "transfer_syntaxes": "|".join(sorted(ok.transfer_syntax.dropna().unique())),
        "rows": _num(_mode(ok.Rows)), "cols": _num(_mode(ok.Columns)),
        "n_matrix_sizes": int(ok[["Rows", "Columns"]].drop_duplicates().shape[0]),
        "photometric": _mode(ok.PhotometricInterpretation), "bits_stored": _num(_mode(ok.BitsStored)),
        "multiframe": bool((ok.NumberOfFrames.dropna().astype(float) > 1).any()),
        "rescale_slope": _num(_mode(ok.RescaleSlope)), "rescale_intercept": _num(_mode(ok.RescaleIntercept)),
        "rescale_type": _mode(ok.RescaleType),
        "slice_thickness": _num(_mode(ok.SliceThickness)),
        "spacing_between_slices_tag": _num(_mode(ok.SpacingBetweenSlices)),
        "manufacturer": normalize_manufacturer(_mode(ok.Manufacturer)),
        "model": _mode(ok.ManufacturerModelName), "software": _mode(ok.SoftwareVersions),
        "field_strength": _num(_mode(ok.MagneticFieldStrength)),
        "series_description": _mode(ok.SeriesDescription),
        "tr": _num(_mode(ok.RepetitionTime)), "te": _num(_mode(ok.EchoTime)), "ti": _num(_mode(ok.InversionTime)),
        "scanning_sequence": _mode(ok.ScanningSequence), "scan_options": _mode(ok.ScanOptions),
        "mr_acq_type": _mode(ok.MRAcquisitionType),
        "laterality_tag": _mode(ok.Laterality), "patient_sex": _mode(ok.PatientSex),
        "patient_age_present": bool(ok.PatientAge.notna().any()),
        "n_echo_numbers": int(ok.EchoNumbers.nunique()), "n_acquisitions": int(ok.AcquisitionNumber.nunique()),
    })
    r["contrast_class"] = contrast_class(r["tr"], r["te"], r["ti"], r["scanning_sequence"], r["series_description"])
    r["fat_sat_header"] = fat_sat_from_header(r["scan_options"], r["series_description"])
    r["fingerprint"] = scanner_fingerprint(r["manufacturer"], r["model"], r["field_strength"], r["software"])

    iops = ok.ImageOrientationPatient.dropna().unique()
    r["n_distinct_iop"] = int(len(iops))
    iop = G.as_vec(_mode(ok.ImageOrientationPatient), 6)
    ps = G.as_vec(_mode(ok.PixelSpacing), 2)
    if ps is not None:
        r["pixel_spacing_row"], r["pixel_spacing_col"] = float(ps[0]), float(ps[1])
        r["fov_col_mm"] = float(ps[1] * r["cols"]) if np.isfinite(r["cols"]) else np.nan
        r["fov_row_mm"] = float(ps[0] * r["rows"]) if np.isfinite(r["rows"]) else np.nan
    if iop is None:
        return r
    r["plane_from_iop"] = G.plane_from_iop(iop)
    r["obliquity_deg"] = G.obliquity_deg(iop)
    r["row_axis"], r["col_axis"] = G.axis_letter(iop[:3]), G.axis_letter(iop[3:6])

    ipp_rows = [(G.as_vec(v, 3), i, name) for v, i, name in zip(ok.ImagePositionPatient, ok.InstanceNumber, ok.path)]
    ipp_rows = [(v, i, n) for v, i, n in ipp_rows if v is not None]
    if len(ipp_rows) < 2:
        return r
    ipp = np.stack([v for v, _, _ in ipp_rows])
    pos = G.slice_positions(ipp, iop)
    r.update({f"sp_{k}": v for k, v in G.spacing_stats(pos).items()})
    inst = np.array([_num(i) for _, i, _ in ipp_rows])
    if np.isfinite(inst).all():
        r["instance_vs_position"] = _order_agreement(inst, pos)  # 1 = InstanceNumber increases with position
    names = np.array([n.rsplit("/", 1)[-1] for _, _, n in ipp_rows])
    r["filename_vs_position"] = _order_agreement(np.argsort(np.argsort(names)), pos)
    if ps is not None and np.isfinite(r["rows"]) and np.isfinite(r["cols"]):
        centers = np.stack([G.image_center(p, iop, int(r["rows"]), int(r["cols"]), ps) for p in ipp])
        c = centers.mean(axis=0)
        r["center_x"], r["center_y"], r["center_z"] = float(c[0]), float(c[1]), float(c[2])
        r["laterality_geom"] = G.laterality_from_center_x(r["center_x"])
    return r


def summarize_studies(series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study, g in series.groupby("StudyInstanceUID"):
        tags = g.laterality_tag.dropna()
        tags = tags[tags.isin(["L", "R"])]
        tag_set = set(tags)
        cx = g.center_x.dropna() if "center_x" in g else pd.Series(dtype=float)
        med_x = float(cx.median()) if len(cx) else np.nan
        fp = g.fingerprint.dropna()
        rows.append({
            "StudyInstanceUID": study,
            "laterality_tag": tags.mode().iloc[0] if len(tags) else None,
            "laterality_tag_conflict": len(tag_set) > 1,
            "center_x_median": med_x,
            "laterality_geom": G.laterality_from_center_x(med_x),
            "site_proxy": fp.mode().iloc[0] if len(fp) else None,
            "n_fingerprints": int(fp.nunique()),
            "manufacturer": g.manufacturer.mode().iloc[0] if g.manufacturer.notna().any() else None,
            "field_strength": g.field_strength.median(),
            "patient_sex": g.patient_sex.dropna().mode().iloc[0] if g.patient_sex.notna().any() else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.data_root is None or a.out is None:
        p = load_paths()
        a.data_root = a.data_root or p.data_root
        a.out = a.out or p.audit_out
    scan_dir = a.out / f"header_scan_{a.split}"

    series_rows, counts, examples, other_files, n_files = [], Counter(), defaultdict(list), [], 0
    for pq in sorted(glob.glob(str(scan_dir / "part-*.parquet"))):
        df = pd.read_parquet(pq)
        n_files += len(df)
        df["series_dir"] = df.path.str.rsplit("/", n=1).str[0]
        for sd, g in df.groupby("series_dir"):
            r = summarize_series(g)
            r["series_dir"] = sd
            r["SeriesInstanceUID"] = sd.rsplit("/", 1)[-1]
            r.setdefault("StudyInstanceUID", sd.split("/", 1)[0])
            series_rows.append(r)
        info = json.loads(Path(pq.replace(".parquet", ".json")).read_text(encoding="utf-8"))
        counts.update(info["counts"])
        other_files += info["other_files"]
        for k, v in info["examples"].items():
            ex = examples[k]
            ex.extend(x for x in v if x not in ex and len(ex) < 8)

    series = pd.DataFrame(series_rows)
    csv = pd.read_csv(a.data_root / f"{a.split}_series.csv")
    series = series.merge(csv, on=["StudyInstanceUID", "SeriesInstanceUID"], how="outer", indicator="csv_match")
    series.to_parquet(a.out / f"series_{a.split}.parquet", index=False)
    studies = summarize_studies(series[series.csv_match != "right_only"])
    studies.to_parquet(a.out / f"study_{a.split}.parquet", index=False)
    (a.out / f"tags_{a.split}.json").write_text(json.dumps(
        {"n_files": n_files, "counts": counts, "examples": examples, "other_files": other_files}, indent=1), encoding="utf-8")
    print(f"{len(series)} series, {len(studies)} studies, {n_files} files, {len(counts)} distinct tags")
    print(series.csv_match.value_counts().to_string())


if __name__ == "__main__":
    main()
