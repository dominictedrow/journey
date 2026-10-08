"""S01 steps B4–B5: read every DICOM header in the corpus (no pixels) and inventory every tag.

For each file: a row of the fields we need (geometry, sequence, scanner, transfer syntax, size) plus a
truncation check (file shorter than header + expected pixel bytes). For the whole corpus: how many files
carry each tag, with example values. Work is split into shards of series; each shard writes its own
parquet + json, so an interrupted scan resumes where it stopped.

    python -m knee.audit.header_scan --split train --workers 12
    python -m knee.audit.header_scan --split test

Outputs in <audit_out>/header_scan_<split>/: part-XXXXX.parquet (one row per file), part-XXXXX.json (tag counts).
Runtime estimate: public notebooks read 73k headers in 2.7 min with a process pool ([GC]); the full 819k
should take roughly 30–60 min on a SATA SSD. UNVERIFIED for this machine; the script prints its rate.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

from knee.paths import load_paths

# Fields kept per file. Multi-valued ones are stored as backslash-joined strings.
FIELDS = [
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "InstanceNumber", "AcquisitionNumber", "EchoNumbers",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "SliceLocation", "Rows", "Columns", "NumberOfFrames", "SamplesPerPixel", "BitsAllocated", "BitsStored",
    "PixelRepresentation", "PhotometricInterpretation", "RescaleSlope", "RescaleIntercept", "RescaleType",
    "WindowCenter", "WindowWidth", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "MagneticFieldStrength", "ReceiveCoilName", "SeriesDescription", "SeriesNumber", "ImageType", "Laterality",
    "BodyPartExamined", "PatientSex", "PatientAge", "PatientPosition", "RepetitionTime", "EchoTime", "InversionTime",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType", "EchoTrainLength", "FlipAngle",
    "FrameOfReferenceUID",
]
# Never copy these values into committed docs (identifiers).
REDACT_VR = {"UI"}
REDACT_KEYWORDS = {"PatientID"}
MAX_EXAMPLES = 8


def _str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in v)
    return str(v)


def _tag_name(elem) -> str:
    if elem.keyword:
        return elem.keyword
    g = elem.tag.group
    if 0x6000 <= g <= 0x60FF and g % 2 == 0:
        return f"Overlay({g:04X},{elem.tag.element:04X})"
    return f"({g:04X},{elem.tag.element:04X}){'-private' if elem.tag.is_private else ''}"


def scan_file(path: Path, rel: str, counts: Counter, examples: dict) -> dict:
    row: dict = {"path": rel, "file_size": None, "error": None}
    try:
        row["file_size"] = path.stat().st_size
        with open(path, "rb") as fp:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            header_end = fp.tell()  # start of the PixelData element (verified on the test stub: size - pos - 12 == pixels)
    except InvalidDicomError as e:
        row["error"] = f"InvalidDicom: {e}"[:200]
        return row
    except Exception as e:  # noqa: BLE001 - the audit must record, not crash
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row

    meta = getattr(ds, "file_meta", None)
    ts = str(getattr(meta, "TransferSyntaxUID", "")) if meta is not None else ""
    row["transfer_syntax"] = ts
    for elem in list(meta or []) + list(ds):
        name = _tag_name(elem)
        counts[name] += 1
        ex = examples[name]
        if len(ex) < MAX_EXAMPLES and elem.VR not in REDACT_VR and name not in REDACT_KEYWORDS and elem.VR != "SQ":
            s = _str(elem.value)
            if s is not None and len(s) < 80 and s not in ex:
                ex.append(s)
    for f in FIELDS:
        row[f] = _str(ds[f].value) if f in ds else None

    # Truncation check for uncompressed syntaxes: pixels = rows*cols*bytes*frames*samples after a 12-byte element header.
    try:
        frames = int(ds.get("NumberOfFrames", 1) or 1)
        expected = int(ds.Rows) * int(ds.Columns) * (int(ds.BitsAllocated) // 8) * frames * int(ds.get("SamplesPerPixel", 1))
        row["header_bytes"] = header_end
        row["pixel_bytes_expected"] = expected
        if ts in ("1.2.840.10008.1.2.1", "1.2.840.10008.1.2"):
            hdr = 12 if ts.endswith(".1") else 8
            row["truncated"] = row["file_size"] < header_end + hdr + expected
        else:
            row["truncated"] = None  # compressed: only a decode can tell
    except Exception as e:  # noqa: BLE001
        row["truncated"] = None
        row["error"] = f"size-check {type(e).__name__}: {e}"[:200]
    return row


def scan_shard(shard_id: int, series_dirs: list[str], root: str, out_dir: str) -> dict:
    root_p, out_p = Path(root), Path(out_dir)
    counts: Counter = Counter()
    examples: dict = defaultdict(list)
    rows, other_files = [], []
    t0 = time.time()
    for sd in series_dirs:
        for entry in os.scandir(root_p / sd):
            if not entry.is_file():
                other_files.append(f"{sd}/{entry.name} (dir)")
                continue
            rel = f"{sd}/{entry.name}"
            if not entry.name.lower().endswith(".dcm"):
                other_files.append(rel)
            rows.append(scan_file(Path(entry.path), rel, counts, examples))
    df = pd.DataFrame(rows)
    df.to_parquet(out_p / f"part-{shard_id:05d}.parquet", index=False)
    info = {"shard": shard_id, "n_series": len(series_dirs), "n_files": len(rows), "counts": counts,
            "examples": examples, "other_files": other_files, "seconds": time.time() - t0}
    (out_p / f"part-{shard_id:05d}.json").write_text(json.dumps(info), encoding="utf-8")
    return {"shard": shard_id, "n_files": len(rows), "seconds": info["seconds"]}


def list_series_dirs(split_dir: Path) -> tuple[list[str], list[str]]:
    """Relative 'study/series' paths, plus anything unexpected at the study level (loose files)."""
    series, loose = [], []
    for st in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not st.is_dir():
            loose.append(st.name)
            continue
        for se in sorted(os.scandir(st.path), key=lambda e: e.name):
            (series if se.is_dir() else loose).append(f"{st.name}/{se.name}")
    return series, loose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--series-per-shard", type=int, default=200)
    ap.add_argument("--limit-series", type=int, default=0, help="scan only the first N series (smoke test)")
    a = ap.parse_args()
    if a.data_root is None or a.out is None:
        p = load_paths()
        a.data_root = a.data_root or p.data_root
        a.out = a.out or p.audit_out

    split_dir = a.data_root / f"{a.split}_series"
    out_dir = a.out / f"header_scan_{a.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    series, loose = list_series_dirs(split_dir)
    if a.limit_series:
        series = series[: a.limit_series]
    shards = [series[i: i + a.series_per_shard] for i in range(0, len(series), a.series_per_shard)]
    todo = [i for i in range(len(shards)) if not (out_dir / f"part-{i:05d}.json").exists()]
    (out_dir / "_listing.json").write_text(json.dumps({"n_series": len(series), "loose_entries": loose,
                                                        "series_per_shard": a.series_per_shard}), encoding="utf-8")
    print(f"{len(series)} series in {len(shards)} shards; {len(todo)} to do; {len(loose)} loose entries; "
          f"{a.workers} workers")

    t0, n_files = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(scan_shard, i, shards[i], str(split_dir), str(out_dir)) for i in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            n_files += r["n_files"]
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] shard {r['shard']}: {r['n_files']} files | total {n_files} "
                  f"| {n_files / max(el, 1e-9):.0f} files/s | {el / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
