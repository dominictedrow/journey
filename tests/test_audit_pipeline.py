"""End-to-end smoke test of header_scan → series_summary → decode variants on a tiny synthetic series."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid  # noqa: E402

from knee.audit import header_scan, series_summary  # noqa: E402

STUDY, SERIES = "1.2.3", "1.2.3.4"


def _write_slice(path: Path, z: float, inst: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, meta.MediaStorageSOPInstanceUID, "MR"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.InstanceNumber = STUDY, SERIES, inst
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-100.0, -50.0, z]  # left edge at x=-100 → centre x ≈ -84 → right knee
    ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 3
    ds.Rows = ds.Columns = 64
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.Manufacturer, ds.MagneticFieldStrength, ds.Laterality = "SIEMENS", 3, "R"
    ds.RepetitionTime, ds.EchoTime, ds.ScanningSequence, ds.SeriesDescription = 3000, 30, "SE", "pd_tse_fs_tra"
    ds.PixelData = np.full((64, 64), inst, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_pipeline(tmp_path: Path):
    data = tmp_path / "data"
    sd = data / "train_series" / STUDY / SERIES
    sd.mkdir(parents=True)
    for i, z in enumerate([6.0, 0.0, 3.0, 9.0]):  # shuffled positions; InstanceNumber runs against position
        _write_slice(sd / f"{i}.dcm", z, inst=10 - int(z / 3))
    # A truncated copy and a non-DICOM file.
    raw = (sd / "0.dcm").read_bytes()
    (sd / "trunc.dcm").write_bytes(raw[:-100])
    (sd / "notes.txt").write_text("x")
    pd.DataFrame([{"StudyInstanceUID": STUDY, "SeriesInstanceUID": SERIES, "Fluid_Sensitive": 1,
                   "Fat_Suppression": 1, "Anatomical_Plane": "Axial"}]).to_csv(data / "train_series.csv", index=False)

    out = tmp_path / "audit" / "header_scan_train"
    out.mkdir(parents=True)
    header_scan.scan_shard(0, [f"{STUDY}/{SERIES}"], str(data / "train_series"), str(out))
    df = pd.read_parquet(out / "part-00000.parquet")
    assert len(df) == 6
    assert df.set_index(df.path.str.rsplit("/", n=1).str[-1]).loc["trunc.dcm", "truncated"]
    assert df.set_index(df.path.str.rsplit("/", n=1).str[-1]).loc["notes.txt", "error"] is not None
    info = json.loads((out / "part-00000.json").read_text(encoding="utf-8"))
    assert any(f.endswith("notes.txt") for f in info["other_files"])

    # Summary ignores the bad files for geometry.
    s = series_summary.summarize_series(df)
    assert s["n_files"] == 6 and s["n_errors"] == 1 and s["n_truncated"] == 1
    assert s["plane_from_iop"] == "Axial" and s["sp_median_mm"] == pytest.approx(3.0)
    assert s["sp_n_duplicate"] == 1  # the truncated copy repeats a position
    assert s["laterality_geom"] == "R" and s["laterality_tag"] == "R"
    assert s["contrast_class"] == "PD" and s["fat_sat_header"]
    assert s["instance_vs_position"] < 0.5
    shutil.rmtree(tmp_path)
