"""S06: the Kaggle script kernel end to end on synthetic DICOMs: stage a bundle (code + tiny checkpoint + arms.json),
run notebooks/infer.py against a fake competition dir, check submission.csv, the probe and the timing summary."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

from test_data import make_series  # noqa: E402

from knee.paths import LABELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_infer_script_end_to_end(tmp_path):
    comp = tmp_path / "comp"
    rows = []
    for study, x0 in (("9.1", -90.0), ("9.2", 90.0)):
        for j, (plane, fluid) in enumerate((("Sagittal", 1), ("Coronal", 1), ("Axial", 0))):
            uid = f"{study}.{j}"
            make_series(comp / "test_series", study, uid, plane, n=5, x0=x0, desc="pd_tse_fs" if fluid else "pd_tse")
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": uid, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    pd.DataFrame(rows).to_csv(comp / "test_series.csv", index=False)
    # 9.3 is listed in test.csv but has no folder → must still get a (fallback) row
    pd.DataFrame({"StudyInstanceUID": ["9.2", "9.1", "9.3"]}).to_csv(comp / "test.csv", index=False)

    from knee.model.knee_vit import KneeViT
    cfg = {"model": {"kind": "knee_vit", "cfg": {"slice_vit": {"img_size": 16, "patch": 8, "dim": 16, "depth": 1,
                                                                "heads": 2}}},
           "data": {"format": "bag", "train_windows": 2, "eval_windows": 3, "out_px": 16}}
    torch.save({"model": KneeViT(cfg["model"]["cfg"]).state_dict(), "cfg": cfg}, tmp_path / "model_avg.pt")
    labels = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{k: [0.2, 0.4] for k in LABELS}})
    labels.to_parquet(tmp_path / "labels.parquet")
    manifest = tmp_path / "cache_manifest.json"
    from knee.data.preprocess import PreprocParams
    manifest.write_text(json.dumps({"params": PreprocParams(out_px=64).to_dict()}))

    bb = _load("build_kaggle_bundle", REPO / "scripts" / "build_kaggle_bundle.py")
    bundle = tmp_path / "bundle"
    spec = bb.build(bundle, [tmp_path / "model_avg.pt"] * 2, [1.0, 2.0], "mean", tmp_path / "labels.parquet", None,
                    "someone", "knee-bundle", json.loads(manifest.read_text())["params"])
    assert (bundle / "knee_src" / "knee" / "infer" / "predict.py").exists()
    assert not list((bundle / "knee_src").rglob("__pycache__"))
    assert spec["prior"] == pytest.approx([0.3] * 12)

    nb = _load("infer_nb", REPO / "notebooks" / "infer.py")
    out = tmp_path / "out"
    try:
        rc = nb.main(["--comp-dir", str(comp), "--bundle-dir", str(bundle), "--out", str(out), "--workers", "0"])
    finally:
        sys.path[:] = [p for p in sys.path if "knee_src" not in p]
    assert rc == 0
    sub = pd.read_csv(out / "submission.csv")
    assert sub.columns.tolist() == ["StudyInstanceUID", *LABELS]
    assert sub.StudyInstanceUID.astype(str).tolist() == ["9.2", "9.1", "9.3"]
    np.testing.assert_allclose(sub.iloc[0, 1:].to_numpy(float), 0.5)   # zero-init heads
    np.testing.assert_allclose(sub.iloc[2, 1:].to_numpy(float), 0.3)   # missing folder → prior
    summ = json.loads((out / "run_summary.json").read_text())
    assert summ["n_studies"] == 3 and summ["n_errors"] == 1
    assert summ["probe"]["fluid_eq_fatsup"] == 1.0
    assert sum(summ["probe"]["transfer_syntaxes"].values()) == 6
    assert set(summ["stages_s"]) >= {"setup", "probe", "load_arms", "predict", "write", "total"}


def test_infer_refuses_mismatched_preprocessing(tmp_path):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "arms.json").write_text(json.dumps({"arms": [], "preproc": {"version": "v0-old"}}))
    comp = tmp_path / "c"
    comp.mkdir()
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(comp / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], "SeriesInstanceUID": []}).to_csv(comp / "test_series.csv", index=False)
    nb = _load("infer_nb2", REPO / "notebooks" / "infer.py")
    with pytest.raises(RuntimeError, match="preprocessing"):
        nb.main(["--comp-dir", str(comp), "--bundle-dir", str(bundle), "--out", str(tmp_path / "o"), "--probe", "0"])


def test_bundle_restores_source_out_px_for_derived_cache():
    """A derived 256-px cache's manifest says out_px=256, but its windows are 384-px renders resized with INTER_AREA.
    The bundle must render at 384 so the arm's data.out_px resize reproduces them (review 2026-09-30)."""
    bb = _load("build_kaggle_bundle_px", REPO / "scripts" / "build_kaggle_bundle.py")
    params = {"version": "v1", "out_px": 256, "crop_mm": 160.0}
    derived = {"params": params, "derived_from": {"src_out_px": 384}}
    assert bb.preproc_from_manifest(derived) == {**params, "out_px": 384}
    assert bb.preproc_from_manifest({"params": params}) == params
