"""S02 preprocessing + cache + datasets. CPU-only, synthetic DICOMs (plus the real test stub if KNEE_STUB_DIR is set).

Brief checklist: slice order monotonic in physical position; laterality rule vs the tag; parity (a study through the
direct/notebook path == the cache bytes); both datasets load a batch; fixed-mm crop keeps anatomy the same size.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
torch = pytest.importorskip("torch")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid  # noqa: E402

from knee.data import build_cache, crop_resize as CR, geometry as G  # noqa: E402
from knee.data.dataset import (CacheReader, StudyBagDataset, StudyWindowsDataset, collate_bag,  # noqa: E402
                               eval_centres, train_centres, window_indices)
from knee.data.preprocess import PreprocParams, preprocess_study  # noqa: E402
from knee.data.series_select import assign_slots  # noqa: E402
from knee.paths import LABELS  # noqa: E402

PX = 32  # tiny output grid for fast tests
PARAMS = PreprocParams(crop_mm=24.0, out_px=PX)  # 24 mm < the 32 mm synthetic FOV, so the crop is exercised


def write_slice(path: Path, study: str, series: str, ipp, iop, img: np.ndarray, spacing=(0.5, 0.5), inst=1,
                laterality=None, desc="pd_tse_fs", tr=3000, te=30):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, meta.MediaStorageSOPInstanceUID, "MR"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.InstanceNumber = study, series, inst
    ds.ImageOrientationPatient = [float(v) for v in iop]
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.PixelSpacing, ds.SliceThickness = list(spacing), 3
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.Manufacturer, ds.MagneticFieldStrength = "SIEMENS", 3
    if laterality:
        ds.Laterality = laterality
    ds.RepetitionTime, ds.EchoTime, ds.ScanningSequence, ds.SeriesDescription = tr, te, "SE", desc
    ds.PixelData = img.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def make_series(root: Path, study: str, series: str, plane: str, n: int = 6, x0: float = -90.0, laterality=None,
                shuffle=True, desc="pd_tse_fs", size=64) -> None:
    """A series in canonical orientation whose slice k has value 100*(k+1) plus a marker in the top-left corner."""
    u, v = G.CANONICAL_UV[plane]
    nrm = np.cross(u, v)
    d = root / study / series
    d.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(0).permutation(n) if shuffle else np.arange(n)
    centre = np.array([x0, 0.0, 0.0])
    for fi, k in enumerate(order):
        img = np.full((size, size), 100 * (k + 1), dtype=np.uint16)
        img[:20, :20] = 4000  # marker: top-left in canonical orientation (survives the centre crop)
        ipp = centre - u * (size / 2) * 0.5 - v * (size / 2) * 0.5 + nrm * 3.0 * k
        write_slice(d / f"{fi}.dcm", study, series, ipp, np.concatenate([u, v]), img, inst=n - k,
                    laterality=laterality, desc=desc)


# ---------------------------------------------------------------------------------------------------- geometry
@pytest.mark.parametrize("plane", ["Axial", "Coronal", "Sagittal"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("flip_r", [False, True])
@pytest.mark.parametrize("flip_c", [False, True])
def test_canonical_inplane_ops_undo_any_dihedral(plane, transpose, flip_r, flip_c):
    """Store a canonical image in any of the 8 transposed/flipped layouts (with a matching IOP); the canonical
    ops must recover the original exactly."""
    u, v = G.CANONICAL_UV[plane]
    canon = np.arange(12 * 12).reshape(12, 12)
    img, r, c = canon, u.copy(), v.copy()
    if transpose:
        img, r, c = img.T, c, r
    if flip_r:
        img, c = img[::-1], -c
    if flip_c:
        img, r = img[:, ::-1], -r
    r = r + 0.15 * np.cross(r, c)  # a ~9° oblique tilt must not change the answer
    r /= np.linalg.norm(r)
    ops = G.canonical_inplane_ops(np.concatenate([r, c]), plane)
    np.testing.assert_array_equal(G.apply_inplane_ops(img, ops), canon)


def test_resolve_laterality_tag_then_geometry():
    assert G.resolve_study_laterality(["L", "L", None], [-80.0], 20) == ("L", "tag")
    assert G.resolve_study_laterality(["L", "R"], [-80.0, -85.0], 20) == ("R", "geometry")  # conflicting tags
    assert G.resolve_study_laterality([None], [75.0, 80.0], 20) == ("L", "geometry")
    assert G.resolve_study_laterality([None], [5.0], 20) == ("U", "none")  # inside the dead zone → no flip


# ---------------------------------------------------------------------------------------------------- crop/resize
def test_fixed_mm_crop_keeps_physical_size():
    """A 10 mm bright square imaged at 0.2 mm/px and at 0.4 mm/px must come out the same size in pixels."""
    widths = []
    for sp in (0.2, 0.4):
        n = int(round(200 / sp))  # 200 mm FOV
        img = np.zeros((n, n), np.float32)
        half = int(round(5 / sp))
        img[n // 2 - half: n // 2 + half, n // 2 - half: n // 2 + half] = 1
        crop = CR.crop_mm_native(img, np.array([sp, sp]), 100.0)
        out = CR.pad_center(CR.resize_crop(crop, np.array([sp, sp]), 100.0, 100), 100)
        assert out.shape == (100, 100)
        widths.append(int((out[50] > 0.5).sum()))
    assert widths[0] == widths[1] == 10  # 10 mm at 1 mm/px


def test_small_fov_is_padded_not_stretched():
    img = np.ones((50, 50), np.float32)  # 50 px × 0.5 mm = 25 mm FOV, smaller than the 40 mm crop
    out = CR.pad_center(CR.resize_crop(CR.crop_mm_native(img, np.array([0.5, 0.5]), 40.0),
                                       np.array([0.5, 0.5]), 40.0, 40), 40)
    assert out.shape == (40, 40) and out.sum() == pytest.approx(25 * 25, abs=30)


# ---------------------------------------------------------------------------------------------------- slots/windows
def test_assign_slots_prefers_fluid_pd_and_more_slices():
    s = [dict(series_uid="a", plane="Sagittal", fluid=True, contrast_class="STIR", n_slices=30, series_number=1),
         dict(series_uid="b", plane="Sagittal", fluid=True, contrast_class="PD", n_slices=20, series_number=2),
         dict(series_uid="c", plane="Sagittal", fluid=False, contrast_class="T1", n_slices=20, series_number=3),
         dict(series_uid="d", plane="Axial", fluid=False, contrast_class="PD", n_slices=40, series_number=4),
         dict(series_uid="e", plane="Axial", fluid=True, contrast_class="T2", n_slices=20, series_number=5)]
    assert assign_slots(s) == {"sag_fluid": "b", "sag_other": "c", "axial": "e"}


def test_window_helpers():
    rng = np.random.default_rng(0)
    for n in (1, 2, 12, 30, 160):
        c = train_centres(n, 8, rng)
        assert len(c) == 8 and c.min() >= 0 and c.max() <= n - 1 and np.all(np.diff(c) >= 0)
        e = eval_centres(n, 8)
        assert e.min() >= 0 and e.max() <= n - 1
        idx = window_indices(n, e)
        assert idx.shape == (len(e), 3) and idx.min() >= 0 and idx.max() <= n - 1
    assert eval_centres(30, None).tolist() == list(range(0, 30))  # 2–98% of 30 slices spans all of them
    np.testing.assert_array_equal(window_indices(5, np.array([0, 2, 4])), [[0, 0, 1], [1, 2, 3], [3, 4, 4]])


# ---------------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Two studies: a right knee (geometry only) and a tagged left knee; 3 planes each; shuffled filenames."""
    root = tmp_path_factory.mktemp("data")
    split = root / "train_series"
    rows = []
    for study, x0, lat in (("1.1", -90.0, None), ("1.2", 90.0, "L")):
        for j, (plane, fluid) in enumerate((("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1))):
            uid = f"{study}.{j}"
            make_series(split, study, uid, plane, n=6 + j, x0=x0, laterality=lat,
                        desc="pd_tse_fs" if fluid else "pd_tse")
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": uid, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    pd.DataFrame(rows).to_csv(root / "train_series.csv", index=False)
    return root


def _info(root):
    from knee.data.preprocess import series_info_from_csv
    return series_info_from_csv(pd.read_csv(root / "train_series.csv"))


def test_preprocess_orders_slices_and_canonicalizes(synth):
    right, sm_r = preprocess_study(synth / "train_series" / "1.1", _info(synth), PARAMS)
    left, sm_l = preprocess_study(synth / "train_series" / "1.2", _info(synth), PARAMS)
    assert (sm_r["laterality"], sm_r["laterality_source"], sm_r["mirrored"]) == ("R", "geometry", False)
    assert (sm_l["laterality"], sm_l["laterality_source"], sm_l["mirrored"]) == ("L", "tag", True)
    assert sm_r["slots"] == {"sag_fluid": "1.1.0", "sag_other": "1.1.1", "cor_fluid": "1.1.2", "axial": "1.1.3"}
    for s in right + left:
        assert np.all(np.diff(s.positions_mm) > 0), "slice positions must ascend"
        assert s.pixels.shape[1:] == (PX, PX) and s.pixels.dtype == np.uint8 and s.valid.all()
    # right knee: slice brightness rises with the canonical position (value 100*(k+1) was written at step k),
    # and the marker is in the top-left corner
    for s in right:
        means = s.pixels[:, PX // 2, PX // 2].astype(int)
        assert np.all(np.diff(means) > 0)
        assert s.pixels[0, 1, 1] == 255 and s.pixels[0, 1, -2] < 255
    # left knee: cor/ax mirrored left-right (marker moves to the top-right), sagittal slice order reversed
    by_plane = {s.meta["plane"]: s for s in left if s.meta["slot"] != "sag_other"}
    for plane in ("Coronal", "Axial"):
        p = by_plane[plane].pixels
        assert p[0, 1, -2] == 255 and p[0, 1, 1] < 255
        assert np.all(np.diff(p[:, PX // 2, PX // 2].astype(int)) > 0)
    sag = by_plane["Sagittal"].pixels
    assert sag[0, 1, 1] == 255 and np.all(np.diff(sag[:, PX // 2, PX // 2].astype(int)) < 0)


def test_parity_cache_equals_direct_and_datasets(synth, tmp_path):
    """Build the cache in a worker, then compare its bytes with a direct preprocess_study call (the notebook path)."""
    out = tmp_path / "cache"
    (out / "shards").mkdir(parents=True)
    info = _info(synth)
    build_cache.build_shard(0, ["1.1", "1.2"], str(synth / "train_series"), str(out / "shards"), PARAMS.to_dict(),
                            info)
    (out / "cache_manifest.json").write_text(json.dumps({"params": PARAMS.to_dict()}))
    build_cache.merge_index(out)
    r = CacheReader(out)
    for study in ("1.1", "1.2"):
        direct, _ = preprocess_study(synth / "train_series" / study, info, PARAMS)
        cached = {row["series_uid"]: row for row in r.series(study)}
        for s in direct:
            np.testing.assert_array_equal(np.asarray(r.pixels(cached[s.series_uid])), s.pixels)
            np.testing.assert_allclose(cached[s.series_uid]["positions_mm"], s.positions_mm)

    r2 = pickle.loads(pickle.dumps(r))  # spawn-safety: picklable, memmaps reopened lazily
    labels = pd.DataFrame(np.full((1, 12), 1.0), index=["1.1"], columns=LABELS)
    ds_a = StudyWindowsDataset(r2, ["1.1", "1.2"], labels, windows_per_slot=4, train=True, out_px=16)
    b = ds_a[0]
    assert b["x"].shape == (5, 4, 3, 16, 16) and b["x"].dtype == torch.uint8
    assert b["slot_mask"].tolist() == [True, True, True, False, True] and b["label_mask"].all()
    assert not ds_a[1]["label_mask"].any()
    ds_b = StudyBagDataset(r2, ["1.1", "1.2"], labels, windows_per_series=3, train=True)
    batch = collate_bag([ds_b[0], ds_b[1]])
    assert batch["x"].shape == (2, 12, 3, PX, PX) and batch["window_mask"].all()
    ds_e = StudyBagDataset(r2, ["1.1"], eval_windows_per_series=None)
    assert len(ds_e[0]["x"]) == 6 + 7 + 8 + 9  # every slice of every series
    assert set(ds_e[0]["plane"].tolist()) == {0, 1, 2}


    # model-input parity: the inference path (preprocess_study → arm_input) builds exactly the eval batch the
    # training datasets build from the cache, for both input formats
    from knee.infer.predict import arm_input
    bag_cfg = {"format": "bag", "train_windows": 3, "eval_windows": 4, "out_px": 16}
    slot_cfg = {"format": "slots", "train_windows": 3, "eval_windows": 4, "out_px": 16}
    for study in ("1.1", "1.2"):
        direct, _ = preprocess_study(synth / "train_series" / study, info, PARAMS)
        got = arm_input(direct, bag_cfg)
        ref = collate_bag([StudyBagDataset(r2, [study], eval_windows_per_series=4, out_px=16)[0]])
        for k in ("x", "window_mask", "z", "z_mm", "plane", "fluid", "contrast", "field", "slot", "series_idx"):
            torch.testing.assert_close(got[k], ref[k], msg=k)
        got = arm_input(direct, slot_cfg)
        ref = StudyWindowsDataset(r2, [study], windows_per_slot=4, out_px=16)[0]
        for k in ("x", "slot_mask", "z"):
            torch.testing.assert_close(got[k][0], ref[k], msg=k)


def test_predict_dirs_and_submission(synth, tmp_path):
    """End to end on synthetic DICOMs: a tiny KneeViT checkpoint → predictions → submission.csv with asserts;
    a broken study folder gets the fallback row instead of crashing."""
    from knee.infer.predict import SUBMISSION_COLUMNS, load_arm, predict_dirs, write_submission
    from knee.model.knee_vit import KneeViT
    cfg = {"model": {"kind": "knee_vit", "cfg": {"slice_vit": {"img_size": 16, "patch": 8, "dim": 16, "depth": 1,
                                                                "heads": 2}}},
           "data": {"format": "bag", "train_windows": 2, "eval_windows": 3, "out_px": 16}}
    m = KneeViT(cfg["model"]["cfg"])
    torch.save({"model": m.state_dict(), "cfg": cfg}, tmp_path / "model_avg.pt")
    arm = load_arm(tmp_path / "model_avg.pt", torch.device("cpu"))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "junk").mkdir()
    (tmp_path / "broken" / "junk" / "x.dcm").write_bytes(b"not a dicom")
    dirs = [synth / "train_series" / "1.1", synth / "train_series" / "1.2", tmp_path / "broken"]
    prior = np.linspace(0.1, 0.9, 12)
    pred, timing = predict_dirs(dirs, _info(synth), [arm, arm], torch.device("cpu"), PARAMS, workers=0,
                                fallback=prior)
    assert pred.StudyInstanceUID.tolist() == ["1.1", "1.2", "broken"]
    assert timing.error.iloc[:2].isna().all() and timing.error.iloc[2]
    np.testing.assert_allclose(pred.iloc[2, 1:].to_numpy(dtype=float), prior)
    np.testing.assert_allclose(pred.iloc[0, 1:].to_numpy(dtype=float), 0.5)  # zero-init head
    sub = write_submission(pred, ["1.2", "1.1", "broken", "missing"], tmp_path / "submission.csv", prior)
    assert list(sub.columns) == SUBMISSION_COLUMNS and sub.StudyInstanceUID.tolist()[0] == "1.2"
    np.testing.assert_allclose(sub.iloc[3, 1:].to_numpy(dtype=float), prior)


@pytest.mark.skipif(not os.environ.get("KNEE_STUB_DIR"), reason="set KNEE_STUB_DIR to the Kaggle test stub dir")
def test_real_stub_slice_order_and_laterality():
    """On real DICOMs: positions ascend, spacing matches the header-based audit, the tagged study is L."""
    from knee.data.preprocess import series_info_from_csv
    root = Path(os.environ["KNEE_STUB_DIR"])
    info = series_info_from_csv(pd.read_csv(root / "test_series.csv"))
    for st in sorted(p.name for p in (root / "test_series").iterdir() if p.is_dir()):
        series, sm = preprocess_study(root / "test_series" / st, info, PreprocParams(out_px=64))
        for s in series:
            assert np.all(np.diff(s.positions_mm) > 0)
            assert s.meta["plane"] == s.meta["plane_csv"]
        tags = {s.meta["laterality_tag"] for s in series} - {None}
        if tags:
            assert sm["laterality"] in tags and sm["laterality_source"] == "tag"


def test_predict_dirs_grouped_equals_single(synth, tmp_path, monkeypatch):
    """Several studies per forward and a bigger inference window_chunk give the same predictions as one study at a
    time; a failing grouped forward falls back to one-by-one, not to the fallback row."""
    from knee.infer import predict as P
    from knee.model.knee_vit import KneeViT
    cfg = {"model": {"kind": "knee_vit", "cfg": {"slice_vit": {"img_size": 16, "patch": 8, "dim": 16, "depth": 1,
                                                                "heads": 2}, "window_chunk": 2}},
           "data": {"format": "bag", "train_windows": 2, "eval_windows": 3, "out_px": 16}}
    torch.manual_seed(0)
    m = KneeViT(cfg["model"]["cfg"])
    for p in m.head.parameters():  # the logit layers are zero-init: make the predictions differ per study
        torch.nn.init.normal_(p, std=0.5)
    torch.save({"model": m.state_dict(), "cfg": cfg}, tmp_path / "model_avg.pt")
    dirs = [synth / "train_series" / "1.1", synth / "train_series" / "1.2", synth / "train_series" / "1.1"]
    dev = torch.device("cpu")
    one = P.predict_dirs(dirs, _info(synth), [P.load_arm(tmp_path / "model_avg.pt", dev)], dev, PARAMS, workers=0)[0]
    assert not np.allclose(one.iloc[0, 1:].to_numpy(dtype=float), one.iloc[1, 1:].to_numpy(dtype=float))
    big = P.load_arm(tmp_path / "model_avg.pt", dev, {"window_chunk": 64})
    assert big[0].window_chunk == 64
    grouped, timing = P.predict_dirs(dirs, _info(synth), [big], dev, PARAMS, workers=0, studies_per_forward=2)
    assert grouped.StudyInstanceUID.tolist() == one.StudyInstanceUID.tolist() and timing.error.isna().all()
    np.testing.assert_allclose(grouped.iloc[:, 1:].to_numpy(dtype=float), one.iloc[:, 1:].to_numpy(dtype=float),
                               rtol=1e-5, atol=1e-6)

    real = P.run_arm

    def no_groups(model, batch, device):
        if len(batch["study_uid"]) > 1:
            raise RuntimeError("simulated OOM")
        return real(model, batch, device)
    monkeypatch.setattr(P, "run_arm", no_groups)
    retried, timing = P.predict_dirs(dirs, _info(synth), [big], dev, PARAMS, workers=0, studies_per_forward=3)
    assert timing.error.isna().all()
    np.testing.assert_allclose(retried.iloc[:, 1:].to_numpy(dtype=float), one.iloc[:, 1:].to_numpy(dtype=float),
                               rtol=1e-5, atol=1e-6)


def test_build_cache_refuses_mismatched_resume(tmp_path):
    """Shards count as done when their parquet exists, but the shard → studies mapping comes from the current args:
    a --limit-studies smoke run into the real --out, or a changed --studies-per-shard, would drop or duplicate studies
    on resume (review 2026-09-30)."""
    studies = [f"s{i:03d}" for i in range(48)]
    p = PARAMS.to_dict()
    old = {"params": p, "studies_per_shard": 16, "n_studies": 40, "studies_sha1": build_cache.studies_sha1(studies[:40])}
    mf = tmp_path / "cache_manifest.json"
    build_cache.check_resume(old, p, studies[:40], 16, mf)  # same run: fine
    with pytest.raises(SystemExit, match="study list"):
        build_cache.check_resume(old, p, studies, 16, mf)
    with pytest.raises(SystemExit, match="studies_per_shard"):
        build_cache.check_resume(old, p, studies[:40], 8, mf)
    with pytest.raises(SystemExit, match="params"):
        build_cache.check_resume(old, {**p, "out_px": 64}, studies[:40], 16, mf)
    legacy = {"params": p, "studies_per_shard": 16, "n_studies": 40}  # manifest written before the hash existed
    with pytest.raises(SystemExit, match="n_studies"):
        build_cache.check_resume(legacy, p, studies, 16, mf)
