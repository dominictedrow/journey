"""Regression tests for the open items of bug.md (review 2026-09-30, second batch): #6–#10 and the low-severity list.
CPU-only, tiny models, fake caches and synthetic DICOMs."""
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

from fake_cache import make_fake_cache  # noqa: E402
from test_data import PARAMS, _info, make_series, synth  # noqa: E402,F401  (synth is a fixture)

from knee.data import dataset as D  # noqa: E402
from knee.data.preprocess import preprocess_study  # noqa: E402
from knee.paths import LABELS  # noqa: E402
from knee.train import common as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TINY_VIT = {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2}


def _write(path, d):
    path.write_text(yaml.safe_dump(d))
    return path


def _sup_cfg(tmp_path, f, fold=-1, **optim):
    return {"cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu", "precision": "fp32",
            "model": {"kind": "knee_vit", "cfg": {"slice_vit": TINY_VIT}}, "labels": {"file": str(f["labels"])},
            "folds": {"file": str(f["folds"]), "fold": fold},
            "data": {"out_px": 32, "train_windows": 2, "eval_windows": 3, "batch_size": 1, "workers": 0},
            "optim": {"epochs": 2, "accum": 1, **optim}, "select": {"top_k": 2}, "log_every": 1000}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ #10: persistent loaders
def test_sampler_encodes_epoch_and_stops_early():
    s = C.ResumableRandomSampler(10, seed=1, encode_epoch=True)
    s.set_epoch(3, start=2, num_samples=8)
    idx = list(s)
    assert len(idx) == len(s) == 6 and all(30 <= i < 40 for i in idx)
    plain = C.ResumableRandomSampler(10, seed=1)
    plain.set_epoch(3)
    assert [i - 30 for i in idx] == list(plain)[2:8]  # same order, just offset
    assert D.decode_index(37, 10, epoch=0) == (3, 7) and D.decode_index(7, 10, epoch=5) == (5, 7)


def test_dataset_reads_epoch_from_index(tmp_path):
    """A persistent worker's dataset copy never sees set_epoch: the epoch-encoded index alone must give the same
    sample as set_epoch(epoch) + the plain index."""
    f = make_fake_cache(tmp_path / "c", n_studies=4, px=32)
    r = D.CacheReader(f["cache"])
    ds = D.StudyBagDataset(r, r.study_uids, windows_per_series=2, train=True, series_dropout=0.3, seed=3)
    ref = D.StudyBagDataset(r, r.study_uids, windows_per_series=2, train=True, series_dropout=0.3, seed=3)
    ref.set_epoch(2)
    a, b = ds[2 * 4 + 1], ref[1]
    assert a["study_uid"] == b["study_uid"]
    for k in ("x", "z", "series_idx"):
        torch.testing.assert_close(a[k], b[k])
    assert not torch.equal(ds[1 * 4 + 1]["z"], a["z"])  # another epoch draws other windows


def test_train_sup_computes_no_discarded_micro_batches(tmp_path, monkeypatch):
    """7 studies at accum 3 = 2 steps per epoch: the 7th micro-batch used to be forward/backward-ed and then zeroed.
    Now each epoch stops at its last full step: 6 backward passes per epoch."""
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=7, px=48)
    n_backward = [0]
    build = train_sup.build_model

    def build_with_hook(mcfg):
        m = build(mcfg)
        m.head.cls_b.register_hook(lambda g: n_backward.__setitem__(0, n_backward[0] + 1))
        return m
    monkeypatch.setattr(train_sup, "build_model", build_with_hook)
    train_sup.main(["--config", str(_write(tmp_path / "s.yaml", _sup_cfg(tmp_path, f, accum=3)))])
    assert n_backward[0] == 2 * 6


# ------------------------------------------------------------------------------------------ train_sup low items
class _OomStub(torch.nn.Module):
    """Raises (an OOM stand-in) on any multi-study forward, in any precision; fine one study at a time. The old
    retry re-ran the whole batch in fp32 and would hit the same OOM."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(4, len(LABELS))

    def forward(self, b):
        if len(b["study_uid"]) > 1:
            raise RuntimeError("CUDA out of memory (stub)")
        return self.w(b["x"].float())


def test_predict_retries_a_failed_batch_one_study_at_a_time():
    from knee.train import train_sup
    rng = np.random.default_rng(0)
    items = [{"x": torch.from_numpy(rng.normal(size=4).astype(np.float32)), "study_uid": f"s{i}"} for i in range(5)]

    def collate(b):
        return {"x": torch.stack([i["x"] for i in b]), "study_uid": [i["study_uid"] for i in b]}
    model = _OomStub()
    uids, p, st = train_sup.predict(model, items, collate, torch.device("cpu"), "bf16_cpu", 0, batch_size=2)
    assert uids == [f"s{i}" for i in range(5)] and np.isfinite(p).all()
    assert st == {"n_retried_fp32": 2, "n_nonfinite": 0}  # the 2 two-study batches; the last batch has 1 study
    _, p32, _ = train_sup.predict(model, items, collate, torch.device("cpu"), "fp32", 0, batch_size=1)
    np.testing.assert_allclose(p[:4], p32[:4], rtol=0, atol=1e-6)


def test_check_run_identity_accepts_a_remounted_folds_file(tmp_path, capsys):
    from knee.train import train_sup
    run = tmp_path / "run"
    run.mkdir()
    a = tmp_path / "in_a" / "folds.csv"
    a.parent.mkdir()
    a.write_text("StudyInstanceUID,fold\nx,0\n")
    rec = {"run_name": "r", "folds": {"fold": 0, "file": str(a)}}
    (run / "config.yaml").write_text(yaml.safe_dump(rec))
    b = tmp_path / "in_b" / "folds.csv"  # the same bytes under another mount
    b.parent.mkdir()
    shutil.copy(a, b)
    train_sup.check_run_identity(run, {**rec, "folds": {"fold": 0, "file": str(b)}})
    shutil.rmtree(a.parent)  # the recorded path is gone (another /kaggle/input/<slug>): accepted by name, warned
    train_sup.check_run_identity(run, {**rec, "folds": {"fold": 0, "file": str(b)}})
    assert "no longer exists" in capsys.readouterr().out
    a.parent.mkdir()
    a.write_text("StudyInstanceUID,fold\nx,1\n")  # a different split at the recorded path
    with pytest.raises(SystemExit, match="folds.file"):
        train_sup.check_run_identity(run, {**rec, "folds": {"fold": 0, "file": str(b)}})


def test_stop_during_eval_is_honoured_and_missing_epoch_files_are_skipped(tmp_path, monkeypatch):
    """A STOP that lands while the epoch-end eval runs stops right after that epoch's checkpoint, not an epoch later;
    the resumed run finishes, and an epoch file lost before the final average is skipped instead of crashing."""
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=8, px=48)
    cp = _write(tmp_path / "s.yaml", _sup_cfg(tmp_path, f, fold=0, epochs=3))
    real = train_sup.evaluate
    calls = [0]

    def evaluate_then_stop(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            (tmp_path / "run" / "STOP").touch()
        return real(*a, **k)
    monkeypatch.setattr(train_sup, "evaluate", evaluate_then_stop)
    assert train_sup.main(["--config", str(cp)]) == {"stopped": True}
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["state"]["epoch"] == 1 and ck["state"]["batch_in_epoch"] == 0
    assert ck["rank_bank"]["logits"] is not None  # the rank bank is checkpointed too
    (tmp_path / "run" / "epochs" / "ep000.pt").unlink()
    res = train_sup.main(["--config", str(cp)])
    assert 0 not in res["top_epochs"] and len(res["top_epochs"]) >= 1
    assert np.isfinite(res["val_macro_auc"])


def test_final_falls_back_to_the_best_epoch_when_the_average_is_worse(tmp_path, monkeypatch):
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=8, px=48)
    real = train_sup.evaluate
    n_final = [0]

    def evaluate(*a, n_boot=1000, **k):
        res = real(*a, n_boot=n_boot, **k)
        if n_boot == 1000 and n_final[0] == 0:  # the first final eval = the averaged model: make it look worse
            n_final[0] += 1
            res["val_macro_auc"] = -1.0
        return res
    monkeypatch.setattr(train_sup, "evaluate", evaluate)
    res = train_sup.main(["--config", str(_write(tmp_path / "s.yaml", _sup_cfg(tmp_path, f, fold=0, epochs=3)))])
    best = res["best_epoch"]
    assert res["used_epochs"] == [best] and res["avg_val_macro_auc"] == -1.0
    assert res["val_macro_auc"] == pytest.approx(res["best_epoch_val_macro_auc"])
    avg = torch.load(tmp_path / "run" / "model_avg.pt", weights_only=False)
    ep = torch.load(tmp_path / "run" / "epochs" / f"ep{best:03d}.pt", weights_only=False)["model"]
    assert avg["epochs"] == [best] and all(torch.equal(avg["model"][k], ep[k]) for k in ep)
    oof = pd.read_parquet(tmp_path / "run" / "oof.parquet")
    oof_best = pd.read_parquet(tmp_path / "run" / "epochs" / f"oof_ep{best:03d}.parquet")
    pd.testing.assert_frame_equal(oof[LABELS], oof_best[LABELS])


def test_rank_loss_thresholds_match_binarize_and_bank_state_roundtrips():
    from knee.train.losses import RankBank, pairwise_rank_loss
    from knee.utils.metrics import binarize
    assert binarize(np.array([[0.6], [0.4]])).tolist() == [[1.0], [0.0]]  # labels are float64 from parquet
    y = torch.tensor([[0.6], [0.4]])
    s = torch.tensor([[0.0], [0.0]], requires_grad=True)
    assert pairwise_rank_loss(s, y).item() == pytest.approx(np.log(2))  # 0.6 vs 0.4 is a pair, like the metric
    bank = RankBank(4)
    bank.push(torch.randn(3, 12), torch.rand(3, 12), torch.ones(3, 12, dtype=torch.bool))
    other = RankBank(1)
    other.load_state_dict(bank.state_dict())
    assert other.size == 4 and all(torch.equal(x, z) for x, z in zip(other.get(), bank.get()))


def test_slice_pos_projection_has_no_weight_decay():
    from knee.model.knee_vit import KneeViT
    from knee.train.optim import param_groups
    m = KneeViT({"slice_vit": TINY_VIT})
    groups = param_groups(m, 1e-3, 0.05)
    wd_of = {n: g["weight_decay"] for g in groups for n in g["names"]}
    assert wd_of["slice_pos.proj.weight"] == 0.0
    assert wd_of["head.cls_w"] == 0.05  # the classifier weight stays decayed (standard)


# ------------------------------------------------------------------------------------------ #6: Track A frozen BN
def test_track_a_batchnorm_frozen_in_training():
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    torch.manual_seed(0)
    m = TrackAModel({"backbone": "resnet10t", "pretrained": False, "grad_checkpointing": False, "window_chunk": 4})
    bns = m.bn_modules()
    assert bns and all(not p.requires_grad for b in bns for p in b.parameters(recurse=False))
    before = [b.running_mean.clone() for b in bns]
    m.train()
    assert all(not b.training for b in bns) and m.head.training
    batch = {"x": torch.randint(0, 255, (1, 6, 3, 32, 32), dtype=torch.uint8),
             "window_mask": torch.ones(1, 6, dtype=torch.bool), "z": torch.zeros(1, 6),
             "slot": torch.zeros(1, 6, dtype=torch.long)}
    m(batch).sum().backward()
    assert all(torch.equal(a, b.running_mean) for a, b in zip(before, bns))
    unfrozen = TrackAModel({"backbone": "resnet10t", "pretrained": False, "grad_checkpointing": False,
                            "freeze_bn": False}).train()
    assert all(b.training for b in unfrozen.bn_modules())


def test_chunked_cast_gives_the_same_features():
    from knee.model.knee_vit import KneeViT
    torch.manual_seed(0)
    m = KneeViT({"slice_vit": TINY_VIT}).eval()
    x = torch.randint(0, 255, (7, 3, 32, 32), dtype=torch.uint8)
    m.window_chunk = 2
    a = m.encode_windows(x)
    m.window_chunk = 100
    torch.testing.assert_close(a, m.encode_windows(x))


# ------------------------------------------------------------------------------------------ #7, #8, shard deadline
def _tiny_arm_ckpt(path: Path) -> Path:
    from knee.model.knee_vit import KneeViT
    cfg = {"model": {"kind": "knee_vit", "cfg": {"slice_vit": {"img_size": 16, "patch": 8, "dim": 16, "depth": 1,
                                                                "heads": 2}}},
           "data": {"format": "bag", "train_windows": 2, "eval_windows": 3, "out_px": 16}}
    torch.save({"model": KneeViT(cfg["model"]["cfg"]).state_dict(), "cfg": cfg}, path)
    return path


def test_arms_with_the_same_inputs_share_one_sample(synth, tmp_path, monkeypatch):  # noqa: F811
    """5 folds of one recipe: the windows are built and collated once per study, not once per arm."""
    from knee.infer import predict as P
    arm = P.load_arm(_tiny_arm_ckpt(tmp_path / "m.pt"), torch.device("cpu"))
    dirs = [synth / "train_series" / "1.1", synth / "train_series" / "1.2"]
    n = {"sample": 0, "collate": 0}
    real_s, real_c = P.arm_sample, P.collate_arm
    monkeypatch.setattr(P, "arm_sample", lambda *a: (n.__setitem__("sample", n["sample"] + 1), real_s(*a))[1])
    monkeypatch.setattr(P, "collate_arm", lambda *a: (n.__setitem__("collate", n["collate"] + 1), real_c(*a))[1])
    one, _ = P.predict_dirs(dirs, _info(synth), [arm], torch.device("cpu"), PARAMS, workers=0)
    n.update(sample=0, collate=0)
    five, _ = P.predict_dirs(dirs, _info(synth), [arm] * 5, torch.device("cpu"), PARAMS, workers=0)
    assert n == {"sample": 2, "collate": 2}
    pd.testing.assert_frame_equal(one, five)


class _NanArm(torch.nn.Module):
    def forward(self, b):
        return torch.full((b["x"].shape[0], len(LABELS)), float("nan"))


def test_non_finite_arm_output_gets_the_fallback_row(synth, tmp_path):  # noqa: F811
    from knee.infer import predict as P
    good = P.load_arm(_tiny_arm_ckpt(tmp_path / "m.pt"), torch.device("cpu"))
    bad = (_NanArm(), good[1])
    prior = np.linspace(0.1, 0.9, 12)
    dirs = [synth / "train_series" / "1.1"]
    uids, per_arm, timing = P.predict_raw(dirs, _info(synth), [good, bad], torch.device("cpu"), PARAMS, workers=0,
                                          fallback=prior)
    np.testing.assert_allclose(per_arm[1][0], prior)
    np.testing.assert_allclose(per_arm[0][0], 0.5)  # the healthy arm keeps its own output
    assert "non-finite output from arm(s) [1]" in timing.error.iloc[0]


def test_write_submission_replaces_inf_instead_of_asserting(tmp_path):
    from knee.infer.predict import write_submission
    prior = np.full(12, 0.3)
    pred = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{k: [0.7, 0.7] for k in LABELS}})
    pred.loc[1, "ACL"] = np.inf
    sub = write_submission(pred, ["a", "b"], tmp_path / "s.csv", prior)
    np.testing.assert_allclose(sub.iloc[0, 1:].to_numpy(float), 0.7)
    np.testing.assert_allclose(sub.iloc[1, 1:].to_numpy(float), 0.3)


def test_shard_past_the_deadline_is_killed_not_rerun(tmp_path):
    from knee.infer import predict as P
    t0 = time.time()
    pred, timing, info = P.predict_dirs_multi([tmp_path / "s1", tmp_path / "s2"], {}, [(str(tmp_path / "x.pt"), {})],
                                              ["cpu"], tmp_path / "w", deadline=time.time())
    assert info["lost"] == [0] and info["retried"] == [] and len(pred) == 0
    assert time.time() - t0 < 60


# ------------------------------------------------------------------------------------------ data / preprocessing
def test_usable_drops_fully_failed_series_and_max_series_keeps_slots():
    vol = np.full((4, 8, 8), 50, np.uint8)
    rows = [{"slot": None, "positions_mm": np.arange(4.0), "plane": "Axial"},
            {"slot": "axial", "positions_mm": np.arange(4.0), "plane": "Axial", "valid": [False] * 4},
            {"slot": None, "positions_mm": np.arange(4.0), "plane": "Coronal"},
            {"slot": "sag_fluid", "positions_mm": np.arange(4.0), "plane": "Sagittal", "n_decode_fail": 1}]
    series = [(vol, r) for r in rows]
    assert [r["plane"] for _, r in D.usable(series)] == ["Axial", "Coronal", "Sagittal"]
    rng = np.random.default_rng(0)
    out = D.bag_sample(series, 2, 2, 8, False, rng, max_series=1)
    assert set(out["slot"].tolist()) == {D.SLOTS.index("sag_fluid")}  # the slotted series, not folder order
    grid = D.slots_sample(series, 2, 8, False, rng)
    assert not grid["slot_mask"][D.SLOTS.index("axial")]  # all-black series never fills a slot


def test_series_window_streaming_equals_concat_then_stride():
    from knee.data.normalize import series_window
    rng = np.random.default_rng(0)
    slices = [rng.normal(100, 30, (37, 41)).astype(np.float32) for _ in range(9)]
    flat = np.concatenate([s.ravel() for s in slices])
    for max_samples in (500, 1234, 10**9):
        ref = flat[:: int(np.ceil(flat.size / max_samples))] if flat.size > max_samples else flat
        assert series_window(slices, 1, 99, max_samples) == tuple(float(v) for v in np.percentile(ref, [1, 99]))


def test_dedup_clusters_positions_across_a_bucket_edge():
    from knee.data.preprocess import _dedup_positions

    class F:
        def __init__(self, inst, path):
            self.header, self.path = {"InstanceNumber": inst}, path
    pos = np.array([0.024, 0.026, 3.0, 3.01])  # 0.024/0.05 rounds to 0, 0.026/0.05 to 1: the old buckets kept both
    files = [F(2, "b"), F(1, "a"), F(5, "c"), F(4, "d")]
    assert sorted(_dedup_positions(pos, files)) == [1, 3]  # one per cluster, lowest InstanceNumber


def test_decode_rescale_in_float32_and_monochrome1_per_series():
    from pydicom.dataset import Dataset

    from knee.data.dicom_io import decode_pixels

    class DS(Dataset):
        pass
    arrs = [np.array([[0, 10], [20, 30]], np.uint16), np.array([[0, 5], [5, 5]], np.uint16)]
    outs = []
    for arr in arrs:
        ds = DS()
        ds.RescaleSlope, ds.RescaleIntercept, ds.PhotometricInterpretation = 2.5, 1.0, "MONOCHROME1"
        DS.pixel_array = property(lambda self, a=arr: a)
        outs.append(decode_pixels(ds))
    for arr, img in zip(arrs, outs):
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, -(arr * 2.5 + 1.0))  # negated: the same mapping for every slice


def test_render_slots_decodes_only_slotted_series(tmp_path, monkeypatch):
    from knee.data import preprocess as PP
    root = tmp_path / "ts"
    rows = []
    for j, (plane, fluid) in enumerate((("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Axial", 1))):
        uid = f"7.{j}"
        make_series(root, "7", uid, plane, n=6 if j < 3 else 4, desc="pd_tse_fs")
        rows.append({"SeriesInstanceUID": uid, "Fluid_Sensitive": fluid, "Anatomical_Plane": plane})
    info = PP.series_info_from_csv(pd.DataFrame(rows))
    rendered = []
    real = PP._render_series
    monkeypatch.setattr(PP, "_render_series", lambda sh, *a: (rendered.append(sh.series_uid), real(sh, *a))[1])
    full, sm_full = preprocess_study(root / "7", info, PARAMS)
    assert len(rendered) == 4
    rendered.clear()
    sl, sm_sl = preprocess_study(root / "7", info, PARAMS, render="slots")
    assert sorted(rendered) == ["7.0", "7.1", "7.2"]  # the 4-slice second axial gets no slot
    assert sm_sl["slots"] == sm_full["slots"]
    by_uid = {s.series_uid: s for s in full}
    for s in sl:
        np.testing.assert_array_equal(s.pixels, by_uid[s.series_uid].pixels)
        assert s.meta["slot"] == by_uid[s.series_uid].meta["slot"]


def test_derive_shard_handles_an_empty_shard(tmp_path):
    from knee.data.derive_cache import derive_shard
    src = tmp_path / "shard_00000.u8"
    src.write_bytes(b"")
    assert derive_shard(src, tmp_path / "out.u8", 64, 32) == 0 and (tmp_path / "out.u8").stat().st_size == 0


def test_load_paths_needs_only_cache_root(tmp_path, monkeypatch):
    from knee.paths import load_paths
    for k in ("KNEE_DATA_ROOT", "KNEE_CACHE_ROOT", "KNEE_AUDIT_OUT"):
        monkeypatch.delenv(k, raising=False)
    cf = tmp_path / "paths.yaml"
    cf.write_text(yaml.safe_dump({"cache_root": str(tmp_path / "cache")}))
    p = load_paths(cf)
    assert p.cache_root == tmp_path / "cache"
    with pytest.raises(KeyError, match="audit_out"):
        _ = p.audit_out
    with pytest.raises(KeyError, match="data_root"):
        _ = p.train_csv


def test_cache_resume_refuses_changed_preprocessing_code(tmp_path):
    from knee.data import build_cache
    from knee.data.preprocess import preproc_code_hash
    studies = ["a", "b"]
    old = {"params": {"x": 1}, "studies_per_shard": 5, "studies_sha1": build_cache.studies_sha1(studies),
           "preproc_code_hash": preproc_code_hash()}
    build_cache.check_resume(old, {"x": 1}, studies, 5, tmp_path / "m.json")
    with pytest.raises(SystemExit, match="preprocessing code changed"):
        build_cache.check_resume({**old, "preproc_code_hash": "0" * 16}, {"x": 1}, studies, 5, tmp_path / "m.json")


def test_notebook_preproc_parity_check():
    from knee.data.preprocess import PREPROC_VERSION, PreprocParams, preproc_code_hash
    nb = _load("infer_nb_parity", REPO / "notebooks" / "infer.py")
    h = preproc_code_hash()
    full = PreprocParams(out_px=64).to_dict()
    assert nb.check_preproc({"preproc": full, "preproc_code_hash": h}, PreprocParams, PREPROC_VERSION, h).out_px == 64
    with pytest.raises(RuntimeError, match="no preprocessing block"):
        nb.check_preproc({}, PreprocParams, PREPROC_VERSION, h)
    with pytest.raises(RuntimeError, match="lacks"):
        nb.check_preproc({"preproc": {"version": PREPROC_VERSION}}, PreprocParams, PREPROC_VERSION, h)
    drift = {"preproc": full, "preproc_code_hash": "f" * 16}
    with pytest.raises(RuntimeError, match="code hash"):
        nb.check_preproc(drift, PreprocParams, PREPROC_VERSION, h)
    assert nb.check_preproc(drift, PreprocParams, PREPROC_VERSION, h, allow_drift=True).out_px == 64
    assert nb.check_preproc({"preproc": full}, PreprocParams, PREPROC_VERSION, h).out_px == 64  # old bundle: warns


def test_bundle_carries_the_code_hash(tmp_path):
    from knee.data.preprocess import PreprocParams
    bb = _load("build_kaggle_bundle_hash", REPO / "scripts" / "build_kaggle_bundle.py")
    ck = _tiny_arm_ckpt(tmp_path / "m.pt")
    spec = bb.build(tmp_path / "b", [ck], [1.0], "mean", None, None, "u", "s", PreprocParams().to_dict(), "abc")
    assert json.loads((tmp_path / "b" / "arms.json").read_text())["preproc_code_hash"] == "abc" == \
        spec["preproc_code_hash"]


# ------------------------------------------------------------------------------------------ MAE
def test_mae_resume_refuses_a_schedule_change():
    from knee.ssl.pretrain_mae import DEFAULTS, check_resume_cfg, merge
    old = merge(DEFAULTS, {"data": {"batch_size": 64}})
    check_resume_cfg(old, merge(DEFAULTS, {"data": {"batch_size": 64}, "ckpt_minutes": 5}))  # harmless change
    with pytest.raises(SystemExit, match="data.batch_size: 64 → 32"):
        check_resume_cfg(old, merge(DEFAULTS, {"data": {"batch_size": 32}}))
    check_resume_cfg(old, merge(DEFAULTS, {"data": {"batch_size": 32}}), allow=True)


def test_mae_norm_pix_eps_damps_flat_patches():
    from knee.model.slice_vit import SliceViT
    from knee.ssl.mae import MAE
    torch.manual_seed(0)
    x = torch.full((1, 3, 32, 32), 10, dtype=torch.uint8)
    x[..., ::2, ::2] = 11  # one grey level of "noise"
    t6 = MAE(SliceViT(**TINY_VIT), dec_dim=32, dec_depth=1, dec_heads=2)(x)["target"]
    t2 = MAE(SliceViT(**TINY_VIT), dec_dim=32, dec_depth=1, dec_heads=2, norm_pix_eps=1e-2)(x)["target"]
    assert t6.std() > 0.9 and t2.std() < 0.1
