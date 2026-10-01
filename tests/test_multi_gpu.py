"""Both Kaggle T4s: inference sharded over several devices must equal the one-device path (the arm blend, including
the rank blend, runs once over all studies), a dead shard must be retried and then fall back, and the fold launcher
must run folds concurrently, one per GPU slot. Devices here are 'cpu' processes: the logic is device-agnostic."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

from test_data import make_series  # noqa: E402

from knee.data.preprocess import PreprocParams, series_info_from_csv  # noqa: E402
from knee.infer import predict as P  # noqa: E402
from knee.paths import LABELS  # noqa: E402


def _ckpt(path: Path, seed: int) -> str:
    from knee.model.knee_vit import KneeViT
    cfg = {"model": {"kind": "knee_vit", "cfg": {"slice_vit": {"img_size": 16, "patch": 8, "dim": 16, "depth": 1,
                                                                "heads": 2}}},
           "data": {"format": "bag", "train_windows": 2, "eval_windows": 3, "out_px": 16}}
    m = KneeViT(cfg["model"]["cfg"])
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # random weights everywhere (zero-init heads would make every output 0.5)
        for p in m.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    torch.save({"model": m.state_dict(), "cfg": cfg}, path)
    return str(path)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi")
    rows = []
    studies = [f"7.{i}" for i in range(5)]
    for s_i, study in enumerate(studies):
        planes = (("Sagittal", 1), ("Coronal", 1), ("Axial", 0))[: 1 + s_i % 3]  # 1–3 series: unequal costs
        for j, (plane, fluid) in enumerate(planes):
            uid = f"{study}.{j}"
            make_series(tmp / "test_series", study, uid, plane, n=4 + s_i, x0=-90.0 + 40 * s_i,
                        desc="pd_tse_fs" if fluid else "pd_tse")
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": uid, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    ts = pd.DataFrame(rows)
    dirs = [tmp / "test_series" / u for u in [*studies, "7.missing"]]
    specs = [(_ckpt(tmp / "a.pt", 1), {}), (_ckpt(tmp / "b.pt", 2), {})]
    cost = ts.groupby("StudyInstanceUID").size().astype(float).to_dict()
    return tmp, dirs, series_info_from_csv(ts), specs, cost


def test_shard_studies_balances_cost_and_keeps_order():
    dirs = [Path(f"s{i}") for i in range(7)]
    cost = {"s0": 5, "s1": 1, "s2": 1, "s3": 4, "s4": 1, "s5": 1, "s6": 1}
    shards = P.shard_studies(dirs, 2, cost)
    assert sorted(i for s in shards for i in s) == list(range(7))
    assert all(s == sorted(s) for s in shards)
    loads = [sum(cost[f"s{i}"] for i in s) for s in shards]
    assert max(loads) - min(loads) <= 1
    assert P.shard_studies(dirs, 3) and len(P.shard_studies(dirs[:1], 2)[1]) == 0  # more shards than studies is fine


@pytest.mark.parametrize("how", ["mean", "rank"])
def test_multi_device_equals_single_device(setup, how):
    tmp, dirs, info, specs, cost = setup
    params = PreprocParams(out_px=64)
    prior = np.full(len(LABELS), 0.3)
    arms = [P.load_arm(ck, torch.device("cpu"), perf) for ck, perf in specs]
    single, t1 = P.predict_dirs(dirs, info, arms, torch.device("cpu"), params, workers=0, weights=[1.0, 2.0],
                                how=how, fallback=prior)
    multi, t2, meta = P.predict_dirs_multi(dirs, info, specs, ["cpu", "cpu"], tmp / f"shards_{how}", params,
                                           workers_per_device=0, weights=[1.0, 2.0], how=how, fallback=prior,
                                           cost=cost)
    assert meta["retried"] == [] and meta["lost"] == [] and sum(meta["shard_sizes"]) == len(dirs)
    assert all(n > 0 for n in meta["shard_sizes"])
    a = single.set_index("StudyInstanceUID").sort_index()
    b = multi.set_index("StudyInstanceUID").sort_index()
    assert a.index.tolist() == b.index.tolist()
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=0, atol=1e-6)
    assert a.loc[[u for u in a.index if u != "7.missing"]].std().min() > 0  # the test means something
    assert set(t2.device) == {"cpu"} and len(t2) == len(t1)
    assert t2.set_index("StudyInstanceUID").loc["7.missing", "error"]


def test_dead_shard_is_retried_then_falls_back(setup, tmp_path):
    _, dirs, info, specs, cost = setup
    prior = np.full(len(LABELS), 0.3)
    # cuda:99 exists nowhere (CPU build: no CUDA; GPU box: invalid ordinal): the child dies, the retry fails too
    pred, timing, meta = P.predict_dirs_multi(dirs, info, specs, ["cpu", "cuda:99"], tmp_path, PreprocParams(out_px=64),
                                              workers_per_device=0, fallback=prior, cost=cost)
    assert meta["retried"] == [1] and meta["lost"] == [1]
    expected = [d.name for d in dirs]
    sub = P.write_submission(pred, expected, tmp_path / "submission.csv", prior)
    lost = [expected[i] for i in P.shard_studies(dirs, 2, cost)[1]]
    assert lost and np.allclose(sub.set_index("StudyInstanceUID").loc[lost, LABELS].to_numpy(), 0.3)
    assert sub.StudyInstanceUID.tolist() == expected


def test_multi_fold_launcher_runs_one_fold_per_slot(tmp_path):
    from knee.train import multi_fold as MF
    stub = tmp_path / "stub.py"  # stands in for train_sup: records its fold, GPU and overlap with the others
    stub.write_text(
        "import json, os, sys, time\n"
        "fold = [a.split('=')[1] for a in sys.argv if a.startswith('folds.fold=')][0]\n"
        "t0 = time.time(); time.sleep(1.0)\n"
        "json.dump({'fold': int(fold), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 't0': t0, 't1': time.time(),\n"
        "           'argv': sys.argv[1:]}, open(os.path.join(sys.argv[1], f'f{fold}.json'), 'w'))\n"
        "sys.exit(3 if fold == '2' else 0)\n")
    res = MF.run_folds([0, 1, 2], ["0", "1"], lambda fold: [sys.executable, str(stub), str(tmp_path),
                                                            f"folds.fold={fold}"],
                       tmp_path / "logs", poll_s=0.05)
    recs = {r["fold"]: r for r in (json.loads((tmp_path / f"f{k}.json").read_text()) for k in range(3))}
    assert recs[0]["gpu"] != recs[1]["gpu"]                       # folds 0 and 1 ran on different GPUs …
    assert recs[0]["t0"] < recs[1]["t1"] and recs[1]["t0"] < recs[0]["t1"]  # … at the same time
    assert recs[2]["t0"] >= min(recs[0]["t1"], recs[1]["t1"]) - 0.05       # fold 2 waited for a free GPU
    assert res == {0: 0, 1: 0, 2: 3}
    assert (tmp_path / "logs" / "fold2.log").exists()


# ------------------------------------------------------------------------------------------ multi_fold.main (bug.md, 2026-09-30)
def _mf_main(tmp_path, monkeypatch, epochs=3, extra=()):
    """multi_fold.main with run_folds replaced by a recorder (no training): returns (exit code, launched folds, the
    command each fold would run, the log dir)."""
    import yaml
    from knee.train import multi_fold as MF
    cfg = {"cache_dir": str(tmp_path / "cache"), "run_dir": str(tmp_path / "X"), "optim": {"epochs": epochs}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    seen = {}

    def fake_run_folds(folds, gpus, cmd_for_fold, log_dir, **kw):
        seen.update(folds=list(folds), cmds={f: cmd_for_fold(f) for f in folds}, log_dir=Path(log_dir))
        return {f: 0 for f in folds}
    monkeypatch.setattr(MF, "run_folds", fake_run_folds)
    rc = MF.main(["--config", str(tmp_path / "c.yaml"), "--folds", "0", "1", "--gpus", "0", "1", *extra])
    return rc, seen


def _run_dir_arg(cmd):
    import yaml
    return yaml.safe_load([a for a in cmd if a.startswith("run_dir=")][-1].split("=", 1)[1])


def test_multi_fold_explicit_run_dir_is_a_root(tmp_path, monkeypatch):
    """Bug 1: an explicit run_dir shared by every fold becomes the run root, one subdir per fold."""
    rc, seen = _mf_main(tmp_path, monkeypatch)
    assert seen["folds"] == [0, 1] and rc == 1  # nothing wrote metrics.json
    assert {f: _run_dir_arg(c) for f, c in seen["cmds"].items()} == {0: str(tmp_path / "X" / "fold0"),
                                                                     1: str(tmp_path / "X" / "fold1")}
    assert seen["log_dir"] == tmp_path / "X" / "multi_fold"


def test_multi_fold_skips_finished_folds(tmp_path, monkeypatch):
    """Bug 2: a resumed multi_fold must not re-launch a fold whose metrics.json says it finished."""
    for f in (0, 1):
        (tmp_path / "X" / f"fold{f}").mkdir(parents=True)
    (tmp_path / "X" / "fold0" / "metrics.json").write_text(json.dumps({"fold": 0, "epochs": 3}))
    (tmp_path / "X" / "fold1" / "metrics.json").write_text(json.dumps({"fold": 1, "epochs": 2}))  # epochs raised
    rc, seen = _mf_main(tmp_path, monkeypatch, epochs=3)
    assert seen["folds"] == [1] and rc == 1
    rc, seen = _mf_main(tmp_path, monkeypatch, epochs=3, extra=["--rerun-finished"])
    assert seen["folds"] == [0, 1]
    (tmp_path / "X" / "fold1" / "metrics.json").write_text(json.dumps({"fold": 1, "epochs": 3}))
    (tmp_path / "X" / "fold0" / "metrics.json").write_text(json.dumps({"fold": 0}))  # pre-fix format: counts as done
    seen.clear()
    rc, seen = _mf_main(tmp_path, monkeypatch, epochs=3)
    assert rc == 0 and seen == {}  # all done: nothing spawned


def test_run_folds_clears_stale_stop(tmp_path):
    """A STOP left by an earlier deadline must not pause the fold at its first step."""
    from knee.train import multi_fold as MF
    stop = {f: tmp_path / f"fold{f}" / "STOP" for f in (0, 1)}
    for p in stop.values():
        p.parent.mkdir()
        p.touch()
    stub = "import os, sys; sys.exit(7 if os.path.exists(sys.argv[1]) else 0)"
    res = MF.run_folds([0, 1], ["0"], lambda f: [sys.executable, "-c", stub, str(stop[f])], tmp_path / "logs",
                       poll_s=0.05, stop_file_for_fold=stop.get)
    assert res == {0: 0, 1: 0} and not any(p.exists() for p in stop.values())
