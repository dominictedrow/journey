"""MAE + training-loop tests (CPU, tiny models on a fake cache): shapes, no information leak through the conv stem,
LLRD groups, losses, metrics, and full pretrain/supervised runs including STOP + resume."""
import json
import math

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from fake_cache import make_fake_cache  # noqa: E402
from knee.data.dataset import CacheReader  # noqa: E402
from knee.model.knee_vit import KneeViT  # noqa: E402
from knee.model.slice_vit import SliceViT  # noqa: E402
from knee.ssl.mae import MAE, patchify, random_masking, unpatchify  # noqa: E402
from knee.ssl.mae_data import MAEWindowDataset, random_resized_crop  # noqa: E402
from knee.train import common as C  # noqa: E402
from knee.train.losses import RankBank, pairwise_rank_loss, soft_bce  # noqa: E402
from knee.train.optim import param_groups, warmup_cosine  # noqa: E402
from knee.utils.metrics import binarize, bootstrap_macro_auc, macro_auc, per_finding_auc, rank_blend  # noqa: E402

TINY_VIT = {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2}


# ------------------------------------------------------------------------------------------ MAE
def test_patchify_roundtrip_and_masking():
    x = torch.randn(2, 3, 32, 48)
    torch.testing.assert_close(unpatchify(patchify(x, 8), 8, (4, 6), 3), x)
    keep, restore, mask = random_masking(5, 24, 0.75, "cpu")
    assert keep.shape == (5, 6) and (mask.sum(1) == 18).all()
    for i in range(5):  # kept ids are exactly the unmasked positions
        assert set(keep[i].tolist()) == set(torch.nonzero(mask[i] == 0).flatten().tolist())


def test_mae_forward_and_learns():
    torch.manual_seed(0)
    mae = MAE(SliceViT(**TINY_VIT), dec_dim=32, dec_depth=1, dec_heads=2)
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    out = mae(x)
    assert out["pred"].shape == (4, 16, 8 * 8 * 3) and out["mask"].sum() == 4 * 12
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-3)
    first = None
    for _ in range(40):
        loss = mae(x, generator=torch.Generator().manual_seed(0))["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.9


def test_conv_stem_mae_does_not_leak_masked_pixels():
    """With zero_masked_pixels, the encoder output must not depend on what is under the masked patches."""
    torch.manual_seed(0)
    mae = MAE(SliceViT(**TINY_VIT, tokenizer="conv_stem"), dec_dim=32, dec_depth=1, dec_heads=2).eval()
    assert mae.zero_masked_pixels
    x1 = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    _, _, mask = random_masking(1, 16, 0.75, "cpu", torch.Generator().manual_seed(3))
    pix = mae._pixel_mask(mask, (4, 4)).bool().expand(1, 3, 32, 32)
    x2 = x1.clone()
    x2[pix] = 255 - x2[pix]  # change only the masked pixels
    o1 = mae(x1, generator=torch.Generator().manual_seed(3))
    o2 = mae(x2, generator=torch.Generator().manual_seed(3))
    vis = o1["mask"][0] == 0
    # the prediction for masked patches only depends on the visible tokens → identical
    torch.testing.assert_close(o1["pred"][0, ~vis], o2["pred"][0, ~vis])


def test_mae_dataset_and_crop(tmp_path):
    f = make_fake_cache(tmp_path / "c", n_studies=3, px=64)
    ds = MAEWindowDataset(CacheReader(f["cache"]), out_px=32, windows_per_series=2)
    assert len(ds) == 2 * ds.n_series
    x = ds[0]
    assert x.shape == (1, 3, 32, 32) and x.dtype == torch.uint8  # (windows_per_read, C, H, W)
    assert torch.equal(ds[0], x)  # deterministic given (seed, epoch, index): resumable
    win = np.random.default_rng(0).integers(0, 255, (3, 64, 64), dtype=np.uint8)
    out = random_resized_crop(win, 32, np.random.default_rng(0), scale=(1, 1), ratio=(1, 1))
    assert out.shape == (3, 32, 32)


# ------------------------------------------------------------------------------------------ optim / losses / metrics
def test_llrd_groups():
    m = KneeViT({"slice_vit": TINY_VIT})
    groups = param_groups(m, 1e-3, 0.05, layer_decay=0.5, layer_id=m.layer_id, n_layers=m.n_layers)
    by_name = {n: g for g in groups for n in g["names"]}
    assert by_name["head.cls_w"]["lr"] == pytest.approx(1e-3)
    assert by_name["encoder.blocks.1.attn.qkv.weight"]["lr"] == pytest.approx(1e-3 * 0.5)
    assert by_name["encoder.tokenizer.proj.0.weight"]["lr"] == pytest.approx(1e-3 * 0.5 ** 3)
    assert by_name["encoder.blocks.0.norm1.weight"]["weight_decay"] == 0.0
    assert by_name["series_type.tables.plane.weight"]["weight_decay"] == 0.0
    assert by_name["encoder.blocks.0.attn.qkv.weight"]["weight_decay"] == 0.05
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1) and warmup_cosine(100, 10, 100) == pytest.approx(0)


def test_losses():
    logits = torch.zeros(2, 12, requires_grad=True)
    y = torch.rand(2, 12)
    assert soft_bce(logits, y).item() == pytest.approx(math.log(2), rel=1e-6)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    assert soft_bce(logits, y, mask).item() == 0.0
    # no pairs → 0 but differentiable
    r = pairwise_rank_loss(logits, torch.ones(2, 12))
    r.backward()
    assert r.item() == 0.0
    # a correctly ordered pair has lower loss than a reversed one
    s = torch.tensor([[2.0] * 12, [-2.0] * 12])
    t = torch.tensor([[1.0] * 12, [0.0] * 12])
    assert pairwise_rank_loss(s, t) < pairwise_rank_loss(-s, t)
    bank = RankBank(3)
    for _ in range(5):
        bank.push(torch.randn(1, 12), torch.ones(1, 12), torch.ones(1, 12, dtype=torch.bool))
    assert len(bank.get()[0]) == 3
    cur = torch.full((1, 12), 0.0, requires_grad=True)  # a negative ranked against banked positives
    loss = pairwise_rank_loss(cur, torch.zeros(1, 12), None, *bank.get())
    loss.backward()
    assert loss.item() > 0 and cur.grad.abs().sum() > 0


def test_metrics():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
    p = np.array([[0.9, 0.1], [0.2, 0.3], [0.8, 0.5], [0.1, 0.2]])
    a = per_finding_auc(y, p)
    assert a[0] == 1.0 and np.isnan(a[1])  # column 2 has one class → skipped
    assert macro_auc(y, p) == (1.0, 1)
    b = binarize(np.array([0.1, 0.5, 0.9, np.nan]))
    assert b[0] == 0 and np.isnan(b[1]) and b[2] == 1 and np.isnan(b[3])
    bs = bootstrap_macro_auc(y, p, n_boot=50)
    assert bs["lo"] <= bs["auc"] <= bs["hi"]
    rb = rank_blend([p, p[:, ::-1]])
    assert rb.shape == p.shape


def test_average_state_dicts():
    a = {"w": torch.ones(2), "n": torch.tensor(3)}
    b = {"w": torch.zeros(2), "n": torch.tensor(5)}
    avg = C.average_state_dicts([a, b])
    torch.testing.assert_close(avg["w"], torch.full((2,), 0.5))
    assert avg["n"].item() == 3


# ------------------------------------------------------------------------------------------ full runs
def _write(path, d):
    path.write_text(yaml.safe_dump(d))
    return path


def test_pretrain_mae_run_stop_resume(tmp_path):
    from knee.ssl import pretrain_mae
    f = make_fake_cache(tmp_path / "c", n_studies=4, px=48)
    cfg = {"run_name": "t", "cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu",
           "precision": "fp32", "model": {**TINY_VIT, "drop_path": 0.0},
           "mae": {"dec_dim": 32, "dec_depth": 1, "dec_heads": 2},
           "data": {"out_px": 32, "windows_per_series": 2, "batch_size": 4, "workers": 0},
           "optim": {"epochs": 2, "warmup_epochs": 1}, "log_every": 1, "recon_every_steps": 3,
           "snapshot_every_epochs": 1}
    cp = _write(tmp_path / "mae.yaml", cfg)
    pilot = pretrain_mae.main(["--config", str(cp), "--pilot", "7"])
    assert pilot["windows_per_s"] > 0 and (tmp_path / "run" / "pilot" / "pilot.json").exists()
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run" / "STOP").touch()
    r = pretrain_mae.main(["--config", str(cp)])
    assert r["stopped"] and (tmp_path / "run" / "last.pt").exists()
    r = pretrain_mae.main(["--config", str(cp)])  # resumes and finishes
    assert r["epoch"] == 2
    enc = torch.load(tmp_path / "run" / "encoder_final.pt", weights_only=False)
    m = KneeViT({"slice_vit": TINY_VIT})
    assert m.load_encoder(enc["encoder"]) == ([], [])
    assert list((tmp_path / "run" / "recon").glob("*.png"))


@pytest.mark.parametrize("head", ["attn_mil", "finding_query"])
def test_train_sup_knee_vit(tmp_path, head):
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=8, px=48)
    hcfg = {"kind": head} if head == "attn_mil" else {"kind": head, "heads": 2}
    cfg = {"run_name": "t", "cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu",
           "precision": "fp32", "model": {"kind": "knee_vit", "cfg": {"slice_vit": TINY_VIT, "head": hcfg}},
           "labels": {"file": str(f["labels"])}, "folds": {"file": str(f["folds"]), "fold": 0},
           "gold": {"file": str(f["gold"])},
           "data": {"out_px": 32, "train_windows": 2, "eval_windows": 3, "batch_size": 2, "workers": 0},
           "optim": {"epochs": 3, "accum": 1, "layer_decay": 0.75}, "select": {"top_k": 2}, "log_every": 1}
    cp = _write(tmp_path / "sup.yaml", cfg)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "STOP").touch()
    assert train_sup.main(["--config", str(cp)])["stopped"]
    res = train_sup.main(["--config", str(cp)])
    assert len(res["top_epochs"]) == 2 and np.isfinite(res["val_macro_auc"])
    assert (tmp_path / "run" / "oof.parquet").exists() and (tmp_path / "run" / "auc_table.md").exists()
    assert json.loads((tmp_path / "run" / "metrics.json").read_text())["fold"] == 0


def test_train_sup_init_from_mae_encoder(tmp_path):
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=4, px=48)
    enc = SliceViT(**TINY_VIT)
    C.save_atomic({"encoder": enc.state_dict(), "epoch": 1}, tmp_path / "enc.pt")
    cfg = {"cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu", "precision": "fp32",
           "model": {"kind": "knee_vit", "cfg": {"slice_vit": TINY_VIT}}, "init": {"encoder": str(tmp_path / "enc.pt")},
           "labels": {"file": str(f["labels"])}, "folds": {"file": str(f["folds"]), "fold": -1},
           "data": {"out_px": 32, "train_windows": 2, "batch_size": 1, "workers": 0},
           "optim": {"epochs": 1, "accum": 2}, "select": {"top_k": 1}}
    res = train_sup.main(["--config", str(_write(tmp_path / "s.yaml", cfg))])
    assert res["top_epochs"] == [0]


def test_train_sup_track_a_slots(tmp_path):
    pytest.importorskip("timm")
    from knee.train import train_sup
    f = make_fake_cache(tmp_path / "c", n_studies=6, px=48)
    cfg = {"cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu", "precision": "fp32",
           "model": {"kind": "trackA", "cfg": {"backbone": "resnet10t", "pretrained": False,
                                               "grad_checkpointing": False}},
           "labels": {"file": str(f["labels"])}, "folds": {"file": str(f["folds"]), "fold": 1},
           "data": {"format": "slots", "out_px": 32, "train_windows": 2, "eval_windows": 3, "batch_size": 2,
                    "workers": 0},
           "optim": {"epochs": 1, "accum": 1, "backbone_lr_mult": 0.1}, "select": {"top_k": 1}}
    res = train_sup.main(["--config", str(_write(tmp_path / "a.yaml", cfg))])
    assert np.isfinite(res["val_macro_auc"]) or res["val_n_findings"] == 0
