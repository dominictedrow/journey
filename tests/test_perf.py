"""Performance step 1 (docs/plan/03_performance_plan.md §5): every change here must leave the results unchanged.

- the sync-free rank loss equals the old per-finding loop; branch-free mask helpers equal the old branches
- fp16/auto precision plumbing and the GPU-side loss meter
- MAE loader with several windows per series read
- threaded DICOM reads give the same preprocessing output as serial reads
- inference folds (LayerScale, conv+BN) give the same outputs
"""
import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from fake_cache import make_fake_cache  # noqa: E402
from knee.data.dataset import CacheReader  # noqa: E402
from knee.model.attention import _safe_key_mask  # noqa: E402
from knee.model.fuse import fold_bn_into_conv, fold_layerscale, fold_timm_conv_bn, prepare_for_inference  # noqa: E402
from knee.model.knee_vit import KneeViT, drop_windows  # noqa: E402
from knee.ssl.mae_data import MAEWindowDataset  # noqa: E402
from knee.train import common as C  # noqa: E402
from knee.train.losses import pairwise_rank_loss  # noqa: E402

TINY = {"slice_vit": {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2, "drop_path": 0.0},
        "head": {"kind": "finding_query", "heads": 2}}


# ------------------------------------------------------------------------------------------ host syncs
def _rank_loss_loop(logits, targets, label_mask=None, ref_logits=None, ref_targets=None, ref_mask=None,
                    pos_thr=0.6, neg_thr=0.4):
    """The previous implementation (boolean indexing per finding), kept as the reference."""
    s, y = logits.float(), targets.float()
    m = torch.ones_like(y, dtype=torch.bool) if label_mask is None else label_mask.bool()
    if ref_logits is not None and len(ref_logits):
        rs, ry = ref_logits.float().detach(), ref_targets.float()
        rm = torch.ones_like(ry, dtype=torch.bool) if ref_mask is None else ref_mask.bool()
    else:
        rs = ry = rm = None
    per_label = []
    for k in range(y.shape[1]):
        pos, neg = m[:, k] & (y[:, k] > pos_thr), m[:, k] & (y[:, k] < neg_thr)
        diffs = [s[pos, k][:, None] - s[neg, k][None, :]]
        if rs is not None:
            rpos, rneg = rm[:, k] & (ry[:, k] > pos_thr), rm[:, k] & (ry[:, k] < neg_thr)
            diffs += [s[pos, k][:, None] - rs[rneg, k][None, :], rs[rpos, k][:, None] - s[neg, k][None, :]]
        d = torch.cat([x.flatten() for x in diffs])
        if d.numel():
            per_label.append(F.softplus(-d).mean())
    if not per_label:
        return logits.sum() * 0.0
    return torch.stack(per_label).mean()


@pytest.mark.parametrize("B,R", [(1, 0), (2, 0), (1, 7), (3, 20)])
def test_rank_loss_vectorized_equals_loop(B, R):
    g = torch.Generator().manual_seed(B * 100 + R)
    logits = torch.randn(B, 12, generator=g, requires_grad=True)
    y = torch.rand(B, 12, generator=g)
    m = torch.rand(B, 12, generator=g) > 0.2
    ref = (torch.randn(R, 12, generator=g), torch.rand(R, 12, generator=g), torch.rand(R, 12, generator=g) > 0.2) \
        if R else (None, None, None)
    a = pairwise_rank_loss(logits, y, m, *ref)
    b = _rank_loss_loop(logits, y, m, *ref)
    torch.testing.assert_close(a, b)
    ga, = torch.autograd.grad(a, logits)
    gb, = torch.autograd.grad(_rank_loss_loop(logits, y, m, *ref), logits)
    torch.testing.assert_close(ga, gb)


def test_safe_key_mask_and_drop_windows_branch_free():
    m = torch.tensor([[False, False, False], [True, False, True], [False, True, False]])
    torch.testing.assert_close(_safe_key_mask(m), torch.tensor([[True, False, False], [True, False, True],
                                                                 [False, True, False]]))
    torch.manual_seed(0)
    mask = torch.tensor([[False, True, True, True], [False, False, False, False]])
    for _ in range(20):
        out = drop_windows(mask, 0.99)
        assert out[0].sum() >= 1 and not (out & ~mask).any() and not out[1].any()


# ------------------------------------------------------------------------------------------ precision
def test_precision_resolution_and_scaler_on_cpu():
    cpu = torch.device("cpu")
    assert C.resolve_precision(cpu, "auto") == "fp32"
    assert C.resolve_precision(cpu, "fp16") == "fp16"
    assert not C.make_scaler(cpu, "fp16").is_enabled()  # the scaler only runs on CUDA


def test_gpu_meter_counts_nonfinite():
    m = C.GpuMeter(torch.device("cpu"))
    for v in (1.0, 3.0, float("nan"), float("inf")):
        m.add(torch.tensor(v))
    mean, bad = m.read()
    assert bad == 2 and mean == pytest.approx(4.0 / 4)  # finite sum / all micro-batches


# ------------------------------------------------------------------------------------------ MAE loader
def test_mae_windows_per_read(tmp_path):
    f = make_fake_cache(tmp_path / "c", n_studies=3, px=64)
    r = CacheReader(f["cache"])
    one = MAEWindowDataset(r, out_px=32, windows_per_series=4)
    four = MAEWindowDataset(r, out_px=32, windows_per_series=4, windows_per_read=4)
    assert one.n_windows == four.n_windows == 4 * one.n_series and len(four) == one.n_series
    x = four[0]
    assert x.shape == (4, 3, 32, 32) and x.dtype == torch.uint8 and torch.equal(four[0], x)
    assert one[0].shape == (1, 3, 32, 32)
    with pytest.raises(ValueError):
        MAEWindowDataset(r, windows_per_series=4, windows_per_read=3)


def test_pretrain_mae_with_windows_per_read(tmp_path):
    from knee.ssl import pretrain_mae
    f = make_fake_cache(tmp_path / "c", n_studies=4, px=48)
    cfg = {"cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu", "precision": "auto",
           "model": {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2, "drop_path": 0.0},
           "mae": {"dec_dim": 32, "dec_depth": 1, "dec_heads": 2},
           "data": {"out_px": 32, "windows_per_series": 2, "windows_per_read": 2, "batch_size": 4, "workers": 0},
           "optim": {"epochs": 1, "warmup_epochs": 0}, "log_every": 1, "recon_every_steps": 0}
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(cfg))
    r = pretrain_mae.main(["--config", str(tmp_path / "m.yaml")])
    assert r["epoch"] == 1 and (tmp_path / "run" / "encoder_final.pt").exists()


# ------------------------------------------------------------------------------------------ threaded reads
def test_threaded_reads_equal_serial(tmp_path, monkeypatch):
    pytest.importorskip("pydicom")
    from test_data import PARAMS, make_series
    from knee.data import dicom_io
    from knee.data.preprocess import preprocess_study
    for j, plane in enumerate(["Sagittal", "Coronal", "Axial"]):
        make_series(tmp_path, "9.1", f"9.1.{j}", plane, n=7)
    runs = []
    for threads in (1, 8):
        monkeypatch.setattr(dicom_io, "IO_THREADS", threads)
        dicom_io._pool = None
        runs.append(preprocess_study(tmp_path / "9.1", None, PARAMS))
    (s1, m1), (s8, m8) = runs
    assert m1 == m8 and len(s1) == len(s8) == 3
    for a, b in zip(s1, s8):
        assert np.array_equal(a.pixels, b.pixels) and np.array_equal(a.positions_mm, b.positions_mm)
        assert a.meta == b.meta
    (tmp_path / "9.1" / "9.1.0" / "bad.dcm").write_bytes(b"not a dicom")  # still skipped, never fatal
    assert len(preprocess_study(tmp_path / "9.1", None, PARAMS)[0]) == 3


# ------------------------------------------------------------------------------------------ inference folds
def _randomize_layerscale(m):
    for n, p in m.named_parameters():
        if n.endswith(".gamma"):
            p.data = torch.rand_like(p) * 2 - 0.5


def test_fold_layerscale_same_output():
    from test_model import fake_bag
    torch.manual_seed(0)
    m = KneeViT(TINY).eval()
    _randomize_layerscale(m)
    b = fake_bag()
    with torch.no_grad():
        ref = m(b)
        n = fold_layerscale(m)
        out = m(b)
    assert n == 2 * 2 + 2 * 2 + 2 * 2  # 2 encoder blocks + 2 ctx blocks + 2 query blocks, 2 LayerScales each
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not any(k.endswith(".gamma") for k in m.state_dict())


def test_fold_bn_into_conv_same_output():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1, bias=False)
    bn = torch.nn.BatchNorm2d(6)
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-1, 1)
    bn.eval()
    x = torch.randn(2, 4, 9, 9)
    with torch.no_grad():
        ref = bn(conv(x))
        fold_bn_into_conv(conv, bn)
        torch.testing.assert_close(conv(x), ref, rtol=1e-5, atol=1e-5)


def test_fold_timm_coatnet_same_output():
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    m = timm.create_model("coatnet_nano_rw_224", pretrained=False, num_classes=0, img_size=64).eval()
    for mod in m.modules():  # non-trivial BN statistics, as after training
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.5, 0.5)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref = m(x)
        folded = prepare_for_inference(m, conv_bn=True)
        out = m(x)
    assert folded["conv_bn"] > 0
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert fold_timm_conv_bn(m) == 0  # idempotent: nothing left to fold
