"""Per-op FLOPs / bytes / launch counts for one training or inference step of each workload.

Runs the real model code on CPU under bf16 autocast (so the dtypes and the autocast casts are real), records every
aten op that is not a view, and classifies it. A small window count is used and FLOPs/bytes are scaled linearly to the
real window count (every op in the encoder is batched over windows, so the op count does not change).
"""
import sys, json, collections
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry
from torch.utils._pytree import tree_flatten

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src")
torch.manual_seed(0)
torch.set_num_threads(8)


def tbytes(x):
    return x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0


def sdpa_flops(args, fwd):
    q, k = args[0], args[1]
    B, h, Lq, d = q.shape
    Lk = k.shape[-2]
    f = 4 * B * h * Lq * Lk * d
    return f if fwd else int(2.5 * f)


class Rec(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func.is_view or func._overloadpacket.__name__ in ("detach", "lift_fresh", "_local_scalar_dense"):
            return out
        name = func._overloadpacket.__name__
        ins = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        flops = 0
        pk = func._overloadpacket
        if "scaled_dot_product" in name:
            flops = sdpa_flops(args, fwd="backward" not in name)
        elif pk in flop_registry:
            try:
                flops = flop_registry[pk](*args, **kwargs, out_val=out)
            except Exception:
                flops = 0
        b_in = sum(tbytes(t) for t in ins)
        b_out = sum(tbytes(t) for t in outs)
        cat = "gemm" if flops else ("cast" if name in ("_to_copy", "copy_") else "mem")
        self.rows.append((name, cat, flops, b_in + b_out, str(outs[0].dtype) if outs else ""))
        return out


def run(tag, step_fn, scale):
    rec = Rec()
    with rec:
        step_fn()
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for name, cat, f, b, dt in rec.rows:
        a = agg[(cat, name)]
        a[0] += 1; a[1] += f * scale; a[2] += b * scale
    return {"tag": tag, "ops": [[c, n, v[0], v[1], v[2]] for (c, n), v in agg.items()]}


# ------------------------------------------------------------------ workloads
from knee.model.knee_vit import KneeViT
from knee.model.trackA import TrackAModel
from knee.model.slice_vit import SliceViT
from knee.ssl.mae import MAE
import yaml

ac = lambda: torch.autocast("cpu", dtype=torch.bfloat16)


def bag(L, px):
    z = torch.zeros(1, L)
    zl = torch.zeros(1, L, dtype=torch.long)
    return {"x": torch.randint(0, 255, (1, L, 3, px, px), dtype=torch.uint8), "window_mask": torch.ones(1, L, dtype=torch.bool),
            "z": torch.rand(1, L), "z_mm": z, "plane": zl, "fluid": zl, "contrast": zl, "field": zl, "slot": zl,
            "series_idx": zl}


results = []
which = sys.argv[1:] or ["B_train", "B_infer", "MAE", "A_train", "A_infer"]

if "B_train" in which or "B_infer" in which:
    cfgB = yaml.safe_load(open(f"{REPO}/configs/trackB_vits_mil.yaml"))["model"]["cfg"]
    mB = KneeViT(cfgB)
    Ltr, Lev, n = 44, 88, 4          # 5.5 series × 8 train / × 16 eval windows; run 4, scale
    if "B_train" in which:
        mB.train()
        def stepB():
            with ac():
                lg = mB(bag(n, 256))
            lg.float().sum().backward()
        results.append(run("TrackB train microbatch (1 study, 44 win @256, ckpt on)", stepB, Ltr / n))
    if "B_infer" in which:
        mB.eval()
        def infB():
            with torch.no_grad(), ac():
                mB(bag(n, 256))
        results.append(run("TrackB inference (1 study, 88 win @256)", infB, Lev / n))

if "MAE" in which:
    c = yaml.safe_load(open(f"{REPO}/configs/mae_vits.yaml"))
    mc = dict(c["model"]); mc.pop("grad_checkpointing", None)
    enc = SliceViT(**mc, grad_checkpointing=False)
    mae = MAE(enc, **c["mae"]).train()
    n = 8
    def stepM():
        with ac():
            out = mae(torch.randint(0, 255, (n, 3, 256, 256), dtype=torch.uint8))
        out["loss"].backward()
    results.append(run("MAE step (128 windows @256, mask .75)", stepM, 128 / n))

if "A_train" in which or "A_infer" in which:
    cfgA = yaml.safe_load(open(f"{REPO}/configs/trackA_coatnet.yaml"))["model"]["cfg"]
    cfgA["pretrained"] = False
    mA = TrackAModel(cfgA)
    n = 2
    def bA(L):
        b = bag(L, 384); return b
    if "A_train" in which:
        mA.train()
        def stepA():
            with ac():
                lg = mA(bA(n))
            lg.float().sum().backward()
        results.append(run("TrackA train microbatch (1 study, 15 win @384, ckpt on)", stepA, 15 / n))
    if "A_infer" in which:
        mA.eval()
        def infA():
            with torch.no_grad(), ac():
                mA(bA(n))
        results.append(run("TrackA inference (1 study, 60 win @384)", infA, 60 / n))

OUT = Path(__import__("os").environ.get("PERF_OUT", "."))
json.dump(results, open(OUT / f"ops_{'_'.join(which)}.json", "w"))
print("done", [r["tag"] for r in results])
