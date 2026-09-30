"""Performance plan §6 on a Kaggle T4 (private GPU script kernel, internet off): measure what
docs/plan/03_performance_plan.md §2 estimated. Synthetic inputs only (no competition data is read), models built from
`knee-src` with random weights (timing doesn't depend on the weights).

Per workload and variant: median step time (CUDA events), peak VRAM, and a torch.profiler breakdown of CUDA kernel
time by category (gemm/conv, attention, softmax, norm, cast, elementwise, other) plus the top kernels by name. OOM is
recorded, not fatal. Variants settle plan items C1 (grad checkpointing off), B2 (inference folds), B3 (channels_last),
B4 (torch.compile on sm75) and B5 (which SDPA kernel CoAtNet's biased attention uses).

Outputs in /kaggle/working/profile/: results.json (everything) and summary.md (the table for the plan doc).

Push (Dom's OK given 2026-09-30): kaggle kernels push -p notebooks/kaggle_profile --accelerator NvidiaTeslaT4
CPU smoke test (tiny sizes, no CUDA): PROFILE_SMOKE=1 KNEE_SRC=src python notebooks/kaggle_profile/gpu_profile.py
"""
import collections
import copy
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

SMOKE = os.environ.get("PROFILE_SMOKE") == "1"
WORK = Path("/kaggle/working/profile") if not SMOKE else Path(os.environ.get("PERF_OUT", ".")) / "profile_smoke"
WORK.mkdir(parents=True, exist_ok=True)


def find_src() -> Path:
    if os.environ.get("KNEE_SRC"):
        return Path(os.environ["KNEE_SRC"])
    pats = ["/kaggle/input/datasets/*/knee-src", "/kaggle/input/datasets/*/knee-src/*", "/kaggle/input/knee-src",
            "/kaggle/input/knee-src/*"]
    for p in pats:
        for d in sorted(glob.glob(p)):
            if (Path(d) / "knee/__init__.py").exists():
                return Path(d)
    raise FileNotFoundError("knee-src not found")


SRC = find_src()
sys.path.insert(0, str(SRC))

from knee.model.knee_vit import KneeViT  # noqa: E402
from knee.model.slice_vit import SliceViT  # noqa: E402
from knee.model.fuse import prepare_for_inference  # noqa: E402
from knee.ssl.mae import MAE  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16  # the T4 has no bf16 tensor cores (resolve_precision → "fp16")

# ---- configs (copied from configs/*.yaml at 519ca33; the kernel only has src/knee, not configs/)
CFG_B = {"slice_vit": {"img_size": 256, "patch": 16, "dim": 384, "depth": 12, "heads": 6, "tokenizer": "conv_stem",
                       "stem_norm": "gn", "pos2d": "sincos", "pool": "gap", "qk_norm": True, "layerscale": 0.1,
                       "drop_path": 0.1, "attn_impl": "sdpa", "grad_checkpointing": True},
         "slice_pos": {"n_freq": 8, "use_mm": False},
         "series_type": {"fields": ["plane", "fluid", "contrast", "field"], "max_series": 0},
         "head": {"kind": "attn_mil", "att_dim": 256, "dropout": 0.2, "gated": False},
         "window_chunk": 64, "window_dropout": 0.1}
CFG_A = {"backbone": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "pretrained": False,
         "head": {"kind": "attn_mil", "att_dim": 256, "dropout": 0.2, "gated": False},
         "grad_checkpointing": True, "window_chunk": 16, "window_dropout": 0.0, "slot_embed": True, "slice_pos": False}
CFG_MAE_ENC = {"img_size": 256, "patch": 16, "dim": 384, "depth": 12, "heads": 6, "tokenizer": "conv_stem",
               "stem_norm": "gn", "pos2d": "sincos", "pool": "gap", "qk_norm": True, "layerscale": 0.1,
               "drop_path": 0.0, "attn_impl": "sdpa", "grad_checkpointing": False}
CFG_MAE = {"mask_ratio": 0.75, "dec_dim": 256, "dec_depth": 4, "dec_heads": 8, "norm_pix": True}

# Real sizes (plan §2) vs smoke sizes
if SMOKE:
    CFG_A["backbone"] = "coatnet_rmlp_nano_rw_224"
    SZ = {"A_px": 224, "A_tr": 2, "A_ev": 3, "B_px": 64, "B_tr": 3, "B_ev": 4, "MAE_n": 4, "MAE_px": 64}
    CFG_B["slice_vit"].update(img_size=64, depth=2)
    CFG_MAE_ENC.update(img_size=64, depth=2)
    WARM, ITERS, PROF = 1, 2, 1
else:
    SZ = {"A_px": 384, "A_tr": 15, "A_ev": 60, "B_px": 256, "B_tr": 44, "B_ev": 88, "MAE_n": 128, "MAE_px": 256}
    WARM, ITERS, PROF = 3, 10, 3

T0 = time.time()
RESULTS = {"env": {}, "runs": []}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def bag(L: int, px: int, n_slots: int = 5) -> dict:
    zl = torch.zeros(1, L, dtype=torch.long, device=DEV)
    return {"x": torch.randint(0, 255, (1, L, 3, px, px), dtype=torch.uint8, device=DEV),
            "window_mask": torch.ones(1, L, dtype=torch.bool, device=DEV),
            "z": torch.rand(1, L, device=DEV), "z_mm": torch.zeros(1, L, device=DEV),
            "plane": zl, "fluid": zl, "contrast": zl, "field": zl, "series_idx": zl,
            "slot": (torch.arange(L, device=DEV) % n_slots)[None]}


def amp():
    return torch.autocast(DEV.type, dtype=AMP_DTYPE, enabled=DEV.type == "cuda")


# ---- kernel categories (names are CUDA kernel names from the profiler)
CATS = [("attention", r"fmha|flash|attention|mem_eff|efficient_attention"),
        ("softmax", r"softmax"),
        ("gemm_conv", r"gemm|cutlass|xmma|s1688|h1688|s884|h884|turing_|volta_|conv|implicit|wgrad|dgrad|cudnn|winograd"),
        ("norm", r"layer_norm|layernorm|batch_norm|batchnorm|group_norm|welford|bn_fw|bn_bw|native_norm"),
        ("cast_copy", r"copy|cast|convert|to_copy|fill"),
        ("elementwise", r"elementwise|vectorized|reduce|foreach|index|scatter|gather|cat")]


def categorize(name: str) -> str:
    n = name.lower()
    for cat, pat in CATS:
        if re.search(pat, n):
            return cat
    return "other"


def kernel_breakdown(prof) -> dict:
    by_name = collections.defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            t = getattr(e, "device_time", None)
            if t is None:
                t = getattr(e, "cuda_time", 0.0)
            by_name[e.name][0] += 1
            by_name[e.name][1] += t / 1000.0  # µs → ms
    cats = collections.defaultdict(float)
    for n, (_, ms) in by_name.items():
        cats[categorize(n)] += ms
    total = sum(cats.values())
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:40]
    return {"total_ms": total, "cats_ms": dict(cats), "n_kernels": sum(v[0] for v in by_name.values()),
            "top": [[n[:160], c, round(ms, 3), categorize(n)] for n, (c, ms) in top]}


def op_table(prof) -> str:
    for key in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
        try:
            return prof.key_averages().table(sort_by=key, row_limit=30, max_name_column_width=60)
        except Exception:  # noqa: BLE001
            continue
    return ""


def bench(name: str, variant: str, step, per_iter_div: int = 1, extra: dict | None = None) -> dict:
    """step() runs one step. Times ITERS steps after WARM warm-ups, then profiles PROF more."""
    rec = {"workload": name, "variant": variant, **(extra or {})}
    log(f"== {name} [{variant}]")
    try:
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_first = time.time()
        step()
        sync()
        rec["first_step_s"] = round(time.time() - t_first, 2)  # includes cudnn autotune / compile
        for _ in range(WARM):
            step()
        sync()
        times = []
        for _ in range(ITERS):
            if DEV.type == "cuda":
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); step(); b.record(); b.synchronize()
                times.append(a.elapsed_time(b))
            else:
                t = time.perf_counter(); step(); times.append((time.perf_counter() - t) * 1000)
        rec["ms_median"] = round(statistics.median(times) / per_iter_div, 2)
        rec["ms_min"] = round(min(times) / per_iter_div, 2)
        rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEV.type == "cuda" else None
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if DEV.type == "cuda" else [])
        with torch.profiler.profile(activities=acts) as prof:
            for _ in range(PROF):
                step()
            sync()
        kb = kernel_breakdown(prof)
        kb["total_ms"] = round(kb["total_ms"] / PROF / per_iter_div, 2)
        kb["cats_ms"] = {k: round(v / PROF / per_iter_div, 2) for k, v in kb["cats_ms"].items()}
        rec["kernels"] = kb
        # CPU-side dispatch: wall time of the profiled steps vs GPU busy time (GPU idle ⇒ launch/host bound)
        rec["gpu_busy_frac"] = round(kb["total_ms"] / max(rec["ms_median"], 1e-6), 3) if DEV.type == "cuda" else None
        (WORK / f"ops_{name}_{variant}.txt").write_text(op_table(prof))
        rec["status"] = "ok"
        log(f"   {rec['ms_median']} ms, peak {rec['peak_vram_gb']} GB, gpu busy {rec['gpu_busy_frac']}, "
            f"cats {kb['cats_ms']}")
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "OOM"
        rec["error"] = str(e).splitlines()[0][:300]
        log("   OOM")
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
        rec["traceback"] = traceback.format_exc()[-2000:]
        log("   ERROR", rec["error"])
    finally:
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    RESULTS["runs"].append(rec)
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=1))
    return rec


def logits_of(out):
    return out[0] if isinstance(out, tuple) else out


def train_step_fn(model, batch, scaler):
    """One micro-batch: fwd + bwd under fp16 autocast with loss scaling (what train_sup does on the T4). The
    optimizer step runs once per `accum` (24 / 16) micro-batches, so it's timed separately."""
    tgt = torch.rand(1, 12, device=DEV)
    lossf = torch.nn.BCEWithLogitsLoss()

    def step():
        with amp():
            lg = logits_of(model(batch))
        loss = lossf(lg.float(), tgt)
        scaler.scale(loss).backward()
    return step


def opt_step_fn(model, scaler):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    def step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    return step


def free(*objs):
    for o in objs:
        del o
    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def scaler_():
    return torch.amp.GradScaler(DEV.type, enabled=DEV.type == "cuda")


def has_bn_after_conv(m) -> bool:
    return any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())


# =====================================================================================================================
def env_info():
    e = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    try:
        import timm
        e["timm"] = timm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton
        e["triton"] = triton.__version__
    except Exception as ex:  # noqa: BLE001
        e["triton"] = f"missing ({type(ex).__name__})"
    if DEV.type == "cuda":
        e["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        e["capability"] = list(torch.cuda.get_device_capability(0))
        e["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        e["bf16_supported_reported"] = torch.cuda.is_bf16_supported()
        e["sdpa_backends_enabled"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                      "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                      "math": torch.backends.cuda.math_sdp_enabled()}
        try:
            e["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.sm,power.limit,"
                                              "temperature.gpu", "--format=csv"], capture_output=True, text=True,
                                             timeout=20).stdout
        except Exception:  # noqa: BLE001
            pass
    e["cpu_count"] = os.cpu_count()
    RESULTS["env"] = e
    log(json.dumps(e, indent=1))


def run_trackB():
    for ckpt in (True, False):
        cfg = copy.deepcopy(CFG_B)
        cfg["slice_vit"]["grad_checkpointing"] = ckpt
        m = KneeViT(cfg).to(DEV).train()
        sc = scaler_()
        b = bag(SZ["B_tr"], SZ["B_px"])
        bench("B_train", f"ckpt_{'on' if ckpt else 'off'}", train_step_fn(m, b, sc))
        if ckpt:
            bench("B_optstep", "adamw", opt_step_fn(m, sc))
        free(m, b)

    b = bag(SZ["B_ev"], SZ["B_px"])
    m = KneeViT(CFG_B).to(DEV).eval()

    def inf(model):
        def step():
            with torch.no_grad(), amp():
                model(b)
        return step
    bench("B_infer", "autocast", inf(m))
    folded = prepare_for_inference(m)
    bench("B_infer", "autocast+folded", inf(m), extra={"folded": folded})
    # B4: torch.compile of the per-window encoder only (the bag masking outside it is data-dependent)
    if DEV.type == "cuda" or SMOKE:
        try:
            m.encoder = torch.compile(m.encoder)
            bench("B_infer", "autocast+folded+compile", inf(m))
        except Exception as e:  # noqa: BLE001
            RESULTS["runs"].append({"workload": "B_infer", "variant": "compile", "status": "error", "error": str(e)[:600]})
    free(m, b)


def run_mae():
    enc = SliceViT(**CFG_MAE_ENC)
    mae = MAE(enc, **CFG_MAE).to(DEV).train()
    sc = scaler_()
    x = torch.randint(0, 255, (SZ["MAE_n"], 3, SZ["MAE_px"], SZ["MAE_px"]), dtype=torch.uint8, device=DEV)
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)

    def step():  # MAE runs accum 1, so the optimizer step is part of every step
        with amp():
            out = mae(x)
        sc.scale(out["loss"]).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        sc.step(opt)
        sc.update()
        opt.zero_grad(set_to_none=True)
    bench("MAE", f"bs{SZ['MAE_n']}", step)
    free(mae, x, opt)


def run_trackA():
    from knee.model.trackA import TrackAModel
    for ckpt, cl in ((True, False), (False, False), (True, True), (False, True)):
        cfg = copy.deepcopy(CFG_A)
        cfg["grad_checkpointing"] = ckpt
        m = TrackAModel(cfg).to(DEV).train()
        if cl:
            m = m.to(memory_format=torch.channels_last)
        sc = scaler_()
        b = bag(SZ["A_tr"], SZ["A_px"])
        bench("A_train", f"ckpt_{'on' if ckpt else 'off'}{'+cl' if cl else ''}", train_step_fn(m, b, sc))
        if ckpt and not cl:
            bench("A_optstep", "adamw", opt_step_fn(m, sc))
        free(m, b)

    b = bag(SZ["A_ev"], SZ["A_px"])
    cfg = copy.deepcopy(CFG_A)
    cfg["grad_checkpointing"] = False  # load_arm turns it off for inference

    def inf(model):
        def step():
            with torch.no_grad(), amp():
                model(b)
        return step
    m = TrackAModel(cfg).to(DEV).eval()
    bench("A_infer", "autocast", inf(m))
    folded = prepare_for_inference(m)
    bench("A_infer", "autocast+folded", inf(m), extra={"folded": folded})
    m = m.to(memory_format=torch.channels_last)
    bench("A_infer", "autocast+folded+cl", inf(m))
    if DEV.type == "cuda" and time.time() - T0 < 45 * 60:
        try:
            m.backbone = torch.compile(m.backbone)
            bench("A_infer", "autocast+folded+cl+compile", inf(m))
        except Exception as e:  # noqa: BLE001
            RESULTS["runs"].append({"workload": "A_infer", "variant": "compile", "status": "error", "error": str(e)[:600]})
    free(m, b)


def write_summary():
    lines = ["| workload | variant | status | ms (median) | peak VRAM GB | GPU busy | gemm_conv | attention | softmax | "
             "norm | cast_copy | elementwise | other | first step s |", "|" + "---|" * 14]
    for r in RESULTS["runs"]:
        k = r.get("kernels", {}).get("cats_ms", {})
        lines.append("| " + " | ".join(str(x) for x in [
            r["workload"], r["variant"], r["status"], r.get("ms_median", ""), r.get("peak_vram_gb", ""),
            r.get("gpu_busy_frac", ""), *[k.get(c, "") for c in ("gemm_conv", "attention", "softmax", "norm",
                                                                  "cast_copy", "elementwise", "other")],
            r.get("first_step_s", "")]) + " |")
    attn = collections.Counter()
    for r in RESULTS["runs"]:
        if r["workload"].startswith("A_"):
            for n, c, ms, cat in r.get("kernels", {}).get("top", []):
                if cat in ("attention", "softmax") or "bmm" in n.lower():
                    attn[n[:120]] += ms
    lines += ["", "Track A attention-related kernels (ms summed over its runs):"]
    lines += [f"- {ms:.1f} ms `{n}`" for n, ms in attn.most_common(12)]
    (WORK / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    torch.manual_seed(0)
    env_info()
    for fn in (run_trackB, run_mae, run_trackA):
        # as in the trainers: cudnn autotuning only for MAE (fixed shapes); train_sup leaves it off (bag sizes vary)
        torch.backends.cudnn.benchmark = fn is run_mae
        try:
            fn()
        except Exception:  # noqa: BLE001
            log("workload failed:\n" + traceback.format_exc())
        write_summary()
    RESULTS["total_s"] = round(time.time() - T0, 1)
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=1))
    log("done")
