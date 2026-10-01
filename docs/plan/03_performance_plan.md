# 03 — Performance plan: compute vs memory bandwidth vs overhead

Written 2026-09-30 (Mac, no GPU). Question from Dom: *which is worse for us, overhead or memory bandwidth, and what do we do
about each before we spend GPU hours?* This doc answers it with an op-level model of every workload, lists the fixes
ranked by payoff, and says what still needs a real GPU measurement.

## 0. Short answer

| Where the time goes | Verdict | Evidence |
|---|---|---|
| **Kernel-launch / Python overhead** (the GPU waiting on the CPU) | **Not the problem.** 4–34 ms of CPU dispatch per step vs 67–1,300 ms of GPU work | §2 |
| **GPU memory bandwidth** (elementwise, norm, softmax, casts, copies) | **The biggest share on the T4 (60–78% incl. casts)**, about equal to compute on the 5070 | §2 |
| **Compute** (GEMMs, convs, attention) | 23–56%. **Not exhausted**: gradient checkpointing, window counts and resolution are real compute levers | §2, §4 |
| **Data movement outside the model** (disk/NFS reads, CPU decode/resize) | **Dominates end-to-end inference** (the efficiency score) and very likely MAE pretraining | §3 |

So Dom's instinct is half right. Bandwidth is worth attacking, but "overhead" in the kernel-launch sense isn't. The
overhead that matters is **outside the model**, in I/O and the data loader. Compute isn't exhausted either.

## 1. Method (and its limits)

`scripts/perf/opcount.py` runs the real model code (configs as committed) on CPU under bf16 autocast and records every
non-view aten op: FLOPs (PyTorch's flop registry; SDPA by formula), bytes in + out, and dtype. It runs a few windows and
scales linearly to the real window count, which is exact for FLOPs/bytes because the encoders batch over windows.
`scripts/perf/roofline.py` turns that into a time estimate per op: `max(FLOPs / (0.5 × peak), bytes / (0.75 × bandwidth))`,
plus 10 µs of CPU dispatch per op.

    PERF_OUT=<dir> python scripts/perf/opcount.py [B_train B_infer MAE A_train A_infer]
    PERF_OUT=<dir> python scripts/perf/roofline.py

Hardware: T4 = 65 TFLOPS fp16 tensor, 320 GB/s. RTX 5070 = 124 TFLOPS dense FP16 per NVIDIA's whitepaper, which we assume
is the fp16-accumulate rate, so **62 TFLOPS** with fp32 accumulate (what PyTorch uses; UNVERIFIED), 672 GB/s, 48 MB L2.

**Limits. Treat the numbers as ±2×; the ranking is what matters:**
- CPU autocast keeps LayerNorm in bf16, while **CUDA autocast runs LayerNorm/softmax in fp32** (checked on CPU; CUDA list per
  the PyTorch AMP docs). On the GPU the bandwidth share is therefore *higher* than shown.
- Bytes assume every op goes to DRAM (no L2 reuse). That overstates the 5070 most (48 MB L2).
- cuDNN/SDPA kernels on CUDA differ from the CPU ones (e.g. attention with a bias, see B5).
- The 50%/75% efficiency factors are assumptions.

→ **One real `torch.profiler` run on a T4 replaced these estimates with measurements: see §2b** (2026-09-30).

## 2. Results per step (estimates)

| Workload (config as committed) | aten ops | TFLOP | GB moved | FLOP/B | T4 est. | T4 split: compute / bandwidth / casts | 5070 est. | CPU dispatch |
|---|---|---|---|---|---|---|---|---|
| Track A train microbatch (1 study, 15 win @384, ckpt) | 3,384 | 10.2 | 269 | 38 | **1,307 ms** | 23% / 67% / 11% | 792 ms | 34 ms |
| Track A inference (60 win @384) | 859 | 5.7 | 144 | 40 | **626 ms** | 25% / 64% / 11% | 393 ms | 9 ms |
| Track B train microbatch (44 win @256, ckpt) | 1,470 | 1.7 | 33 | 53 | 154 ms | 33% / 50% / 17% | 104 ms | 15 ms |
| Track B inference (88 win @256) | 385 | 1.2 | 20 | 57 | 97 ms | 33% / 48% / 19% | 67 ms | 4 ms |
| MAE step (128 win @256, mask 0.75) | 1,182 | 2.3 | 44 | 52 | 201 ms | 33% / 53% / 14% | 135 ms | 12 ms |

The ridge point (FLOP/B where a kernel stops being bandwidth-bound) is ~200 on the T4 and ~90 on the 5070. Every workload
averages 38–57 FLOP/B, well below both. The GEMMs themselves sit above the ridge. Everything around them is far below it.

**What that means in GPU hours (T4, estimates):**

| Job | Estimate | Note |
|---|---|---|
| Track A, 1 fold (16 epochs × 3,525 studies + dense eval) | **~23 h** | 5 folds ≈ 115 h > the ~90 h of Kaggle quota left before Oct 22. **Optimizing isn't optional here** |
| Track B, 1 fold (20 epochs) | ~3.5 h | |
| MAE, 40 epochs × 97k windows | **~1.7 h of GPU** | *if the loader keeps up* (it very likely won't, §3). The "multi-day" MAE plan was a guess; the GPU side is cheap |

Top single items on the T4 (Track A training): `convolution_backward` 224 ms (compute), attention `bmm` + softmax +
softmax-backward ≈ 290 ms (**materialized L×L scores**, see B5), `_to_copy` (autocast casts) 142 ms, `add`/`mul`/`clone`
≈ 240 ms (bandwidth), BatchNorm + SiLU ≈ 75 ms.

### 2b. Measured on a Kaggle T4 (2026-09-30, `notebooks/kaggle_profile`, ~7 min of GPU)

Synthetic inputs, random weights, the committed configs, fp16 autocast + GradScaler (what `precision: auto` picks on
the T4), batch 1 study. torch 2.10.0+cu128, cuDNN 9.10, timm 1.0.26, Triton 3.6.0; 2× Tesla T4 (15.6 GB), only GPU 0
used. Median of 10 steps after 3 warm-ups; kernel split from `torch.profiler` over 3 more steps. Raw: kernel output
`results.json` (not committed; re-run the kernel to regenerate). "Train" = forward + backward of one micro-batch; the
optimizer step runs once per `accum` micro-batches and is listed separately.

| Workload | Variant | ms / step | est. (§2) | Peak VRAM GB | gemm+conv | attention | norm | casts/copies | elementwise |
|---|---|---|---|---|---|---|---|---|---|
| Track A train (15 win @384) | ckpt on (as committed) | **971** | 1,307 | 3.9 | 40% | 16% | 2% | 19% | 18% |
| | ckpt off | 747 (−23%) | | 11.3 | | | | | |
| | ckpt on + channels_last | 921 (−5%) | | 3.9 | | | | | |
| | **ckpt off + channels_last** | **704 (−28%)** | | 11.3 | | | | | |
| Track A optimizer step | AdamW + clip + scaler | 37 | | | | | | | |
| Track A inference (60 win @384) | autocast | **1,067** | 626 | 2.0 | 41% | 15% | 2% | 24% | 17% |
| | + conv/BN fold | 1,136 (**+6%, slower**) | | 1.5 | | | | | |
| | + fold + channels_last | 1,035 (−3%) | | 1.5 | | | | | |
| | **+ fold + cl + `torch.compile`** | **673 (−37%)** | | 1.3 | | | | | |
| Track B train (44 win @256) | ckpt on (as committed) | **288** | 154 | 1.3 | 22% | 8% | 17% | 32% | 15% |
| | ckpt off | 235 (−18%) | | 3.7 | | | | | |
| Track B optimizer step | AdamW + clip + scaler | 14 | | | | | | | |
| Track B inference (88 win @256) | autocast | **185** | 97 | 0.7 | 24% | 5% | 24% | 33% | 12% |
| | + LayerScale fold | 181 (−2%) | | 0.7 | | | | | |
| | **+ fold + `torch.compile`** | **96 (−48%)** | | 0.5 | | | | | |
| MAE step (128 win @256, incl. optimizer) | as committed | **406** | 201 | 6.7 | 24% | 7% | 18% | 27% | 14% |

Kernel shares are of GPU kernel time for the first variant of each workload. Under `torch.compile` most kernels are
fused Triton kernels, so the category split isn't meaningful there and is omitted.

**What the measurements say:**
1. **Launch overhead is confirmed not to matter.** Summed kernel time ≈ wall time in every eager run (GPU busy 99–102%),
   and "Command Buffer Full" takes ~41% of CPU time: the CPU is ahead and waiting for queue space. The GPU never idles.
2. **Bandwidth-bound work is an even bigger share than estimated.** GEMM + conv + attention is only 29–56% of kernel
   time. In Track B, LayerNorm (fp32 under CUDA autocast, as predicted in §1) plus casts/copies is 58% of inference.
   That's why the ViTs came in ~2× slower than estimated while Track A's convs came in faster.
3. **`torch.compile` works on the T4** (Triton 3.6 on sm75; B4 was UNVERIFIED) and is the largest single win: −37% on
   Track A inference, −48% on Track B. The one-time compile costs 49 s (B) and 86 s (A). Caveat: measured at fixed
   shapes. Real bags vary per study, and `encode_windows` splits into `window_chunk` pieces with a ragged last chunk,
   so the notebook must pad to a fixed chunk size (or accept one dynamic-shape recompile). Not yet tested.
4. **CoAtNet's biased attention already uses the memory-efficient kernel** (`fmha_cutlassF/B_f16_..._sm75`, forward
   and backward; no FlashAttention on sm75). The materialized L×L path seen on CPU doesn't happen on CUDA. **B5 is
   closed: nothing to do.** Attention is 15–16% of Track A.
5. **Gradient checkpointing off** (C1): −23% Track A, −18% Track B. Track A then peaks at 11.3 GB: fits the T4 (15.6 GB),
   too tight for the 5070 (12 GB) without a smaller `window_chunk`.
6. **channels_last** (B3): −5% training, −3% inference on top of the fold. Small but free for training.
7. **The conv/BN fold (B2) made Track A slower in eager mode.** cuDNN adds the new conv bias in a separate broadcast
   `add_` (+360 ms), which costs more than the fused BN-inference kernel it replaces (−345 ms). It's now off by
   default (`prepare_for_inference(conv_bn=False)`). The LayerScale fold gave −2% on Track B and stays on.

**Revised GPU-hour budget (T4, from the measured rows):**

| Job | As committed | With ckpt off + cl, compiled eval |
|---|---|---|
| Track A, 1 fold (16 ep × 3,525 train + 16 × 882 eval) | 15.2 h train + 4.2 h eval ≈ **19 h** | 11.0 + 2.6 ≈ **14 h** |
| Track B, 1 fold (20 ep) | 5.6 + 0.9 ≈ **6.5 h** | 4.6 + 0.5 ≈ **5 h** |
| MAE, 40 ep × 97k windows (315 windows/s) | **3.4 h** of GPU, if the loader delivers 315 windows/s | |
| Inference, 1,300 studies, model only | Track A 23 min, Track B 4 min | A 15 min (+1.5 min compile), B 2 min |

Track A × 5 folds at ~14 h = 70 h still eats most of the ~90 h of quota left before Oct 22. Two levers not tried
here: each Kaggle GPU session has **two T4s**, so two folds could train at once (quota is billed per session, not per GPU:
411 s on T4 ×2 cost 0.12 h; P04 addendum), and evaluating every 2nd epoch halves the eval share. **Coded 2026-09-30 (P04):**
`knee.train.multi_fold` (one fold per GPU) and `predict.predict_dirs_multi` (inference sharded over both T4s).

## 3. Outside the model: I/O and the data loader

**Inference (the efficiency score).** Test ≈ 1,300 studies × ~186 files ≈ 240k DICOMs ≈ 170 GB (scaled from train; UNVERIFIED
for test). Measured NFS throughput (`docs/kaggle_env.md` §4): 173 slices/s at 4 workers, 325/s at 16.
`notebooks/infer.py` uses `cpu_count − 1` = **3 worker processes** on a 4-vCPU notebook:

| | minutes for 1,300 studies |
|---|---|
| Reading at 3–4 workers (~170 slices/s) | **~24 min** |
| Reading at 16 in flight (~325 slices/s) | ~12 min |
| Track A model (60 win, T4) | ~14 min |
| Track B model (88 win, T4) | ~2 min |

Efficiency score = `AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`. With maxAUC ≈ 0.96, **1 minute of runtime is
worth ≈ 0.00085 AUC**, so 12 minutes saved in I/O ≈ +0.01 AUC on the efficiency formula. For the efficiency submission,
**I/O is the first thing to fix, then model size.**

**MAE loader.** At the estimated GPU speed the loader must deliver ~640 windows/s (T4) or ~950 (5070). Each window is a
3-slice read at 384 px (442 KB), a random resized crop and a resize on the CPU, plus Python per item. That's **280–420 MB/s
of reads** and hundreds of cv2 calls per second on 4 vCPUs (Kaggle) or from a SATA QVO (Windows). The GPU will wait on this.

**Track A/B supervised loaders** are fine: 7–19 MB per study at 1–10 studies/s. Track B's 384→256 resize on the CPU costs
about 0.2 cores.

## 4. The optimization menu (ranked by payoff ÷ risk)

Tags: **[exact]** = same outputs up to float rounding, no revalidation needed beyond a parity test. **[val]** = changes
numerics or the model; needs an OOF check. **[arch]** = depends on the Oct 8 gate; do it after the architecture is final.

### A. Data movement outside the model (largest end-to-end win)
| # | Change | Payoff | Tag |
|---|---|---|---|
| A1 | **Inference I/O concurrency**: read files with a thread pool (16–32 in flight; reads release the GIL), decode/preprocess in the 3–4 processes. Today: 3 processes do read + decode serially | ~−12 min per submission (est.) | [exact] |
| A2 | **Header-first, pixels-on-demand**: read every header (`stop_before_pixels`), pick series/slices, then read pixel data only for the slices the windows use | Big for the efficiency submission (few windows). ~0 for the dense-window accuracy submission. Depends on whether NFS is latency- or bandwidth-bound (measure) | [exact] |
| A3 | **MAE: several windows per series read** (read the series slab once, cut K windows) + **one batch per `__getitem__`** (a batch sampler), which removes per-item Python/rng/collate cost | Several × loader throughput | [exact] (same distribution if K is chosen per epoch) |
| A4 | **MAE: random-resized-crop on the GPU** (one batched `F.grid_sample` / `roi_align` over the uint8 384-px windows). Workers then only memcpy | Removes the CPU resize bottleneck; PCIe cost ≈ 57 MB per batch (~5 ms) | [val] (interpolation differs slightly from cv2) |
| A5 | **256-px cache for Track B/MAE** (≈54 GB; fits Kaggle as 3 datasets) | 2.25× fewer bytes, no CPU resize | [val] (resize once vs at load: identical if the same INTER_AREA) |

### B. GPU memory bandwidth inside the model
| # | Change | Payoff (T4 est.) | Tag |
|---|---|---|---|
| B1 | **Inference in pure half** (`model.half()` + half inputs) instead of autocast: removes the cast kernels and halves LN/residual traffic. `run_arm` already has the fp32 retry | casts are 11–19% of GPU time; more on CUDA (LN in fp32 under autocast) | [val] (fp16 numerics; compare OOF) |
| B2 | **Fold LayerScale into `attn.proj` / `mlp.fc2` at inference** (γ·W, γ·b); fold BatchNorm into the preceding conv where BN follows a conv | 1–2 elementwise passes per block; BN+SiLU ≈ 75 ms of 626 in Track A inference | [exact] |
| B3 | **`channels_last`** for Track A (NHWC is cuDNN's tensor-core layout; removes the `clone`/permute copies between conv and attention stages) | usually 10–30% for conv nets on tensor cores (to measure) | [exact] |
| B4 | **`torch.compile`** on Linux/Kaggle: fuses LN→cast, bias+GELU, residual+LayerScale+DropPath into single kernels. That's exactly the bandwidth-bound share | typically the largest generic win for this profile. T4 = sm75: Triton support **UNVERIFIED**; Windows needs a Triton port. Keep it optional (the plan says nothing may depend on it) | [exact] |
| B5 | **Track A attention with the relative-position bias**: timm calls SDPA with a learned `attn_mask`. On CPU that fell back to materialized L×L scores (`bmm` + softmax ≈ 22% of the T4 estimate). Check on CUDA that the memory-efficient kernel handles the bias gradient; if not, precompute the bias once per step (it's shared) or accept it | up to ~−290 ms per microbatch | [exact] |
| B6 | Track B's hand-written ViT, **manual fusions if compile is unavailable**: one `addcmul` for residual+LayerScale, `F.layer_norm` + a Linear with the norm affine folded in at inference, QK-norm kept (stability) | small–medium | [exact] |

### C. Compute (not exhausted yet)
| # | Change | Payoff | Tag |
|---|---|---|---|
| C1 | **Turn gradient checkpointing off, or make it selective** (only CoAtNet's high-res early stages) wherever VRAM allows. It recomputes the forward pass: ≈ +33% of step time. The T4 has 16 GB, more than the 5070 | up to −25% train time | [exact] |
| C2 | **Track A resolution / windows**: FLOPs ∝ pixels × windows. 384→320 px = −31%; 60→36 eval windows = −40% inference | large | [val] [arch] |
| C3 | fp16 training path on the T4 (autocast fp16 + `GradScaler`); **required** before any Kaggle training (bf16 isn't supported there) | enables Kaggle | [val] |
| C4 | ToMe / token pruning in Track B at inference (already planned for S10) | medium | [val] [arch] |
| C5 | INT8 on the T4 (130 TOPS) for the efficiency submission | medium, risky | [val] [arch] |

### D. Launch overhead and host syncs (lowest priority)
The estimates put CPU dispatch at 1–10% of GPU time, so the GPU never waits on Python. Cheap tidy-ups only: keep the
running loss on the GPU and call `.item()` / `isfinite` at log steps instead of every microbatch (each is a host sync that stops the
CPU queuing work ahead); `x[mask]` with a bool mask also syncs (fine at batch 1). **CUDA graphs are not worth it**: bag
lengths vary per study, so graphs would need padding to fixed shapes.

## 5. Order of work

**Status 2026-09-30:** step 1 is coded and CPU-tested (`tests/test_perf.py`, 13 tests; full suite 117 passed): C3 as
`precision: auto` + `GradScaler`, A1 as `dicom_io.read_slices`, A3 as `data.windows_per_read` (config 4), B2 as
`knee.model.fuse.prepare_for_inference` (called by `load_arm`), and D (GPU-side loss meter, vectorized rank loss,
branch-free mask helpers). Speed-ups are **not measured yet**: that needs the T4 profile (§6) and a Kaggle inference run.

1. **Now (architecture-independent, Mac-codable, CPU-testable):** C3 (fp16 + GradScaler), A1, A3, B2, D. All [exact] except
   C3, and C3 is required for Kaggle anyway.
2. ~~One real GPU profile~~ **Done (§2b).** Settled: C1 (off wherever VRAM allows), B3 (small win), B4 (works on the T4,
   biggest win; needs fixed chunk shapes), B5 (nothing to do). B2's conv/BN fold is off by default. Next: C1 + B3 in the
   Track A config for Kaggle training, compile with padded chunks in `infer.py`, then A1/A2 measured in a real inference run.
3. **Step 2 (coded 2026-09-30, CPU-tested, not yet measured on a GPU; Dom asked for it):** B4 in *training* and
   inference at fixed shapes, and B1. `knee.model.compiled`: a bag is split into **bucket** chunk sizes (powers of two
   from `min_bucket` to `window_chunk`; 44 → 32 + 8 + 4), and only a remainder below the smallest bucket is padded
   (with copies of real windows), so the encoder sees a handful of shapes and each compiles once (`dynamic=False`,
   in-place `nn.Module.compile`, so the state_dict keys don't change). Switches, all off by default: `perf.compile` /
   `perf.channels_last` in `train_sup`, `perf.compile` in `pretrain_mae`, and `perf.{half, compile, channels_last}` in
   `arms.json` for `load_arm`. Pure half runs without autocast; `run_arm` keeps the fp32 retry. **Exactness:** SliceViT
   bucketed = plain (values and gradients, tested); Track A is exact in eval, but in training BatchNorm's batch
   statistics come from different window groups → [val]. Half is [val]: feature error ~0.04–0.15% vs fp32 on CPU.
   **Measurement:** `notebooks/kaggle_profile` with `PROFILE_SET=step2` (the default now; cycles 3 bag sizes per
   workload, reports the compute share and graph counts, 30-min cap). Estimated at 20–30 min of T4 quota; **not pushed**
   (Dom, 2026-09-30: wait). Needs `knee-src` re-versioned first.
   **Item 3, bigger matmuls (coded 2026-09-30, CPU-tested):** `optim.effective_batch` derives `accum` from
   `data.batch_size`, so several studies per micro-batch keep the recipe (configs still at batch 1 until the VRAM table);
   `data.eval_batch_size` (4 in the configs, exact); inference `perf.studies_per_forward` (grouped forward, failed
   groups retried one by one) and `perf.window_chunk` (inference keeps no activations, so bigger chunks are free).
   Expected gain is modest (UNVERIFIED, 5–15%): the encoder already packs a step's windows into GEMMs of ~16k token
   rows; the limit is the 384 width, which batching doesn't change. What batching does buy: fewer small remainder
   chunks, per-step costs (head, Python, launches) amortized, and inference no longer one study per forward. The
   step-2 profile measures it (`+bs2/+bs4`, `+4studies`; ms per study), and the cap is now 40 min.
   **Item 4, the MAE loader (coded 2026-09-30, CPU-tested):** A4 `data.gpu_crop` (on in `mae_vits.yaml`): workers
   return raw windows + crop boxes (the same rng calls as the cv2 path), and one `torchvision.ops.roi_align` per batch
   crops and resizes on the GPU (`mae_data.gpu_resized_crop`). On 200 real-distribution 384→256 crops it differs from
   cv2 INTER_AREA by 0.065 grey levels on average, max 1. A5 `python -m knee.data.derive_cache --src <cache> --px 256`:
   a 256-px copy (≈54 GB), bit-identical for Track B's load-time resize; for MAE it changes the crops (cut from 256 px
   and enlarged), so it's [val] there and only worth it if the bench shows MAE is read-bound. `scripts/perf/
   mae_loader_bench.py` measures windows/s per cache × workers × windows_per_read × crop, without a GPU. Not done:
   A3's "one batch per `__getitem__`": `windows_per_read` already amortizes the per-item cost; do it only if the bench
   shows Python per-item overhead is the limit.
4. **After the Oct 8 gate** (architecture final): C2, C4, C5 on the chosen arms; B1 checked on real OOF; the efficiency
   submission's window budget.

## 6. What needs a real GPU (the T4 profile: done 2026-09-30, §2b)

A Kaggle GPU kernel with synthetic inputs (no cache needed) that runs, per workload: `torch.profiler` (CUDA kernel time
by op type), peak VRAM with checkpointing on vs off, channels_last on vs off, `torch.compile` on the T4 (does Triton run on
sm75?), and which SDPA backend CoAtNet's biased attention uses in training. Output: this doc's §2 table with measured
columns. Needs Dom's OK (GPU quota).
