"""Fixed-shape encoding for torch.compile, and pure-half inference (docs/plan/03_performance_plan.md, B1 + B4 in training).

**Why buckets.** torch.compile specializes a graph to the input shape. A study's windows are encoded in chunks of
`window_chunk`, and the last chunk is ragged (44 windows at chunk 64 → one chunk of 44; 60 at chunk 16 → 16, 16, 16,
12), so every new bag size would trigger a recompile (or a slower dynamic-shape graph). Instead, a bag is split into
**bucket** sizes (by default the powers of two from `min_bucket` up to `window_chunk`): 44 → 32 + 8 + 4, 60 at chunk
16 → 16 + 16 + 16 + 8 + 4. Only a remainder smaller than the smallest bucket is padded (15 → 8 + 4 + 3 padded to 4),
so padding costs < min_bucket windows per study, and the encoder only ever sees a handful of shapes, each compiled
once. Padded rows are dropped from the output, so they get no gradient.

Padding copies real windows (cyclically), it doesn't use zeros:
- SliceViT (GroupNorm/LayerNorm only) treats every window independently, so the real outputs are identical either way.
- CoAtNet has BatchNorm. In **training** mode BN normalizes with the chunk's own batch statistics, so the padded
  copies nudge the statistics of the last chunk (a few duplicated windows; zeros would distort them far more). That's
  a small change to the training numerics, the same kind the ragged chunk already causes. In eval mode BN uses the
  running statistics and the output is exact.
- Bucketed chunks are not the same chunks as the plain split, so in training BN's batch statistics also come from
  different groups of windows. Same kind of change, same caveat: Track A training with compile is [val], not [exact].

**Pure half** (`to_half`): the whole model in fp16 with fp16 inputs and no autocast. Under autocast, CUDA runs
LayerNorm/softmax in fp32 and casts around them. On the T4 those casts plus fp32 LayerNorm were 58% of Track B's
inference kernel time (§2b). PyTorch's fp16 LayerNorm still accumulates its statistics in fp32 internally. The heads
already run their softmax in fp32 and the caller casts logits to fp32 before the sigmoid.
"""
from __future__ import annotations

import torch
from torch import nn


def default_buckets(window_chunk: int, min_bucket: int = 4) -> list[int]:
    """Powers of two from min_bucket up to window_chunk, plus window_chunk itself (e.g. 16 → [4, 8, 16])."""
    out, b = {int(window_chunk)}, 1
    while b < window_chunk:
        if b >= min_bucket:
            out.add(b)
        b *= 2
    return sorted(out)


def bucket_for(n: int, buckets: list[int]) -> int:
    """The smallest bucket that holds n windows."""
    for b in buckets:
        if b >= n:
            return b
    raise ValueError(f"chunk of {n} is larger than the largest bucket {buckets[-1]}")


def plan_chunks(n: int, buckets: list[int]) -> list[tuple[int, int]]:
    """Split n windows into (real, shape) chunks: greedily the largest bucket that fits, then the remainder padded up
    to the smallest bucket that holds it. Padding is always < buckets[0] windows (44 at [4..64] → 32 + 8 + 4, no
    padding; 15 at [4, 8, 16] → 8 + 4 + 3→4)."""
    out, rem = [], n
    while rem > 0:
        fit = [b for b in buckets if b <= rem]
        if fit:
            out.append((fit[-1], fit[-1]))
            rem -= fit[-1]
        else:
            out.append((rem, bucket_for(rem, buckets)))
            rem = 0
    return out


def encode_chunks(encode, x: torch.Tensor, window_chunk: int, buckets: list[int] | None = None) -> list[torch.Tensor]:
    """encode(chunk) over x [n, ...]. Without buckets: plain chunks of window_chunk (the last one ragged). With
    buckets (whose largest should be window_chunk): chunks from plan_chunks, a padded chunk filled with copies of its
    own windows, and the padded outputs dropped. Returns the per-chunk outputs, in order."""
    if buckets is None:
        return [encode(c) for c in x.split(window_chunk)]
    outs, i = [], 0
    for real, shape in plan_chunks(x.shape[0], buckets):
        c = x[i:i + real]
        i += real
        if shape > real:
            c = c.index_select(0, torch.arange(shape, device=c.device) % real)
        outs.append(encode(c)[:real])
    return outs


def raise_recompile_limit(n: int = 64) -> None:
    """Dynamo's default of 8 graphs per function is too few: buckets × (train, eval) × (grad, no_grad)."""
    import torch._dynamo.config as dc
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dc, name):
            setattr(dc, name, max(getattr(dc, name), n))


def compile_module(m: nn.Module, mode: str | None = None) -> nn.Module:
    """In-place `nn.Module.compile`: parameter names and state_dict keys stay the same (no `_orig_mod.` prefix).
    dynamic=False: one static graph per bucket shape instead of drifting to a dynamic-shape graph after a recompile."""
    raise_recompile_limit()
    kw = {"dynamic": False}
    if mode and mode != "default":
        kw["mode"] = mode
    m.compile(**kw)
    return m


# ---- compile cache across sessions (PyTorch "Mega-Cache", torch >= 2.7). The scored Kaggle run counts compile time
# (measured 49 s Track B, 86 s Track A per process on the T4, §2b, and every GPU process compiles its own copy).
# A T4 session saves the compiled artifacts (save_compile_cache → <out>/compile_cache_<device>.bin); shipped in the
# bundle as compile_cache.bin, the next run loads them before compiling, so inductor/Triton codegen becomes a cache
# hit (Dynamo still traces). PyTorch validates the artifacts against the torch + Triton versions and the GPU, so they
# only help on the same Kaggle image (pin the notebook's environment) and the same GPU type; on a mismatch they are
# ignored and compilation runs as usual. Speed-up UNVERIFIED until a T4 run measures it.
COMPILE_CACHE_IN = "KNEE_COMPILE_CACHE_IN"    # file to load, set by notebooks/infer.py
COMPILE_CACHE_OUT = "KNEE_COMPILE_CACHE_OUT"  # directory to save into
_cache_loaded = False


def load_compile_cache() -> bool:
    """Load the artifacts named by $KNEE_COMPILE_CACHE_IN once per process. Never raises: a bad or stale cache only
    costs the compile it would have saved."""
    import logging
    import os
    global _cache_loaded
    path = os.environ.get(COMPILE_CACHE_IN)
    if _cache_loaded or not path or not os.path.isfile(path) or not hasattr(torch.compiler, "load_cache_artifacts"):
        return False
    _cache_loaded = True
    try:
        with open(path, "rb") as f:
            info = torch.compiler.load_cache_artifacts(f.read())
        if info is None:  # torch logs why (corrupt, or another torch/Triton/GPU) and returns None
            logging.getLogger(__name__).warning("compile cache %s rejected; compiling from scratch", path)
            return False
        logging.getLogger(__name__).info("compile cache loaded from %s", path)
        return True
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("compile cache %s not loaded: %s", path, e)
        return False


def save_compile_cache(tag: str) -> str | None:
    """Write this process's compile artifacts to $KNEE_COMPILE_CACHE_OUT/compile_cache_<tag>.bin. Never raises."""
    import logging
    import os
    out = os.environ.get(COMPILE_CACHE_OUT)
    if not out or not hasattr(torch.compiler, "save_cache_artifacts"):
        return None
    try:
        r = torch.compiler.save_cache_artifacts()
        if not r or not r[0]:  # nothing was compiled in this process
            return None
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"compile_cache_{tag.replace(':', '')}.bin")
        with open(path, "wb") as f:
            f.write(r[0])
        return path
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("compile cache not saved: %s", e)
        return None


def is_half(model: nn.Module) -> bool:
    p = next(model.parameters(), None)
    return p is not None and p.dtype == torch.float16


def to_half(model: nn.Module) -> nn.Module:
    """Inference only: every parameter and floating buffer in fp16. Call after the folds (fuse.prepare_for_inference).
    Run the model without autocast afterwards: autocast would put the fp32 LayerNorm and its casts back."""
    return model.half()
