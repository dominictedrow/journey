"""Activation memory and forward FLOPs per backbone stage, on CPU (no GPU needed). Picks the stages worth
gradient-checkpointing: the best ones keep many activations for few FLOPs (TrackAModel.set_grad_checkpointing).

    python scripts/perf/stage_memory.py [--backbone coatnet_rmlp_2_rw_384] [--px 384]

Bytes = unique storages saved for backward (saved-tensor hooks) under fp16 autocast, per window. The CPU attention
path keeps the L×L probabilities, which CUDA's memory-efficient SDPA doesn't (03 plan §2b, item 4), so those are
reported separately and left out of the total. Absolute numbers run higher than on CUDA (CPU autocast keeps more
fp32 copies); the per-stage shares are what to read.
"""
from __future__ import annotations

import argparse

import timm
import torch
from torch.utils.flop_counter import FlopCounterMode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="coatnet_rmlp_2_rw_384")
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--n", type=int, default=2, help="windows per forward")
    a = ap.parse_args()
    m = timm.create_model(a.backbone, pretrained=False, num_classes=0)
    for mod in m.modules():  # Track A's freeze_bn: BN in eval mode
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
    cur = torch.randn(a.n, 3, a.px, a.px)
    rows = []
    for name, mod in [("stem", m.stem)] + [(f"stage{i}", s) for i, s in enumerate(m.stages)]:
        seen, att = {}, [0]

        def pack(t):
            if t.dim() == 4 and t.shape[-1] == t.shape[-2] and t.shape[1] != 3 and t.shape[-1] >= 49 \
                    and t.shape[-1] not in (cur.shape[-1], cur.shape[-1] // 2):  # [B, h, L, L] attention
                att[0] += t.numel() * t.element_size()
                return t
            seen[(t.untyped_storage().data_ptr(), t.dtype)] = t.untyped_storage().nbytes()
            return t
        inp = cur.detach().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.float16), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            with FlopCounterMode(display=False) as fc:
                out = mod(inp)
        rows.append((name, sum(seen.values()) / a.n / 1e6, att[0] / a.n / 1e6, fc.get_total_flops() / a.n / 1e9))
        cur = out.detach()
    mem, fl = sum(r[1] for r in rows), sum(r[3] for r in rows)
    print(f"{'part':7s} {'act MB/win':>10s} {'share':>6s} {'(LxL MB)':>9s} {'GFLOP/win':>9s} {'share':>6s}")
    for name, mb, att, gf in rows:
        print(f"{name:7s} {mb:10.1f} {mb / mem:6.0%} {att:9.0f} {gf:9.2f} {gf / fl:6.0%}")


if __name__ == "__main__":
    main()
