"""GPU sanity suite + peak-VRAM table for the S03 session log (run on the Windows box; CUDA required).

    python -m knee.model.sanity --out docs/figures/../vram_table.md   (scripts/run_s03.ps1 calls it)

1. manual ≡ sdpa attention on CUDA (fp32 tight, bf16 loose), and which SDPA kernel is picked.
2. One supervised training step (fwd + bwd, bf16 autocast) for a grid of (H, windows per study L, E, depth,
   grad checkpointing): peak VRAM and ms/step, batch = 1 study. OOM is recorded, not fatal.
3. MAE training steps for a few batch sizes at the pretraining resolution: peak VRAM and windows/s.
The CPU unit tests (tests/test_model.py) cover correctness; this covers what only the GPU can tell us.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from knee.model.attention import attend
from knee.model.knee_vit import KneeViT
from knee.model.slice_vit import SliceViT
from knee.ssl.mae import MAE


def fake_batch(L: int, H: int, device) -> dict:
    b = {"x": torch.randint(0, 256, (1, L, 3, H, H), dtype=torch.uint8, device=device),
         "window_mask": torch.ones(1, L, dtype=torch.bool, device=device), "z": torch.rand(1, L, device=device)}
    for k in ("plane", "fluid", "contrast", "field", "slot", "series_idx"):
        b[k] = torch.zeros(1, L, dtype=torch.long, device=device)
    return b


def check_attention(device) -> list[str]:
    out = []
    q, k, v = (torch.randn(4, 6, 257, 64, device=device) for _ in range(3))
    m = torch.ones(4, 257, dtype=torch.bool, device=device)
    m[1, 200:] = False
    for dt, tol in ((torch.float32, 1e-4), (torch.bfloat16, 3e-2)):
        a = attend(q.to(dt), k.to(dt), v.to(dt), m, impl="manual")
        b = attend(q.to(dt), k.to(dt), v.to(dt), m, impl="sdpa")
        err = (a.float() - b.float()).abs().max().item()
        out.append(f"- manual vs sdpa, {dt}: max |diff| = {err:.2e} ({'OK' if err < tol else 'FAIL'})")
    try:  # which fused kernel runs without a mask (the SliceViT case)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
                out.append(f"- SDPA backend {be.name}: available")
            except RuntimeError as e:
                out.append(f"- SDPA backend {be.name}: NOT available ({str(e)[:80]})")
    except ImportError:
        out.append("- torch.nn.attention.sdpa_kernel not available in this torch")
    return out


def sup_step(H: int, L: int, E: int, depth: int, ckpt: bool, head: str, device) -> tuple[float, float]:
    heads = max(1, E // 64)
    hcfg = {"kind": head} if head == "attn_mil" else {"kind": head, "heads": heads}
    m = KneeViT({"slice_vit": {"img_size": H, "dim": E, "depth": depth, "heads": heads, "grad_checkpointing": ckpt},
                 "head": hcfg}).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    b = fake_batch(L, H, device)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(m(b).float(), torch.rand(1, 12, device=device))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i:
            times.append(time.time() - t0)
    return torch.cuda.max_memory_allocated() / 1e9, 1000 * sum(times) / len(times)


def mae_steps(H: int, bs: int, E: int, depth: int, device) -> tuple[float, float]:
    mae = MAE(SliceViT(img_size=H, dim=E, depth=depth, heads=max(1, E // 64))).to(device).train()
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, 256, (bs, 3, H, H), dtype=torch.uint8, device=device)
    torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, None
    for i in range(6):
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = mae(x)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        n += bs if i >= 2 else 0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9, n / (time.time() - t0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("vram_table.md"))
    ap.add_argument("--quick", action="store_true", help="a smaller grid")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (this is the GPU half of the S03 sanity suite)")
    device = torch.device("cuda")
    lines = [f"# S03 GPU sanity + VRAM table ({torch.cuda.get_device_name()}, torch {torch.__version__})", "",
             "## Attention", *check_attention(device), "",
             "## Supervised step (1 study, bf16 autocast, AdamW; peak allocated GB / ms per step)", "",
             "| H | windows L | E | depth | grad ckpt | head | peak GB | ms/step |", "|---|---|---|---|---|---|---|---|"]
    grid = itertools.product((256,) if a.quick else (224, 256, 384), (40, 80) if a.quick else (40, 80, 120),
                             (384,) if a.quick else (192, 384), (12,), (False, True), ("attn_mil", "finding_query"))
    for H, L, E, depth, ckpt, head in grid:
        try:
            gb, ms = sup_step(H, L, E, depth, ckpt, head, device)
            row = f"| {H} | {L} | {E} | {depth} | {ckpt} | {head} | {gb:.2f} | {ms:.0f} |"
        except torch.cuda.OutOfMemoryError:
            row = f"| {H} | {L} | {E} | {depth} | {ckpt} | {head} | OOM | - |"
        torch.cuda.empty_cache()
        print(row, flush=True)
        lines.append(row)
    lines += ["", "## MAE step (mask 0.75, bf16; peak allocated GB / windows per s)", "",
              "| H | batch | E | depth | peak GB | windows/s |", "|---|---|---|---|---|---|"]
    for H, bs, E in itertools.product((224, 256), (64, 128) if a.quick else (32, 64, 128, 256), (384,)):
        try:
            gb, wps = mae_steps(H, bs, E, 12, device)
            row = f"| {H} | {bs} | {E} | 12 | {gb:.2f} | {wps:.0f} |"
        except torch.cuda.OutOfMemoryError:
            row = f"| {H} | {bs} | {E} | 12 | OOM | - |"
        torch.cuda.empty_cache()
        print(row, flush=True)
        lines.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
