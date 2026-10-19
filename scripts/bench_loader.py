"""P2.6 benchmark: loader-only throughput vs a dummy P5-sized GPU consumer (d512, 8 layers, bf16, compiled),
plus a coupled 200-step run measuring data-wait share. Target: loader >= 3x model, data-wait < 5%.
Usage: python scripts/bench_loader.py [workers=8] [split=train]"""
import sys
import time

import torch
import torch.nn as nn

from casmi.data.loader import make_loader
from casmi.features.peaks import F

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'train'
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 64


class Dummy(nn.Module):
    def __init__(self, d=512, layers=8):
        super().__init__()
        self.inp = nn.Linear(F + 2, d)
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 2048)

    def forward(self, b):
        x = torch.cat([b['peakfeat'].float(), b['mz'][..., None] / 1000, b['inten'][..., None].float()], -1)
        L = x.shape[1]
        pad = torch.arange(L, device=x.device)[None] >= b['npeaks'][:, None]
        h = self.enc(self.inp(x), src_key_padding_mask=pad)
        h = h.masked_fill(pad[..., None], 0).sum(1) / b['npeaks'].clamp(min=1)[:, None]
        M = b['mol_fp'].shape[0]
        mol = torch.zeros(M, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, b['seg'], h)
        mol = mol / torch.bincount(b['seg'], minlength=M).clamp(min=1)[:, None].to(h.dtype)
        bits = torch.arange(7, -1, -1, device=h.device)
        y = ((b['mol_fp'][..., None].int() >> bits) & 1).flatten(1).float()
        return nn.functional.binary_cross_entropy_with_logits(self.head(mol), y)


def to_dev(b):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def main():
    dl = make_loader('data/processed/v1', SPLIT, workers=W)
    it = iter(dl)
    for _ in range(20):
        next(it)
    t, n, Ls = time.time(), 0, []
    for _ in range(300):
        b = next(it)
        n += len(b['npeaks'])
        Ls.append(b['mz'].shape[1])
    load = n / (time.time() - t)
    print(f'loader-only W={W}: {load:.0f} spectra/s, {300 / (time.time() - t):.1f} batches/s, '
          f'mean L {sum(Ls) / len(Ls):.0f}, mean B {n / 300:.0f}', flush=True)

    model = Dummy().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    step = torch.compile(model)

    def train(b):
        with torch.autocast('cuda', torch.bfloat16):
            loss = step(b)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    cached = [to_dev(next(it)) for _ in range(40)]
    for b in cached:  # compile warm-up over the cached shapes
        train(b)
    torch.cuda.synchronize()
    t, n = time.time(), 0
    for i in range(200):
        b = cached[i % len(cached)]
        train(b)
        n += len(b['npeaks'])
    torch.cuda.synchronize()
    gpu = n / (time.time() - t)
    print(f'model-only: {gpu:.0f} spectra/s; loader/model = {load / gpu:.1f}x', flush=True)

    wait, n = 0.0, 0
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(200):
        t0 = time.time()
        b = to_dev(next(it))
        wait += time.time() - t0
        train(b)
        n += len(b['npeaks'])
    torch.cuda.synchronize()
    tot = time.time() - t
    print(f'coupled 200 steps: {n / tot:.0f} spectra/s, data-wait {100 * wait / tot:.1f}% '
          f'(new shapes may recompile; see model-only for steady state)', flush=True)


if __name__ == '__main__':
    main()
