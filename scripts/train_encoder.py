"""P5 training: Encoder on data/processed/v1 shards, bf16 autocast + torch.compile + fused AdamW, cosine lr.
--profile: 200 timed steps -> spectra/s, data-wait share, peak memory (plan/CLAUDE.md profile gate), no checkpoint.
Usage: python scripts/train_encoder.py --name p5_v1 [--steps N] [--lr X] [--profile]"""
import argparse
import csv
import json
import os
import subprocess
import time

import torch

from casmi.data.loader import make_loader
from casmi.models.encoder import Encoder, feature_stats, loss_fn, tanimoto, unpack_fp

ROOT = 'data/processed/v1'
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 64


def to_dev(b):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, split, max_batches=None):
    model.eval()
    tot, tani, n = 0.0, 0.0, 0
    for i, b in enumerate(make_loader(ROOT, split, workers=2, shuffle=False, max_spec_per_mol=None)):
        if max_batches and i >= max_batches:
            break
        b = to_dev(b)
        with torch.autocast('cuda', torch.bfloat16):
            fp, form = model(b)
        loss, _ = loss_fn(fp, form, b)
        m = len(b['mol_idx'])
        tot += loss.item() * m
        tani += tanimoto((fp.float() > 0).float(), unpack_fp(b['mol_fp'])).sum().item()
        n += m
    model.train()
    return {'loss': tot / n, 'tanimoto': tani / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--steps', type=int, default=50_000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--spb', type=int, default=256)
    ap.add_argument('--eval_every', type=int, default=5000)
    ap.add_argument('--profile', action='store_true')
    a = ap.parse_args()
    torch.manual_seed(0)
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    model = Encoder({k: len(v) for k, v in vocab.items()}).to(dev)
    model.set_stats(*(x.to(dev) for x in feature_stats(ROOT)))
    print(f'params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (
        1 + torch.cos(torch.tensor(min(s, a.steps) / a.steps * torch.pi)).item()))
    # compile only the transformer stack: whole-model dynamic recompiles hit an Inductor tiling assert (torch 2.11)
    model.enc = torch.compile(model.enc, dynamic=True)
    fwd = model

    def step(b):
        with torch.autocast('cuda', torch.bfloat16):
            fp, form = fwd(b)
        loss, parts = loss_fn(fp, form, b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        return loss, parts

    dl = make_loader(ROOT, 'train', workers=8, spec_per_batch=a.spb)
    it = (b for _ in iter(int, 1) for b in dl)  # endless: each pass over dl is a new block-shuffled epoch
    os.makedirs(f'runs/{a.name}', exist_ok=True)
    if a.profile:
        for _ in range(60):  # warm-up: compile the common length buckets
            step(to_dev(next(it)))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        wait, n, t = 0.0, 0, time.time()
        for _ in range(200):
            t0 = time.time()
            b = to_dev(next(it))
            wait += time.time() - t0
            loss, _ = step(b)
            n += len(b['npeaks'])
        torch.cuda.synchronize()
        tot = time.time() - t
        rep = {'spectra_per_s': round(n / tot), 'steps_per_s': round(200 / tot, 2), 'data_wait_pct': round(100 * wait / tot, 1),
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'loss': round(loss.item(), 4),
               'epoch_h': round(1_948_633 / (n / tot) / 3600, 2)}
        print(json.dumps(rep), flush=True)
        json.dump(rep, open(f'runs/{a.name}/profile.json', 'w'))
        return
    t, seen, start = time.time(), 0, 1
    state = f'runs/{a.name}/state.pt'
    if os.path.exists(state):  # resume after a crash (WSL CUDA 'unknown error'); data order restarts, fine for SGD
        st = torch.load(state, map_location=dev)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start = st['step'] + 1
        print(f'resumed at step {start}', flush=True)
    for s in range(start, a.steps + 1):
        b = to_dev(next(it))
        loss, parts = step(b)
        seen += len(b['npeaks'])
        if s % 1000 == 0:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': s},
                       state + '.tmp')
            os.replace(state + '.tmp', state)
        if s % 200 == 0:
            print(f'step {s} loss {loss.item():.4f} focal {parts["focal"].item():.4f} formula {parts["formula"].item():.4f} '
                  f'lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t):.0f} spec/s', flush=True)
        if s % a.eval_every == 0 or s == a.steps:
            ev = {sp: evaluate(model, sp) for sp in ('v_lib', 'v_pool', 'v_novel')}
            print(f'eval step {s}', json.dumps(ev), flush=True)
            torch.save({'model': model.state_dict(), 'step': s, 'args': vars(a), 'eval': ev}, f'runs/{a.name}/last.pt')
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    new = not os.path.exists('runs/runs.csv')
    with open('runs/runs.csv', 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['name', 'git', 'steps', 'lr', 'hours', 'eval'])
        w.writerow([a.name, sha, a.steps, a.lr, round((time.time() - t) / 3600, 2), json.dumps(ev)])


if __name__ == '__main__':
    main()
