"""P6 round 2 (D-030) model-mined negatives: per train molecule, the top-K wrong pool_v1 candidates in its +-PPM window
by a trained Contrastive checkpoint's blend score (c * cosine + t * soft Tanimoto, as scripts/eval_encoder.py). Truth keys
(ik14, plain key) and decoys with the molecule's exact packed fp are dropped, as scripts/hard_negatives.py. Window fps
come from casmi.retrieval.pool.gather per mass-sorted chunk (RDKit for PubChem rows; ~68M rows in all, CPU-bound).
Output data/processed/v1/cache/modelneg_<hash>/ in the hard_negatives.py layout, so --hardneg takes it unchanged.
Usage: python scripts/model_negatives.py runs/<name>/last.pt [--limit N_CHUNKS]  (--limit: timing profile, no manifest)"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool as Workers

import numpy as np
import pyarrow.parquet as pq
import torch

from casmi.features.formula import MASS, fkey_counts, formula_key
from casmi.models.contrastive import Contrastive
from casmi.models.encoder import unpack_fp
from casmi.retrieval.pool import PPM, Pool, gather
from scripts.eval_encoder import predict
from scripts.pool_recall import plain_key

V1, POOL = 'data/processed/v1', 'data/processed/pool_v1'
CHUNK, EB = 1024, 65536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--t', type=float, default=4.0)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    cfg = {'ckpt': a.ckpt, 'k': a.k, 'c': a.c, 't': a.t, 'ppm': PPM, 'pool': 'pool_v1'}
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out = f'{V1}/cache/modelneg_{h}' + ('_profile' if a.limit else '')
    if os.path.exists(f'{out}/manifest.json'):
        print('cached', out)
        return
    os.makedirs(out, exist_ok=True)
    K, t = a.k, time.time()
    vocab = json.load(open(f'{V1}/vocab.json'))
    model = Contrastive({k: len(v) for k, v in vocab.items()}).cuda().eval()
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load(a.ckpt, map_location='cuda')['model'].items()})
    prob, emb = predict(model, 'train')
    prob, emb = torch.from_numpy(prob).cuda().half(), torch.from_numpy(emb).cuda()
    print(f'predict {time.time() - t:.0f}s', flush=True)
    pool = Pool(POOL, ('mass', 'ik14', 'fkey'))
    mols = pq.read_table(f'{V1}/train/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
    mf, mfp = np.load(f'{V1}/train/mol_formula.npy'), np.load(f'{V1}/train/mol_fp.npy')
    M = len(mf)
    ok = (mf >= 0).all(1) & (mf.sum(1) > 0)
    mass = np.where(ok, mf.astype(np.float64) @ MASS, np.nan)
    with Workers(14) as wk:
        pk = wk.map(plain_key, mols['smiles'], chunksize=2000)
    rows = np.lib.format.open_memmap(f'{out}/rows.npy', 'w+', np.int64, (M, K))
    fp = np.lib.format.open_memmap(f'{out}/fp.npy', 'w+', np.uint8, (M, K, 256))
    fkey = np.lib.format.open_memmap(f'{out}/fkey.npy', 'w+', np.uint64, (M, K))
    rows[:] = -1
    n_neg = np.zeros(M, np.int16)
    order = np.argsort(mass)
    order = order[np.isfinite(mass[order])]
    chunks = range(0, len(order), CHUNK)[:a.limit]
    same_n, truth_rank = [], []
    tf = 0.0
    for c0 in chunks:
        ids = order[c0:c0 + CHUNK]
        wins = [pool.window(mass[j], PPM) for j in ids]
        allr = np.unique(np.concatenate([np.arange(lo, hi) for lo, hi in wins]))
        t0 = time.time()
        _, gfp = gather(POOL, allr, workers=14)
        tf += time.time() - t0
        G = torch.from_numpy(gfp).cuda()
        cnt = torch.from_numpy(fkey_counts(pool.fkey[allr])).cuda()
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            E = torch.cat([model.mol(unpack_fp(G[i:i + EB]), cnt[i:i + EB]).half() for i in range(0, len(allr), EB)])
        keys, fks = pool.ik14[allr], pool.fkey[allr]
        mfp_g = torch.from_numpy(mfp[ids]).cuda()
        for jj, (j, (lo, hi)) in enumerate(zip(ids, wins)):
            if hi <= lo:
                continue
            s = slice(np.searchsorted(allr, lo), np.searchsorted(allr, lo) + hi - lo)  # window rows are contiguous in allr
            bits, p = unpack_fp(G[s]), prob[j].float()
            inter = bits @ p
            score = a.c * (E[s].float() @ emb[j]) + a.t * inter / (bits.sum(1) + p.sum() - inter).clamp(min=1e-6)
            bad = (G[s] == mfp_g[jj]).all(1).cpu().numpy() | (keys[s] == mols['ik14'][j].encode()) | (keys[s] == pk[j])
            truth = np.flatnonzero((keys[s] == mols['ik14'][j].encode()) | (keys[s] == pk[j]))
            if len(truth):
                truth_rank.append(int((score > score[truth].max()).sum()))
            score[torch.from_numpy(bad).cuda()] = float('-inf')
            n = min(K, int((~bad).sum()))
            top = torch.topk(score, n).indices.cpu().numpy()
            r = s.start + top
            rows[j, :n], fp[j, :n], fkey[j, :n] = allr[r], gfp[r], fks[r]
            n_neg[j] = n
            same_n.append(int((fks[r] == formula_key(np.maximum(mf[j], 0))).sum()))
        print(f'{c0 + len(ids)}/{len(order)} mass<={mass[ids[-1]]:.0f} rows {len(allr)} {time.time() - t:.0f}s '
              f'(fp {tf:.0f}s)', flush=True)
    for x in (rows, fp, fkey):
        x.flush()
    np.save(f'{out}/n.npy', n_neg)
    tr = np.array(truth_rank)
    rep = {'step': 'P6 round 2 (D-030)', 'config': cfg, 'mols': int(M), 'mols_with_mass': int(len(order)),
           'n_neg_mean': float(n_neg[order[:len(chunks) * CHUNK]].mean()),
           'same_formula_mean': float(np.mean(same_n)),
           'train_truth_rank': {'median': float(np.median(tr)), 'top1': float((tr == 0).mean()),
                                'top25': float((tr < 25).mean())},
           'minutes': round((time.time() - t) / 60, 1), 'fp_minutes': round(tf / 60, 1)}
    print(json.dumps(rep), flush=True)
    if not a.limit:
        json.dump(rep, open(f'{out}/manifest.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
