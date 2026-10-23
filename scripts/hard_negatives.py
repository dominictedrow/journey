"""P6.2 hard negatives: per train molecule, K pool_v1 decoys within +-PPM of its exact neutral mass, same formula first
(random order), then the rest of the window (random). Decoys sharing the molecule's ik14 (tautomer or plain key) or its
exact packed fingerprint (stereoisomers; false negatives for InfoNCE) are dropped. Mined on CPU (RDKit fps for PubChem
rows via casmi.retrieval.pool.gather); only the 5070 is visible in WSL, so 'built on the 5060' (PLAN P6.2) is moot.
Output data/processed/v1/cache/hardneg_<hash>/: rows int64[M,K] (-1 pad), fp uint8[M,K,256], fkey uint64[M,K], manifest.
Usage: python scripts/hard_negatives.py"""
import hashlib
import json
import os
import time
from multiprocessing import Pool as Workers

import numpy as np
import pyarrow.parquet as pq

from casmi.features.formula import MASS, formula_key
from casmi.retrieval.pool import PPM, Pool, gather
from scripts.pool_recall import plain_key

V1, POOL = 'data/processed/v1', 'data/processed/pool_v1'
CFG = {'k': 32, 'oversample': 48, 'ppm': PPM, 'seed': 2026, 'pool': 'pool_v1'}
CHUNK = 8192


def main():
    h = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:12]
    out = f'{V1}/cache/hardneg_{h}'
    if os.path.exists(f'{out}/manifest.json'):
        print('cached', out)
        return
    os.makedirs(out, exist_ok=True)
    K, O = CFG['k'], CFG['oversample']
    pool = Pool(POOL, ('mass', 'ik14', 'fkey'))
    mols = pq.read_table(f'{V1}/train/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
    mf = np.load(f'{V1}/train/mol_formula.npy')
    mfp = np.load(f'{V1}/train/mol_fp.npy')
    M = len(mf)
    ok = (mf >= 0).all(1) & (mf.sum(1) > 0)
    mass = np.where(ok, mf.astype(np.float64) @ MASS, np.nan)
    fk = formula_key(np.maximum(mf, 0))
    t = time.time()
    with Workers(6) as wk:
        pk = wk.map(plain_key, mols['smiles'], chunksize=2000)
    print(f'plain keys {time.time() - t:.0f}s', flush=True)
    rows = np.lib.format.open_memmap(f'{out}/rows.npy', 'w+', np.int64, (M, K))
    fp = np.lib.format.open_memmap(f'{out}/fp.npy', 'w+', np.uint8, (M, K, 256))
    fkey = np.lib.format.open_memmap(f'{out}/fkey.npy', 'w+', np.uint64, (M, K))
    rows[:] = -1
    rng = np.random.default_rng(CFG['seed'])
    order = np.argsort(mass)  # mass-sorted chunks touch 1-3 pool bins each
    order = order[np.isfinite(mass[order])]
    same_n, n_neg = [], np.zeros(M, np.int16)
    for c0 in range(0, len(order), CHUNK):
        ids = order[c0:c0 + CHUNK]
        cand = {}
        for j in ids:
            lo, hi = pool.window(mass[j], PPM)
            r = np.arange(lo, hi)
            keys = pool.ik14[lo:hi]
            r = r[(keys != mols['ik14'][j].encode()) & (keys != pk[j])]
            same = pool.fkey[r] == fk[j]
            a, b = rng.permutation(r[same]), rng.permutation(r[~same])
            cand[j] = np.concatenate([a, b])[:O]
            same_n.append(min(len(a), K))
        allr = np.unique(np.concatenate(list(cand.values())))
        _, gfp = gather(POOL, allr, workers=6)
        for j, r in cand.items():
            g = gfp[np.searchsorted(allr, r)]
            keep = np.flatnonzero((g != mfp[j]).any(1))[:K]
            n = len(keep)
            rows[j, :n], fp[j, :n], fkey[j, :n] = r[keep], g[keep], pool.fkey[r[keep]]
            n_neg[j] = n
        print(f'{c0 + len(ids)}/{len(order)} mass<={mass[ids[-1]]:.0f} {time.time() - t:.0f}s', flush=True)
    for a in (rows, fp, fkey):
        a.flush()
    np.save(f'{out}/n.npy', n_neg)
    rep = {'step': 'P6.2', 'config': CFG, 'mols': int(M), 'mols_with_mass': int(len(order)),
           'n_neg_mean': float(n_neg.mean()), 'n_neg_full_pct': round(100 * float((n_neg == K).mean()), 1),
           'same_formula_mean': float(np.mean(same_n)), 'minutes': round((time.time() - t) / 60, 1)}
    json.dump(rep, open(f'{out}/manifest.json', 'w'), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
