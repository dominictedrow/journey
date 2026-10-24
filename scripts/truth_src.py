"""P7 gap diagnosis: V-set truth source mix (curated NP vs PubChem-only) and no-tier MRR by source (runs/p6_ft/top200.npz).
Usage: PYTHONPATH=. python scripts/truth_src.py"""
import numpy as np
import pyarrow.parquet as pq

from casmi.data.splits import VAL
from casmi.retrieval.pool import CURATED_EXT, Pool

ROOT, VC = 'data/processed/v1', 'data/processed/pool_v1/valcand'
pool = Pool('data/processed/pool_v1', ('ik14', 'src'))
t = np.load('runs/p6_ft/top200.npz')
for s in VAL:
    ik = np.array([k.encode() for k in pq.read_table(f'{ROOT}/{s}/mols.parquet', columns=['ik14'])['ik14'].to_pylist()], 'S14')
    wins = np.load(f'{VC}/{s}_windows.npy')
    blend, hit, rows = t[f'{s}_blend'], t[f'{s}_hit'], t[f'{s}_rows']
    cat, rr, ncur, nwin = [], [], [], []
    for j, (lo, hi) in enumerate(wins):
        src = pool.src[lo:hi]
        m = np.flatnonzero(pool.ik14[lo:hi] == ik[j])
        v = int(np.bitwise_or.reduce(src[m])) if len(m) else -1
        cat.append('absent' if v < 0 else 'curated' if v & CURATED_EXT else 'train-only' if v == 1 else 'pubchem')
        ok = rows[j] >= 0
        b = np.where(ok, blend[j], -1e9)
        r = int((b > b[hit[j]].max()).sum()) if hit[j].any() else 999
        rr.append(1 / (r + 1) if r < 25 else 0.0)
        ncur.append(((src & CURATED_EXT) > 0).sum() + ((src & 1) > 0).sum())
        nwin.append(hi - lo)
    cat, rr = np.array(cat), np.array(rr)
    print(s, f'n={len(cat)} median window {int(np.median(nwin))}, median curated+train {int(np.median(ncur))}')
    for c in ('curated', 'pubchem', 'train-only', 'absent'):
        k = cat == c
        if k.any():
            print(f'  {c:10s} share {k.mean():.3f}  MRR {rr[k].mean():.3f}')
