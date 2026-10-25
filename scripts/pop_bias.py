"""D-037 follow-up: is PubChem popularity a near-oracle on a val split? Per molecule, the truth's popularity percentile
among its +-5 ppm window rows (all rows, and curated train|COCONUT|LOTUS rows only).
Writes <valcand>/<split>_pop_pct.npy (curated percentile per molecule, nan = truth not in window); D-038 uses
pct < 0.9 on V-np as the de-biased NP metric (scripts/np_probe.py).
Usage: python scripts/pop_bias.py v_np [v_pool ...]"""
import sys

import numpy as np
import pyarrow.parquet as pq

from casmi.retrieval.pool import Pool

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
pool = Pool(POOL, ('ik14', 'src', 'popularity'))
for split in sys.argv[1:]:
    vc = f'{POOL}/valcand' + ('_' + split if split == 'v_np' else '')
    wins = np.load(f'{vc}/{split}_windows.npy')
    keys = pq.read_table(f'{ROOT}/{split}/mols.parquet', columns=['ik14']).column('ik14').to_pylist()
    out = {'all': [], 'curated': []}
    pct = np.full(len(keys), np.nan, np.float32)
    for j, ((lo, hi), k) in enumerate(zip(wins, keys)):
        if hi <= lo:
            continue
        ik, pop, src = pool.ik14[lo:hi], np.asarray(pool.popularity[lo:hi], np.float32), pool.src[lo:hi]
        t = ik == k.encode()
        if not t.any():
            continue
        tp = pop[t].max()
        for name, m in (('all', np.ones(len(ik), bool)), ('curated', (src & 7) > 0)):
            out[name].append(float((pop[m & ~t] < tp).mean()) if (m & ~t).any() else 1.0)
        pct[j] = out['curated'][-1]
    np.save(f'{vc}/{split}_pop_pct.npy', pct)
    for name, v in out.items():
        v = np.array(v)
        print(split, name, 'n', len(v), 'median pct', round(float(np.median(v)), 3), 'pct>=0.9', round(float((v >= 0.9).mean()), 3),
              'top1 (pct 1.0)', round(float((v == 1.0).mean()), 3))
