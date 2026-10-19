"""P3.3 local val of the P3.1 library baseline: MRR@25 per val split (+ equal-weight mean of lib/pool/novel, D-019).
Truth matched on tautomer-canonical ik14 from structures.parquet (same rule as casmi.metric, no RDKit per guess).
Usage: python scripts/eval_baseline.py"""
import json
import time

import numpy as np
import pyarrow.parquet as pq

from casmi.data.splits import VAL
from casmi.retrieval.library import Library, bin_rows

ROOT = 'data/processed/v1'


def main():
    t0 = time.time()
    sp = pq.read_table(f'{ROOT}/splits.parquet', columns=['ik14', 'split']).to_pydict()
    novel = [k for k, s in zip(sp['ik14'], sp['split']) if s == 'v_novel']
    lib = Library(ROOT, ('train',), exclude_ik14=novel)
    print(f'library: {len(lib.mass)} candidates, {lib.refs.shape[0]} ref spectra ({time.time() - t0:.0f}s)', flush=True)
    res = {}
    for split in VAL:
        d = f'{ROOT}/{split}'
        off = np.load(f'{d}/mol_offsets.npy')
        truth = pq.read_table(f'{d}/mols.parquet', columns=['ik14'])['ik14'].to_pylist()
        q = bin_rows(*(np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks')))
        nm = np.load(f'{d}/neutral_mass.npy')
        rr, inwin, ncand = [], [], []
        for j, k in enumerate(truth):
            m = nm[off[j]:off[j + 1]]
            m = np.median(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan
            ks, _, _ = lib.rank(q[off[j]:off[j + 1]], m)
            rr.append(1 / (ks.index(k) + 1) if k in ks else 0.0)
            c = lib.candidates(m) if np.isfinite(m) else []
            ncand.append(len(c))
            inwin.append(k in set(lib.ik14[c]))
        res[split] = {'mrr': round(float(np.mean(rr)), 4), 'top1': round(float(np.mean(np.array(rr) == 1)), 4),
                      'in_window': round(float(np.mean(inwin)), 4), 'median_cands': float(np.median(ncand))}
        print(split, res[split], f'({time.time() - t0:.0f}s)', flush=True)
    res['weighted'] = round(float(np.mean([res[s]['mrr'] for s in ('v_lib', 'v_pool', 'v_novel')])), 4)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
