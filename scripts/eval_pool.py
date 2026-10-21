"""P4.5 local val of library + pool_v1 ranking (casmi.retrieval.pool): MRR@25 per val split and the D-019 weighted
mean, swept over the library-cosine tier threshold. Honesty: the library excludes V-novel (as P3.3) and every val
structure loses its train source bit in the pool (D-026: V-novel answers stay findable via external sources).
Truth matched on tautomer ik14 or plain InChIKey14 (PubChem rows, D-025).
Usage: python scripts/eval_pool.py -> reports/eval_pool.json"""
import json
import time

import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger

from casmi.data.splits import VAL
from casmi.retrieval.library import Library, bin_rows
from casmi.retrieval.pool import Pool, formula_score, rank
from scripts.pool_recall import plain_key

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
COS_TS = (0.0, 0.1, 0.2, 0.3, 0.5, 2.0)  # 2.0 = no library tier
RDLogger.DisableLog('rdApp.*')


def main():
    t0 = time.time()
    sp = pq.read_table(f'{ROOT}/splits.parquet', columns=['ik14', 'split']).to_pydict()
    novel = [k for k, s in zip(sp['ik14'], sp['split']) if s == 'v_novel']
    strip = np.array(sorted(k.encode() for k, s in zip(sp['ik14'], sp['split']) if s in VAL), 'S14')
    lib, pool = Library(ROOT, ('train',), exclude_ik14=novel), Pool(POOL)
    res = {}
    for split in VAL:
        d = f'{ROOT}/{split}'
        off = np.load(f'{d}/mol_offsets.npy')
        mols = pq.read_table(f'{d}/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
        q = bin_rows(*(np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks')))
        nm = np.load(f'{d}/neutral_mass.npy')
        forms, nform = np.load(f'{d}/formulas.npy', mmap_mode='r'), np.load(f'{d}/nformulas.npy')
        rr = {t: [] for t in COS_TS}
        for j, (k, smi) in enumerate(zip(mols['ik14'], mols['smiles'])):
            a, b = off[j], off[j + 1]
            m = nm[a:b]
            m = np.median(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan
            c, s, has = lib.scores(q[a:b], m) if np.isfinite(m) else ([], [], [])
            lib_cos = {lib.ik14[i].encode(): float(x) for i, x, h in zip(c, s, has) if h}
            fs = formula_score(forms[a:b], nform[a:b])
            truth = {k.encode(), plain_key(smi)}
            for t in COS_TS:
                ks = pool.ik14[rank(pool, m, fs, lib_cos, strip_train=strip, cos_t=t)].tolist()
                hit = [i for i, g in enumerate(ks) if g in truth]
                rr[t].append(1 / (hit[0] + 1) if hit else 0.0)
        res[split] = {str(t): {'mrr': round(float(np.mean(v)), 4), 'top1': round(float(np.mean(np.array(v) == 1)), 4)}
                      for t, v in rr.items()}
        print(split, json.dumps(res[split]), f'({time.time() - t0:.0f}s)', flush=True)
    res['weighted'] = {str(t): round(float(np.mean([res[s][str(t)]['mrr'] for s in ('v_lib', 'v_pool', 'v_novel')])), 4)
                       for t in COS_TS}
    print(json.dumps(res['weighted']))
    json.dump(res, open('reports/eval_pool.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
