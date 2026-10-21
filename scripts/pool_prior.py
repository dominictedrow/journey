"""P4.5 prep: flat per-row ranking inputs for pool_v1, in the same global mass order as mass.npy:
fkey.npy u64 (formula_key of the formula), cid.npy u32 (PubChem CID, 0 = none), level.npy i8 (COCONUT annotation
level, -1 = none), npl.npy f16 (np_likeness, nan = none). Usage: python scripts/pool_prior.py"""
import hashlib
import json
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from casmi.features.formula import formula_key, formula_vector

OUT = 'data/processed/pool_v1'


@lru_cache(maxsize=None)
def fkey(f):
    v = formula_vector(f)
    return int(formula_key(v)) if v is not None else 0


def main():
    m = json.load(open(f'{OUT}/manifest.json'))
    bins, off = m['index']['bins'], m['index']['bin_offsets']
    n = off[-1]
    out = {'fkey': np.zeros(n, np.uint64), 'cid': np.zeros(n, np.uint32), 'level': np.full(n, -1, np.int8),
           'npl': np.full(n, np.nan, np.float16)}
    for b, lo, hi in zip(bins, off[:-1], off[1:]):
        t = pq.read_table(f'{OUT}/pool/{b}', columns=['formula', 'cid', 'level', 'np_likeness'])
        assert t.num_rows == hi - lo
        out['fkey'][lo:hi] = [fkey(f) for f in t['formula'].to_pylist()]
        out['cid'][lo:hi] = np.clip(t['cid'].fill_null(0).to_numpy(), 0, 2**32 - 1)
        out['level'][lo:hi] = t['level'].fill_null(-1).to_numpy()
        out['npl'][lo:hi] = t['np_likeness'].to_numpy(zero_copy_only=False)
        print(b, hi - lo, fkey.cache_info().currsize, flush=True)
    sha = {}
    for k, a in out.items():
        np.save(f'{OUT}/{k}.npy', a)
        sha[f'{k}.npy'] = hashlib.sha256(open(f'{OUT}/{k}.npy', 'rb').read()).hexdigest()
    m['prior'] = {'step': 'P4.5', 'fkey': 'casmi.features.formula.formula_key', 'sha256': sha,
                  'zero_fkey': int((out['fkey'] == 0).sum())}
    json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(m['prior']))


if __name__ == '__main__':
    main()
