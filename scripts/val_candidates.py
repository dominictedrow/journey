"""P5.4 prep: fixed candidate pools for val (plan/CLAUDE.md: never rebuild windows during training/eval).
Per val split: windows.npy int64[M, 2] = pool_v1 row range [lo, hi) at +-5 ppm around the molecule's median neutral
mass. Union of all window rows -> rows.npy (sorted pool rows) + fp.npy uint8[n, 256] (stored fp for curated rows,
RDKit for PubChem rows, same generator). Output: data/processed/pool_v1/valcand/ + manifest.json.
Usage: python scripts/val_candidates.py"""
import hashlib
import json
import os
from multiprocessing import Pool as Workers

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.data.splits import VAL
from casmi.features.fp import fingerprints
from casmi.retrieval.pool import PPM, Pool

V1, POOL = 'data/processed/v1', 'data/processed/pool_v1'
OUT = f'{POOL}/valcand'


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = Pool(POOL)
    wins = {}
    for split in VAL:
        off, nm = np.load(f'{V1}/{split}/mol_offsets.npy'), np.load(f'{V1}/{split}/neutral_mass.npy')
        w = np.zeros((len(off) - 1, 2), np.int64)
        for j in range(len(off) - 1):
            m = nm[off[j]:off[j + 1]]
            if np.isfinite(m).any():
                w[j] = pool.window(np.median(m[np.isfinite(m)]), PPM)
        wins[split] = w
        np.save(f'{OUT}/{split}_windows.npy', w)
    rows = np.unique(np.concatenate([np.arange(lo, hi) for w in wins.values() for lo, hi in w]))
    print('union rows', len(rows), flush=True)
    idx = json.load(open(f'{POOL}/manifest.json'))['index']
    boff = np.array(idx['bin_offsets'])
    fp = np.lib.format.open_memmap(f'{OUT}/fp.npy', 'w+', np.uint8, (len(rows), 256))
    which = np.searchsorted(boff, rows, side='right') - 1
    with Workers(6) as wk:
        for bi in np.unique(which):
            sel = np.flatnonzero(which == bi)
            t = pq.read_table(f'{POOL}/pool/{idx["bins"][bi]}', columns=['smiles', 'fp']).take(pa.array(rows[sel] - boff[bi]))
            stored = t['fp'].to_pylist()
            todo = [i for i, x in enumerate(stored) if x is None]
            for i, x in enumerate(stored):
                if x is not None:
                    fp[sel[i]] = np.frombuffer(x, np.uint8)
            smi = t['smiles'].take(pa.array(todo, pa.int64())).to_pylist() if todo else []
            chunks = [smi[k:k + 5000] for k in range(0, len(smi), 5000)]
            if chunks:
                fp[sel[todo]] = np.concatenate(wk.map(fingerprints, chunks))
            print(idx['bins'][bi], len(sel), len(todo), flush=True)
    fp.flush()
    np.save(f'{OUT}/rows.npy', rows)
    sha = {f: hashlib.sha256(open(f'{OUT}/{f}', 'rb').read()).hexdigest() for f in sorted(os.listdir(OUT)) if f.endswith('.npy')}
    json.dump({'step': 'P5.4 prep', 'ppm': PPM, 'rows': int(len(rows)), 'sha256': sha}, open(f'{OUT}/manifest.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
