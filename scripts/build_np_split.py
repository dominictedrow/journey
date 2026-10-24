"""D-035 (b): V-np, a natural-product val fold: every train molecule with enveda (timsTOF) spectra whose structure is in
COCONUT or LOTUS (pool_v1 src bits); only ~200 exist, most also have public-library spectra. Their enveda spectra are
copied out of the train shard into data/processed/v1/v_np/ (same arrays, molecule order kept) as queries. splits.parquet
stays frozen: training and the val library skip these molecules entirely (casmi.data.splits.np_keys), so V-np is a
strict kind-2 test. Usage: python scripts/build_np_split.py"""
import json
import os

import numpy as np
import pyarrow.parquet as pq

from casmi.retrieval.pool import CURATED_EXT, Pool

V1 = 'data/processed/v1'
OUT = f'{V1}/v_np'
ENVEDA = (2, 3)  # vocab.json ingest_lib ids of enveda-180, enveda-np-examples


def main():
    if os.path.exists(f'{OUT}/mols.parquet'):
        print(f'{OUT} exists; the fold is frozen')
        return
    tr = f'{V1}/train'
    mols = pq.read_table(f'{tr}/mols.parquet')
    ik = np.array([k.encode() for k in mols['ik14'].to_pylist()], 'S14')
    sp = pq.read_table(f'{V1}/splits.parquet', columns=['ik14', 'split', 'n_enveda']).to_pydict()
    env = {k.encode() for k, s, e in zip(*sp.values()) if s == 'train' and e > 0}
    pool = Pool('data/processed/pool_v1', ('ik14', 'src'))
    cur = set()
    for a in range(0, len(pool.ik14), 10_000_000):
        k, s = np.asarray(pool.ik14[a:a + 10_000_000]), pool.src[a:a + 10_000_000]
        cur.update(k[((s & CURATED_EXT) > 0) & np.isin(k, ik)].tolist())
    pick = np.flatnonzero([k in env and k in cur for k in ik])
    off, lib = np.load(f'{tr}/mol_offsets.npy'), np.load(f'{tr}/lib.npy')
    spec = [np.arange(off[j], off[j + 1]) for j in pick]
    spec = [x[np.isin(lib[x], ENVEDA)] for x in spec]
    cnt = np.array([len(x) for x in spec])
    spec = np.concatenate(spec)
    os.makedirs(OUT)
    n_spec, n_mol = off[-1], len(off) - 1
    for f in sorted(os.listdir(tr)):
        if not f.endswith('.npy') or f in ('mol_offsets.npy', 'mol_idx.npy'):
            continue
        a = np.load(f'{tr}/{f}', mmap_mode='r')
        assert len(a) in (n_spec, n_mol) and n_spec != n_mol, f
        np.save(f'{OUT}/{f}', np.asarray(a[spec] if len(a) == n_spec else a[pick]))
    np.save(f'{OUT}/mol_offsets.npy', np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64))
    np.save(f'{OUT}/mol_idx.npy', np.repeat(np.arange(len(pick)), cnt).astype(np.int32))
    pq.write_table(mols.take(pick), f'{OUT}/mols.parquet')
    stats = {'molecules': int(len(pick)), 'spectra': int(len(spec)), 'train_rows': pick.tolist()}
    json.dump(stats, open(f'{OUT}/manifest.json', 'w'))
    print({k: v for k, v in stats.items() if k != 'train_rows'})


if __name__ == '__main__':
    main()
