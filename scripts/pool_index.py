"""P4.2 (rest): add packed Morgan fp (2048 bits -> 256 bytes, same generator as the shards) to curated rows
(src & 7: train/COCONUT/LOTUS; PubChem fps are computed per window at inference, D-024), and write flat arrays in
global mass order: pool_v1/{mass.npy f64, ik14.npy S14, src.npy u8, taut.npy bool}. Bins are already mass-sorted,
so concatenating them in bin order is globally sorted. Usage: python scripts/pool_index.py"""
import glob
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = 'data/processed/pool_v1'
CURATED = 7


def add_fp(path):
    from casmi.features.fp import fingerprints  # same generator as the shard targets
    t = pq.read_table(path)
    if 'fp' in t.column_names:
        t = t.drop_columns(['fp'])
    cur = np.flatnonzero(np.bitwise_and(t['src'].to_numpy(), CURATED) > 0)
    fps = [None] * t.num_rows
    if len(cur):
        arr = fingerprints(t['smiles'].take(pa.array(cur)).to_pylist())
        for i, row in zip(cur, arr):
            fps[i] = row.tobytes()
    t = t.append_column('fp', pa.array(fps, pa.binary(256)))
    pq.write_table(t, path, row_group_size=65536)
    return path, len(cur)


def main():
    bins = sorted(glob.glob(f'{OUT}/pool/bin_*.parquet'))
    with Pool(4) as p:  # dense bins hold millions of rows; 12 GB RAM cap
        n_fp = sum(n for _, n in p.imap_unordered(add_fp, bins))
    dts = {'mass': np.float64, 'ik14': 'S14', 'src': np.uint8, 'taut': np.bool_}
    cols = {k: [] for k in dts}
    offsets = [0]
    for b in bins:
        t = pq.read_table(b, columns=list(dts))
        for k, dt in dts.items():  # convert per bin: 90M Python str objects at once would need ~6 GB
            cols[k].append(np.array(t[k].to_pylist(), dt) if k == 'ik14' else t[k].to_numpy().astype(dt))
        offsets.append(offsets[-1] + t.num_rows)
    sha = {}
    for k in dts:
        a = np.concatenate(cols.pop(k))
        np.save(f'{OUT}/{k}.npy', a)
        del a
        sha[f'{k}.npy'] = hashlib.sha256(open(f'{OUT}/{k}.npy', 'rb').read()).hexdigest()
    mass = np.load(f'{OUT}/mass.npy', mmap_mode='r')
    assert (np.diff(mass) >= 0).all(), 'pool not globally mass-sorted'
    m = json.load(open(f'{OUT}/manifest.json'))
    m['index'] = {'step': 'P4.2', 'fp_rows': int(n_fp), 'fp': 'morgan r2 2048 packed, curated rows only (D-024)',
                  'bins': [os.path.basename(b) for b in bins], 'bin_offsets': offsets, 'sha256': sha}
    m['sha256'] = {os.path.basename(b): hashlib.sha256(open(b, 'rb').read()).hexdigest() for b in bins}  # bins rewritten
    json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps({'rows': offsets[-1], 'fp_rows': int(n_fp)}))


if __name__ == '__main__':
    main()
