"""Sanity-check P2.5 shards: molecule sort/offsets, padding, finiteness, and re-featurize n random rows from source.
Usage: python scripts/check_shards.py [root=data/processed/v1] [n=50]"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

from casmi.data.splits import VAL
from casmi.features import featurize

root = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/v1'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
src = pq.read_table('data/train.parquet', columns=['normalized_smiles']) if n else None  # strings only, ~0.3 GB
pf = pq.ParquetFile('data/train.parquet')
starts = np.cumsum([0] + [pf.metadata.row_group(j).num_rows for j in range(pf.metadata.num_row_groups)])

for split in ('train',) + VAL:
    if not os.path.exists(f'{root}/{split}'):
        print(split, 'missing')
        continue
    L = lambda k: np.load(f'{root}/{split}/{k}.npy', mmap_mode='r')
    mi, off, npk = L('mol_idx'), L('mol_offsets'), L('npeaks')
    mols = pq.read_table(f'{root}/{split}/mols.parquet')
    assert (np.diff(mi) >= 0).all() and mi[0] == 0 and mi[-1] == len(mols) - 1
    assert off[-1] == len(mi) and (mi[off[:-1]] == np.arange(len(mols))).all()
    assert L('mol_fp').any(axis=1).mean() > 0.99
    pf16 = np.asarray(L('peakfeat')[:20000], np.float32)
    assert np.isfinite(pf16).all(), 'peakfeat float16 overflow/NaN'
    rng = np.random.default_rng(0)
    for i in rng.choice(len(mi), min(n, len(mi)), replace=False):
        r = int(L('src_row')[i])
        g = int(np.searchsorted(starts, r, side='right') - 1)
        row = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct']).slice(r - starts[g], 1).to_pylist()[0]
        f = featurize(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'], row['adduct'])
        k = len(f['mz'])
        assert k == npk[i] and np.array_equal(f['mz'], L('mz')[i, :k]) and not L('mz')[i, k:].any()
        assert np.allclose(f['peakfeat'], L('peakfeat')[i, :k], rtol=1e-2, atol=1e-2)
        assert np.array_equal(f['formulas'], L('formulas')[i, :len(f['formulas'])])
        assert mols['smiles'][int(mi[i])].as_py() is not None and src['normalized_smiles'][r].as_py() is not None
    print(split, 'spectra', len(mi), 'mols', len(mols), 'mean npeaks', round(float(np.mean(npk)), 1), 'ok')
