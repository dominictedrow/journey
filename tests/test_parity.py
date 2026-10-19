"""P2.7 train/inference parity: cached shard features == featurize() on the raw spectrum, for 200 random spectra.
Must pass before every submission (plan/CLAUDE.md). Skips when v1 shards are absent (e.g. on Kaggle)."""
import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from casmi.data.splits import VAL
from casmi.features import featurize

ROOT = 'data/processed/v1'
SRC = 'data/train.parquet'
N = {'train': 120, **{s: 20 for s in VAL}}
COLS = ['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'instrument_type', 'ionization_mode', 'ingest_lib']
META = {'adduct': 'adduct', 'instrument_type': 'instrument', 'ionization_mode': 'polarity', 'ingest_lib': 'lib'}

pytestmark = pytest.mark.skipif(not os.path.exists(f'{ROOT}/train/src_row.npy'), reason='v1 shards not built')


def samples():
    rng = np.random.default_rng(2026)
    out = []
    for split, n in N.items():
        src = np.load(f'{ROOT}/{split}/src_row.npy', mmap_mode='r')
        out += [(split, int(i), int(src[i])) for i in rng.choice(len(src), n, replace=False)]
    return out


def raw_rows(rows):
    pf = pq.ParquetFile(SRC)
    starts = np.cumsum([0] + [pf.metadata.row_group(j).num_rows for j in range(pf.metadata.num_row_groups)])
    got = {}
    by_rg = {}
    for r in rows:
        by_rg.setdefault(int(np.searchsorted(starts, r, side='right') - 1), []).append(r)
    for g, rs in by_rg.items():
        t = pf.read_row_group(g, columns=COLS)
        for r in rs:
            got[r] = t.slice(r - starts[g], 1).to_pylist()[0]
    return got


def test_cached_equals_on_the_fly():
    s = samples()
    raw = raw_rows([r for _, _, r in s])
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    for split, i, r in s:
        L = lambda k: np.load(f'{ROOT}/{split}/{k}.npy', mmap_mode='r')[i]
        row = raw[r]
        f = featurize(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'], row['adduct'])
        n, k = len(f['mz']), len(f['formulas'])
        ctx = f'{split} shard row {i} src row {r}'
        assert L('npeaks') == n and L('nformulas') == k, ctx
        assert np.array_equal(L('mz')[:n], f['mz']) and not L('mz')[n:].any(), ctx
        assert np.allclose(L('inten')[:n], f['inten'], atol=1e-3), ctx  # float16 storage
        assert np.allclose(L('peakfeat')[:n], f['peakfeat'], rtol=1e-2, atol=1e-2), ctx
        assert np.array_equal(L('iso'), f['iso']) and L('prec') == row['precursor_mz'], ctx
        assert L('neutral_mass') == f['neutral_mass'] or np.isnan(f['neutral_mass']), ctx
        assert np.array_equal(L('formulas')[:k], f['formulas']), ctx
        for name in ('formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe'):
            assert np.allclose(L(name)[:k], f[name], equal_nan=True), (ctx, name)
        for col, name in META.items():
            assert L(name) == vocab[col].get(row[col], 0), (ctx, name)
