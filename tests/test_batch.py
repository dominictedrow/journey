"""Inference batches (casmi.data.batch from raw rows) == loader batches (from shards) for the same val spectra."""
import json

import numpy as np
import pytest
import torch

from casmi.data.batch import featurize_rows, to_batch
from casmi.data.loader import make_loader
from tests.test_parity import ROOT, pytestmark, raw_rows  # noqa: F401  (skip rule shared)

USED = ('mz', 'inten', 'peakfeat', 'npeaks', 'prec', 'neutral_mass', 'ce', 'iso', 'adduct', 'instrument', 'polarity',
        'formulas', 'nformulas', 'formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe', 'seg')


def test_raw_batch_matches_loader():
    b = next(iter(make_loader(ROOT, 'v_pool', workers=0, shuffle=False, max_spec_per_mol=None, spec_per_batch=64)))
    off = np.load(f'{ROOT}/v_pool/mol_offsets.npy')
    src = np.load(f'{ROOT}/v_pool/src_row.npy', mmap_mode='r')
    spec = np.concatenate([np.arange(off[m], off[m + 1]) for m in b['mol_idx'].tolist()])
    raw = raw_rows([int(src[i]) for i in spec])
    cols = ('collision_energy_ev',)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('data/train.parquet')
    starts = np.cumsum([0] + [pf.metadata.row_group(j).num_rows for j in range(pf.metadata.num_row_groups)])
    for i in spec:  # raw_rows reads only the parity columns; add collision energy
        r = int(src[i])
        g = int(np.searchsorted(starts, r, side='right') - 1)
        raw[r].update(pf.read_row_group(g, columns=list(cols)).slice(r - starts[g], 1).to_pylist()[0])
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    arrs = featurize_rows([raw[int(src[i])] for i in spec], vocab)
    seg = np.repeat(np.arange(len(b['mol_idx'])), np.diff(off)[b['mol_idx'].numpy()])
    got = to_batch(arrs, np.arange(len(spec)), seg)
    for k in USED:
        x, y = b[k], got[k]
        assert x.shape == y.shape, k
        if x.dtype.is_floating_point:
            assert torch.allclose(x.float(), y.float(), rtol=1e-2, atol=1e-2, equal_nan=True), k
        else:
            assert torch.equal(x, y), k
