import os

import numpy as np
import pytest

from casmi.retrieval.pool import gather

POOL = 'data/processed/pool_v1'
VC = f'{POOL}/valcand'


@pytest.mark.skipif(not os.path.exists(f'{VC}/rows.npy'), reason='needs pool_v1 valcand')
def test_gather_matches_valcand():
    """Row-group reads must return the same fps as scripts/val_candidates.py's whole-bin reads."""
    rows, fp = np.load(f'{VC}/rows.npy', mmap_mode='r'), np.load(f'{VC}/fp.npy', mmap_mode='r')
    pick = np.sort(np.random.default_rng(0).choice(len(rows), 3000, replace=False))
    tm = {}
    _, got = gather(POOL, np.asarray(rows[pick]), workers=4, timings=tm)
    assert (got == fp[pick]).all()
    assert set(tm) == {'read', 'fp'}
