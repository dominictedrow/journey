"""P3.2 inference shared by the Kaggle notebook and the local dry run: test.parquet -> submission.csv.
Features are computed on the fly with the same casmi.features functions the shards used (parity)."""
import csv
import time

import numpy as np
import pyarrow.parquet as pq

from casmi.features.clean import P, clean_peaks
from casmi.features.formula import neutral_mass
from casmi.retrieval.library import Library, bin_rows

MAX_RANK = 25  # == casmi.metric.MAX_RANK; not imported because metric needs RDKit at import time
FALLBACK_PPM = (5.0, 20.0, 100.0)  # widen the window before giving up
FALLBACK_SMILES = 'C'


def load_test(path):
    t = pq.read_table(path, columns=['molecule_id', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct']).to_pydict()
    n = len(t['molecule_id'])
    mz, inten, npk = np.zeros((n, P), np.float32), np.zeros((n, P), np.float32), np.zeros(n, np.int64)
    nm = np.full(n, np.nan)
    for i in range(n):
        m, x, _ = clean_peaks(t['ms2_mzs'][i], t['ms2_normalized_intensities'][i], t['precursor_mz'][i])
        mz[i, :len(m)], inten[i, :len(m)], npk[i] = m, x, len(m)
        nm[i] = neutral_mass(t['precursor_mz'][i], t['adduct'][i])
    return t['molecule_id'], mz, inten, npk, nm


def predict(test_path, lib_path, out_path):
    """-> (timings dict, {molecule_id: [smiles...]})."""
    tm, t0 = {}, time.time()
    lib = Library.load(lib_path)
    tm['load_library'] = time.time() - t0
    t0 = time.time()
    ids, mz, inten, npk, nm = load_test(test_path)
    q = bin_rows(mz, inten, npk)
    tm['features'] = time.time() - t0
    t0 = time.time()
    rows = {}
    for mid in ids:
        rows.setdefault(mid, [])
    for i, mid in enumerate(ids):
        rows[mid].append(i)
    out = {}
    for mid, r in rows.items():
        m = nm[r][np.isfinite(nm[r])]
        mass = np.median(m) if len(m) else np.nan
        smi = []
        for ppm in FALLBACK_PPM:
            _, smi, _ = lib.rank(q[r], mass, ppm=ppm, top=MAX_RANK)
            if smi:
                break
        out[mid] = smi or [FALLBACK_SMILES]
    tm['rank'] = time.time() - t0
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['molecule_id', 'smiles'])
        for mid, smi in out.items():
            w.writerow([mid, ';'.join(smi)])
    return {k: round(v, 2) for k, v in tm.items()}, out


def validate(sub_path, test_path):
    """Format check: header, every test molecule_id exactly once, 1..25 ';'-joined non-empty SMILES."""
    want = set(pq.read_table(test_path, columns=['molecule_id'])['molecule_id'].to_pylist())
    with open(sub_path, newline='') as f:
        r = list(csv.reader(f))
    assert r[0] == ['molecule_id', 'smiles'], r[0]
    ids = [row[0] for row in r[1:]]
    assert len(ids) == len(set(ids)), 'duplicate molecule_id'
    assert set(ids) == want, f'missing {len(want - set(ids))}, extra {len(set(ids) - want)}'
    for row in r[1:]:
        g = row[1].split(';')
        assert 1 <= len(g) <= MAX_RANK and all(s.strip() for s in g), row[0]
    return len(ids)
