"""P2.4 check: true-formula rank from rank_formulas on V-pool enveda query spectra (honest: prior is train-only).
Usage: python scripts/eval_formula.py [n_spectra]"""
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from casmi.features.clean import iso_features  # noqa: E402
from casmi.features.formula import formula_vector, neutral_mass, rank_formulas  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'v_pool'
sp = pq.read_table('data/processed/v1/splits.parquet', columns=['ik14', 'split']).to_pydict()
val = {k for k, s in zip(sp['ik14'], sp['split']) if s == SPLIT}
st = pq.read_table('data/processed/v1/structures.parquet', columns=['smiles', 'ik14']).to_pydict()
ik = dict(zip(st['smiles'], st['ik14']))

rows = []
cols = ['normalized_smiles', 'ingest_lib', 'adduct', 'precursor_mz', 'molecular_formula', 'ms2_mzs', 'ms2_normalized_intensities']
for b in pq.ParquetFile('data/train.parquet').iter_batches(batch_size=65536, columns=cols):
    d = b.to_pydict()
    for i, smi in enumerate(d['normalized_smiles']):
        if ik[smi] in val and d['ingest_lib'][i].startswith('enveda'):
            rows.append({c: d[c][i] for c in cols})
    if len(rows) >= N:
        break
rows = rows[:N]

for sigma, w_iso in ((2.0, 0.0),):  # S06 sweep on train: sigma 1–5, w_iso 0–1; this was best
    ranks, t = [], 0.0
    for r in rows:
        true = formula_vector(r['molecular_formula'])
        m = neutral_mass(r['precursor_mz'], r['adduct'])
        iso = iso_features(np.asarray(r['ms2_mzs']), np.asarray(r['ms2_normalized_intensities']), r['precursor_mz'])
        t0 = time.perf_counter()
        cnt, _, _ = rank_formulas(m, iso, sigma_ppm=sigma, w_iso=w_iso, top=1000)
        t += time.perf_counter() - t0
        hit = np.where((cnt == true).all(1))[0] if true is not None and len(cnt) else []
        ranks.append(hit[0] + 1 if len(hit) else 10 ** 6)
    ranks = np.array(ranks)
    print(f'sigma={sigma} w_iso={w_iso} n={len(ranks)} top1 {np.mean(ranks <= 1):.3f} top5 {np.mean(ranks <= 5):.3f} '
          f'top16 {np.mean(ranks <= 16):.3f} found {np.mean(ranks < 10 ** 6):.3f} | {1e3 * t / len(ranks):.2f} ms/spectrum', flush=True)
