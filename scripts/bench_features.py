"""P2.4 benchmark: featurize() ms/spectrum on all test spectra + first N train spectra (target <= 10 ms)."""
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from casmi.features import featurize  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
COLS = ['precursor_mz', 'adduct', 'ms2_mzs', 'ms2_normalized_intensities']
featurize([100.0, 200.0], [1.0, 0.5], 250.0, '[M+H]+')  # numba warm-up
for name, path in (('test', 'data/test.parquet'), ('train', 'data/train.parquet')):
    d = next(pq.ParquetFile(path).iter_batches(batch_size=N, columns=COLS)).to_pydict()
    ts, nf, sub = [], [], []
    for p, a, m, i in zip(d['precursor_mz'], d['adduct'], d['ms2_mzs'], d['ms2_normalized_intensities']):
        t0 = time.perf_counter()
        o = featurize(m, i, p, a)
        ts.append(time.perf_counter() - t0)
        nf.append(len(o['formulas']))
        if len(o['mz']):
            sub.append(o['peakfeat'][:, 6].mean())
    ts = 1e3 * np.array(ts)
    print(f'[{name}] n={len(ts)} ms/spectrum mean {ts.mean():.2f} p50 {np.median(ts):.2f} p99 {np.percentile(ts, 99):.1f} max {ts.max():.0f} '
          f'| no formula {np.mean(np.array(nf) == 0):.3f} | peaks with subformula {np.mean(sub):.3f}', flush=True)
