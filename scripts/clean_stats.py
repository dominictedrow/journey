"""P2.2 check: run casmi.features.clean over all of train and test; report drops, peaks kept, iso rates, speed."""
import sys
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from casmi.features.clean import MAX_PPM, MIN_PEAKS, PREC_RANGE, clean_peaks  # noqa: E402

COLS = ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']


def run(path, train):
    drops, npk, iso, n, t_clean = Counter(), [], np.zeros(3), 0, 0.0
    cols = COLS + (['precursor_error_ppm', 'adduct'] if train else [])
    for b in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=cols):
        prec = b.column('precursor_mz').to_numpy()
        mzc, itc = b.column('ms2_mzs'), b.column('ms2_normalized_intensities')
        off, mzv, itv = mzc.offsets.to_numpy(), mzc.values.to_numpy(), itc.values.to_numpy()
        ppm = b.column('precursor_error_ppm').to_numpy(zero_copy_only=False) if train else None
        t0 = time.perf_counter()
        for r in range(len(prec)):
            m, _, f = clean_peaks(mzv[off[r]:off[r + 1]], itv[off[r]:off[r + 1]], prec[r])
            n += 1
            iso += f[3:]
            if train:
                if not PREC_RANGE[0] <= prec[r] <= PREC_RANGE[1]:
                    drops['prec_range'] += 1
                    continue
                if abs(ppm[r]) > MAX_PPM:  # nan compares False -> kept
                    drops['ppm'] += 1
                    continue
                if len(m) < MIN_PEAKS:
                    drops['min_peaks'] += 1
                    continue
            npk.append(len(m))
        t_clean += time.perf_counter() - t0
        if train:
            drops['adduct_2M_kept'] += sum(a is not None and a.startswith('[2M') for a in b.column('adduct').to_pylist())
    npk = np.array(npk)
    name = 'TRAIN' if train else 'TEST'
    print(f'[{name}] rows {n} kept {len(npk)} ({100 * len(npk) / n:.2f}%) drops {dict(drops)}')
    print(f'[{name}] peaks kept p5/p50/p95/max {np.percentile(npk, [5, 50, 95]).tolist()} {npk.max()} | at P cap {100 * (npk == 128).mean():.1f}% | <{MIN_PEAKS}: {(npk < MIN_PEAKS).sum()}')
    print(f'[{name}] iso present M/M+1/M+2 % {np.round(100 * iso / n, 1).tolist()} | {1e6 * t_clean / n:.0f} us/spectrum', flush=True)


run('data/test.parquet', False)
run('data/train.parquet', True)
