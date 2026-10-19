"""P2.4: formula log-priors from train-split structures only -> casmi/features/formula_prior.npz.
count[7, 21]: log P(count=k) for N P S F Cl Br I; rdbe/hc/oc: log P(bin) (bins in casmi.features.formula.prior_bins)."""
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from casmi.features.formula import CORE, CORE_MAX, N_BINS, PRIOR_PATH, formula_vector, prior_bins  # noqa: E402

sp = pq.read_table('data/processed/v1/splits.parquet', columns=['ik14', 'split']).to_pydict()
train = {k for k, s in zip(sp['ik14'], sp['split']) if s == 'train'}
st = pq.read_table('data/processed/v1/structures.parquet', columns=['ik14', 'formula']).to_pydict()
seen, vecs = set(), []
for k, f in zip(st['ik14'], st['formula']):
    if k in train and k not in seen and (v := formula_vector(f)) is not None and v[0] > 0:
        seen.add(k)
        vecs.append(v)
v = np.stack(vecs)
count = np.full((len(CORE), CORE_MAX.max() + 1), -30.0, np.float32)
for j in range(len(CORE)):
    h = np.bincount(np.minimum(v[:, CORE[j]], CORE_MAX[j]), minlength=CORE_MAX[j] + 1) + 1.0  # add-one smoothing
    count[j, :CORE_MAX[j] + 1] = np.log(h / h.sum())
out = {'count': count}
for name, b in zip(('rdbe', 'hc', 'oc'), prior_bins(v)):
    h = np.bincount(b, minlength=N_BINS[name]) + 1.0
    out[name] = np.log(h / h.sum()).astype(np.float32)
np.savez(PRIOR_PATH, **out)
print(len(vecs), 'train formulas;', {k: np.round(x.ravel()[:8], 2).tolist() for k, x in out.items()})
