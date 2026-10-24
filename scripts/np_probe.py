"""D-035: V-np MRR of a saved top-K ranking (scripts/eval_encoder.py --top K ... --splits v_np) with every candidate vs
only train | COCONUT | LOTUS rows (the Submission #5 probe, casmi.kaggle.predict.P6_SRC).
Usage: python scripts/np_probe.py runs/<name>/top<K>_v_np.npz"""
import sys

import numpy as np

from casmi.retrieval.pool import Pool
from scripts.frag_features import mrr

t = np.load(sys.argv[1])
src = Pool('data/processed/pool_v1', ('src',)).src
blend, hit, rows = t['v_np_blend'], t['v_np_hit'], t['v_np_rows']
ok = rows >= 0
print('all candidates', round(mrr(np.where(ok, blend, -1e9), hit), 4))
cur = ok & (src[np.maximum(rows, 0)] & 7 > 0)
print('train|COCONUT|LOTUS only', round(mrr(np.where(cur, blend, -1e9), hit), 4))
print('truth in saved top-K', round(float(hit.any(1).mean()), 3))
# D-036: blend + lam * PubChem popularity, reranking inside the saved top-K only
pop = np.asarray(Pool('data/processed/pool_v1', ('popularity',)).popularity[np.maximum(rows, 0)], np.float32)
for lam in (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
    s = blend + lam * pop
    print(f'lam {lam}: all', round(mrr(np.where(ok, s, -1e9), hit), 4), 'curated', round(mrr(np.where(cur, s, -1e9), hit), 4))
