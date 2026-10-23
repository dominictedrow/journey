"""P7: replace the P4.5 library tier (every library structure above every non-library one, COS_T = 0) with one scale:
score = model blend + beta * library cosine for library rows, and only rows with cosine >= T forced to the top.
Sweeps (T, beta) on runs/<name>/top<K>.npz (scripts/eval_encoder.py --top). Ranks are exact within the saved set
(top-K by blend + LIB_K best library-cosine rows).
Usage: python scripts/lib_blend.py runs/<name>/top<K>.npz"""
import itertools
import json
import sys

import numpy as np

from casmi.data.splits import VAL
from scripts.frag_features import mrr

TS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 2.0)  # 0.0 = current tier, 2.0 = no forced tier
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
LIB_SHARE = 0.18  # LB-implied kind-1 share: 0.877 p + 0.03 (1 - p) = 0.184 -> p ~ 0.18 (secondary view only)


def main():
    t = np.load(sys.argv[1])
    res = {}
    for T, beta in itertools.product(TS, BETAS):
        r = {}
        for s in VAL:
            blend, c, hit, rows = t[f'{s}_blend'], t[f'{s}_libcos'], t[f'{s}_hit'], t[f'{s}_rows']
            lib = c >= 0
            score = np.where(lib & (c >= T), 10 + c, blend + beta * np.where(lib, c, 0))
            r[s] = round(mrr(np.where(rows >= 0, score, -1e9), hit), 4)
        r['weighted'] = round(np.mean([r['v_lib'], r['v_pool'], r['v_novel']]), 4)
        r['lb_mix'] = round(LIB_SHARE * r['v_lib'] + (1 - LIB_SHARE) * np.mean([r['v_pool'], r['v_novel']]), 4)
        res[f'{T},{beta}'] = r
    for key in ('weighted', 'lb_mix'):
        best = max(res, key=lambda k: res[k][key])
        print(f'best by {key}: T,beta = {best}', json.dumps(res[best]))
    print('current tier (0.0,0.0):', json.dumps(res['0.0,0.0']))
    for k, v in res.items():
        print(k, json.dumps(v))


if __name__ == '__main__':
    main()
