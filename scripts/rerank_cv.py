"""P7.3 (D-040): LightGBM LambdaRank reranker over a saved V-np top-K (scripts/eval_encoder.py --top ... --splits v_np)
and its fragment features (scripts/frag_features.py). 5-fold CV grouped by molecule -> out-of-fold MRR@25 on full and
low-pop V-np (D-038), next to the hand-tuned blend + 0.5 * expl. Then fits on all of V-np -> runs/<name>/rerank.txt.
--nopop drops the pop features (the blend still holds lam * pop): V-np is popularity-biased (D-038).
Usage: python scripts/rerank_cv.py runs/<name>/top<K>_v_np_<tag>.npz [rounds=100] [--nopop]"""
import sys

import lightgbm as lgb
import numpy as np

from casmi.retrieval.pool import Pool
from scripts.frag_features import mrr

POOL = 'data/processed/pool_v1'
NAMES = ('blend', 'blend_gap', 'pop', 'pop_gap', 'expl', 'expl1', 'peak_frac', 'log_nfrag', 'iso')
PARAMS = {'objective': 'lambdarank', 'metric': 'map', 'eval_at': [25], 'learning_rate': 0.05, 'num_leaves': 15,
          'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'verbose': -1, 'seed': 0}


def features(rows, blend, frag):
    """[n_mol, K, len(NAMES)]; *_gap = value minus the molecule's best, so trees see within-molecule position."""
    ok = rows >= 0
    pop = np.where(ok, np.asarray(Pool(POOL, ('popularity',)).popularity[np.maximum(rows, 0)], np.float32), 0)
    b = np.where(ok, blend, np.nan)
    gap = lambda x: x - np.nanmax(np.where(ok, x, np.nan), 1, keepdims=True)
    return np.concatenate([np.stack([b, gap(b), pop, gap(pop)], -1), frag], -1).astype(np.float32)


def fit(x, y, ok, mols, rounds):
    xs = np.concatenate([x[j][ok[j]] for j in mols])
    ys = np.concatenate([y[j][ok[j]] for j in mols]).astype(int)
    return lgb.train(PARAMS, lgb.Dataset(xs, ys, group=[int(ok[j].sum()) for j in mols]), rounds)


def predict(model, x, ok):
    s = np.full(ok.shape, -1e9)
    for j in range(len(x)):
        s[j, ok[j]] = model.predict(x[j][ok[j]])
    return s


def main():
    top = np.load(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    frag = np.load(sys.argv[1].replace('/top', '/frag'))['v_np_feats']
    rows, blend, hit = top['v_np_rows'], top['v_np_blend'], top['v_np_hit']
    ok = rows >= 0
    x = features(rows, blend, frag)
    names = NAMES
    if '--nopop' in sys.argv:
        x, names = np.delete(x, [2, 3], -1), tuple(n for n in NAMES if not n.startswith('pop'))
    low = np.load(f'{POOL}/valcand_v_np/v_np_pop_pct.npy') < 0.9
    hand = np.where(ok, blend + 0.5 * frag[..., 0], -1e9)
    print('hand blend+0.5expl: full', round(mrr(hand, hit), 4), 'low-pop', round(mrr(hand[low], hit[low]), 4))
    folds = np.random.default_rng(0).permutation(len(rows)) % 5
    for r in (25, 50, 100, 200, 400):
        oof = np.full(ok.shape, -1e9)
        for f in range(5):
            m = fit(x, hit, ok, np.flatnonzero(folds != f), r)
            te = np.flatnonzero(folds == f)
            oof[te] = predict(m, x[te], ok[te])
        print(f'lambdarank {r} rounds OOF: full', round(mrr(oof, hit), 4), 'low-pop', round(mrr(oof[low], hit[low]), 4),
              flush=True)
    m = fit(x, hit, ok, np.arange(len(rows)), rounds)
    m.save_model(sys.argv[1].rsplit('/', 1)[0] + '/rerank.txt')
    print('gain', dict(zip(names, np.round(m.feature_importance('gain') / m.feature_importance('gain').sum(), 3))))


if __name__ == '__main__':
    main()
