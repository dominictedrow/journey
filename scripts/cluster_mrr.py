"""P6.4 clustering analysis (Dom's K-means idea): K-means on the true val molecules' candidate-side embeddings
(Contrastive.mol over fp + formula), then MRR@25 / recall / median rank per cluster under the best weighted blend.
Kind-2 splits only (v_pool, v_scaf, v_novel): v_lib is decided by the library tier, not the model.
Input runs/<name>/eval_detail.npz + eval_pool.json (scripts/eval_encoder.py). Output reports/p6_clusters_<name>.{md,json}.
Usage: python scripts/cluster_mrr.py runs/<name> [k=20]"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

ROOT = 'data/processed/v1'
SPLITS = ('v_pool', 'v_scaf', 'v_novel')


def main():
    run, k = sys.argv[1].rstrip('/'), int(sys.argv[2]) if len(sys.argv) > 2 else 20
    name = run.rsplit('/', 1)[-1]
    d = np.load(f'{run}/eval_detail.npz')
    w = json.load(open(f'{run}/eval_pool.json'))['weighted']
    best = max(w, key=w.get)
    g = list(d['grid']).index(best)
    emb = np.concatenate([d[f'{s}_molemb'] for s in SPLITS])
    rank = np.concatenate([d[f'{s}_ranks'][:, g] for s in SPLITS])
    nwin = np.concatenate([d[f'{s}_nwin'] for s in SPLITS])
    split = np.concatenate([[s] * len(d[f'{s}_nwin']) for s in SPLITS])
    smi, form = [], []
    for s in SPLITS:
        t = pq.read_table(f'{ROOT}/{s}/mols.parquet', columns=['smiles', 'formula']).to_pydict()
        smi += t['smiles']
        form += t['formula']
    km = KMeans(k, n_init=10, random_state=0).fit(emb)
    rr = np.where((rank >= 0) & (rank < 25), 1 / (rank + 1), 0.0)
    rows = []
    for c in range(k):
        m = km.labels_ == c
        r = rank[m]
        found = r >= 0
        rep = np.flatnonzero(m)[np.argmin(((emb[m] - km.cluster_centers_[c]) ** 2).sum(1))]
        rows.append({'cluster': c, 'n': int(m.sum()), 'mrr25': round(float(rr[m].mean()), 4),
                     'recall25': round(float(((r >= 0) & (r < 25)).mean()), 3),
                     'recall100': round(float(((r >= 0) & (r < 100)).mean()), 3),
                     'in_window': round(float(found.mean()), 3),
                     'median_rank': int(np.median(r[found])) if found.any() else -1,
                     'median_window': int(np.median(nwin[m])), 'rep_smiles': smi[rep], 'rep_formula': form[rep],
                     'split_mix': {s: int((split[m] == s).sum()) for s in SPLITS}})
    rows.sort(key=lambda x: x['mrr25'])
    found = rank >= 0
    summary = {'run': name, 'blend': best, 'k': k, 'n': int(len(rank)), 'mrr25': round(float(rr.mean()), 4),
               'in_window': round(float(found.mean()), 3), 'median_rank': int(np.median(rank[found])),
               'median_window': int(np.median(nwin)),
               'recall': {n: round(float(((rank >= 0) & (rank < n)).mean()), 3) for n in (1, 10, 25, 100, 1000)}}
    json.dump({'summary': summary, 'clusters': rows}, open(f'reports/p6_clusters_{name}.json', 'w'), indent=1)
    md = [f'# P6.4 clusters — {name} (blend c,t,a,b = {best}, K = {k})', '',
          f'Kind-2 val molecules: {summary["n"]}; MRR@25 {summary["mrr25"]}; truth in +-5 ppm window '
          f'{summary["in_window"]}; median rank {summary["median_rank"]} of median window {summary["median_window"]}; '
          f'recall@1/10/25/100/1000 = {list(summary["recall"].values())}', '',
          '| cluster | n | MRR@25 | R@25 | R@100 | median rank | median window | representative |',
          '|---|---|---|---|---|---|---|---|']
    md += [f'| {r["cluster"]} | {r["n"]} | {r["mrr25"]} | {r["recall25"]} | {r["recall100"]} | {r["median_rank"]} | '
           f'{r["median_window"]} | `{r["rep_formula"]}` {r["rep_smiles"][:60]} |' for r in rows]
    open(f'reports/p6_clusters_{name}.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
