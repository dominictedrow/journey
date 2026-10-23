"""P7.1 benchmark of casmi.rerank.fragment on real val windows: per-candidate runtime (projected to top-50 x 400 on
Kaggle) and a sanity check that truth explains more MS2 intensity than random same-window decoys.
Usage: python scripts/bench_fragment.py [n_mols=60] [per_mol=50] [workers=4] -> logs/bench_fragment.log"""
import json
import sys
import time
from multiprocessing import Pool as Workers

import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger

from casmi.rerank.fragment import explain, fragments
from casmi.retrieval.pool import gather

ROOT, POOL, SPLIT = 'data/processed/v1', 'data/processed/pool_v1', 'v_pool'
RDLogger.DisableLog('rdApp.*')


def work(smi):
    t0 = time.perf_counter()
    m, c = fragments(smi)
    return m, time.perf_counter() - t0


def main():
    n_mols, per, workers = (int(x) for x in (sys.argv[1:] + ['60', '50', '4'][len(sys.argv) - 1:]))
    rng = np.random.default_rng(2026)
    d = f'{ROOT}/{SPLIT}'
    off, wins = np.load(f'{d}/mol_offsets.npy'), np.load(f'{POOL}/valcand/{SPLIT}_windows.npy')
    smiles = pq.read_table(f'{d}/mols.parquet', columns=['smiles'])['smiles'].to_pylist()
    mz, inten, npk = (np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks'))
    pol = np.load(f'{d}/polarity.npy', mmap_mode='r')
    neg_id = json.load(open(f'{ROOT}/vocab.json'))['ionization_mode']['negative']
    mols = rng.choice(np.flatnonzero(wins[:, 1] - wins[:, 0] > per), n_mols, replace=False)
    picks = {j: np.sort(rng.choice(np.arange(*wins[j]), per, replace=False)) for j in mols}
    t0 = time.time()
    rows = np.unique(np.concatenate(list(picks.values())))
    cand, _ = gather(POOL, rows, workers)
    t_gather = time.time() - t0
    jobs = [smiles[j] for j in mols] + cand
    t0 = time.time()
    with Workers(workers) as wk:
        out = wk.map(work, jobs, chunksize=16)
    wall = time.time() - t0
    frag = dict(zip(range(len(jobs)), out))
    per_cand = np.array([t for _, t in out])
    nfrag = np.array([len(m) for m, _ in out])
    pos = {r: n_mols + i for i, r in enumerate(rows)}
    better, gap = [], []
    for i, j in enumerate(mols):
        s = off[j]  # first spectrum of the molecule
        k = int(npk[s])
        x, y = np.asarray(mz[s, :k], np.float64), np.asarray(inten[s, :k], np.float64)
        ch = -1 if pol[s] == neg_id else 1
        t = explain(x, y, frag[i][0], ch)[0]
        dec = np.array([explain(x, y, frag[pos[r]][0], ch)[0] for r in picks[j]])
        better.append((dec < t).mean())
        gap.append(t - dec.mean())
    proj_h = per_cand.mean() * 50 * 400 / 4 / 3600  # Kaggle CPU: 4 cores, assume per-core speed ~ local
    res = {'step': 'P7.1', 'n_cand': len(jobs), 'workers': workers, 'gather_s': round(t_gather, 1),
           'wall_s': round(wall, 1), 'ms_per_cand': {'mean': round(1e3 * per_cand.mean(), 1),
           'p50': round(1e3 * np.median(per_cand), 1), 'p99': round(1e3 * np.quantile(per_cand, 0.99), 1),
           'max': round(1e3 * per_cand.max(), 1)},
           'frags_per_cand': {'mean': round(nfrag.mean(), 1), 'max': int(nfrag.max())},
           'proj_kaggle_h_top50x400_4cores': round(proj_h, 2),
           'truth_beats_decoy_frac': round(float(np.mean(better)), 3),
           'explained_gap_mean': round(float(np.mean(gap)), 3)}
    print(json.dumps(res))


if __name__ == '__main__':
    main()
