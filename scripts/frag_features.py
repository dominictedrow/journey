"""P7.2 fragment + isotope features for each val molecule's top-K candidates (scripts/eval_encoder.py --top), and a
diagnostic: MRR@25 when the model score gets + w * feature for non-library-tier candidates. Diagnostic only; P7.3
fits on a separate fold.
D-039: the diagnostic ranks the no-tier blend (as Kaggle since D-034); v_np also reports the low-pop subset (D-038).
Usage: python scripts/frag_features.py runs/<name>/top<K>.npz [workers=14] [--splits v_np] -> runs/<name>/frag<K>.npz"""
import json
import sys
import time
from multiprocessing import Pool as Workers

import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger

from casmi.data.splits import VAL
from casmi.features.formula import fkey_counts, iso_score
from casmi.rerank.fragment import MAX_SPEC, mol_feats, spectra
from casmi.retrieval.pool import Pool, gather

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
FEATS = ('expl', 'expl1', 'peak_frac', 'log_nfrag', 'iso')  # mol_feats columns + iso
W = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
RDLogger.DisableLog('rdApp.*')


def mrr(score, hit):
    """MRR@25 (first-correct-hit rule) over molecules; rows with no hit in top-K score 0."""
    rr = np.zeros(len(score))
    for j in range(len(score)):
        if hit[j].any():
            r = int((score[j] > score[j][hit[j]].max()).sum())
            rr[j] = 1 / (r + 1) if r < 25 else 0.0
    return float(rr.mean())


def main():
    top = np.load(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 14
    splits = sys.argv[sys.argv.index('--splits') + 1].split(',') if '--splits' in sys.argv else VAL
    neg_id = json.load(open(f'{ROOT}/vocab.json'))['ionization_mode']['negative']
    pool, out, res = Pool(POOL), {}, {}
    for split in splits:
        t0 = time.time()
        d = f'{ROOT}/{split}'
        rows = top[f'{split}_rows']
        off = np.load(f'{d}/mol_offsets.npy')
        mz, inten, npk, pol, iso = (np.load(f'{d}/{k}.npy', mmap_mode='r') for k in
                                    ('mz', 'inten', 'npeaks', 'polarity', 'iso'))
        uniq = np.unique(rows[rows >= 0])
        smiles, _ = gather(POOL, uniq, workers, fps=False)
        smi_of = dict(zip(uniq.tolist(), smiles))
        jobs = []
        for j in range(len(rows)):
            jobs.append(([smi_of.get(r) for r in rows[j].tolist()],
                         spectra(mz, inten, npk, pol, range(off[j], off[j + 1]), neg_id)))
        with Workers(workers) as wk:
            f = np.stack(wk.map(mol_feats, jobs, chunksize=4))
        cnt = fkey_counts(pool.fkey[np.maximum(rows, 0)])
        isof = np.zeros(rows.shape, np.float32)
        for j in range(len(rows)):
            s = range(off[j], min(off[j + 1], off[j] + MAX_SPEC))
            isof[j] = np.mean([iso_score(cnt[j], iso[i]) for i in s], 0)
        feats = np.concatenate([f, isof[..., None]], -1)
        feats[rows < 0] = 0
        out[f'{split}_feats'] = feats
        score, hit = np.where(rows >= 0, top[f'{split}_blend'], -1e9), top[f'{split}_hit']
        low = np.load(f'{POOL}/valcand_v_np/v_np_pop_pct.npy') < 0.9 if split == 'v_np' else None
        r = {'base': round(mrr(score, hit), 4), 'truth_in_topK': round(float(hit.any(1).mean()), 3)}
        if low is not None:
            r['base_lowpop'] = round(mrr(score[low], hit[low]), 4)
        for fi, name in enumerate(FEATS):
            x = feats[..., fi]
            r[name] = {str(w): round(mrr(score + w * x, hit), 4) for w in W[1:]}
            if low is not None:
                r[name]['lowpop'] = {str(w): round(mrr((score + w * x)[low], hit[low]), 4) for w in W[1:]}
            h = hit.any(1)  # truth vs mean of the other top-K candidates, molecules with truth in top-K
            r[name]['truth_minus_mean'] = round(float(np.mean([x[j][hit[j]].max() - x[j][~hit[j] & (rows[j] >= 0)].mean()
                                                               for j in np.flatnonzero(h)])), 4) if h.any() else None
        res[split] = r
        print(split, f'{time.time() - t0:.0f}s', json.dumps(r), flush=True)
    np.savez(sys.argv[1].replace('/top', '/frag'), feats=np.array(FEATS), **out)


if __name__ == '__main__':
    main()
