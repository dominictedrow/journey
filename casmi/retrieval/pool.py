"""P4.5 pool ranking: candidates = pool_v1 rows within +-PPM of the query neutral mass. Rank tiers:
1. structures with library spectra and cosine >= COS_T (by cosine);
2. everything else by formula agreement with the query's ranked formula candidates, then curated NP source
   (COCONUT/LOTUS), COCONUT annotation level, then lowest PubChem CID (older = better-studied compound)."""
import numpy as np

from casmi.features.formula import formula_key

PPM = 5.0
COS_T = 0.0  # reports/eval_pool.json: any library structure above every library-less one (V-lib 0.877)
TRAIN, CURATED_EXT = 1, 6  # src bits: train 1, coconut 2, lotus 4, pubchem 8
ARRAYS = ('mass', 'ik14', 'src', 'fkey', 'cid', 'level', 'npl')


class Pool:
    def __init__(self, root, arrays=ARRAYS):
        self.root = root
        for k in arrays:
            setattr(self, k, np.load(f'{root}/{k}.npy', mmap_mode='r'))

    def window(self, mass, ppm=PPM):
        tol = mass * ppm * 1e-6
        return np.searchsorted(self.mass, mass - tol), np.searchsorted(self.mass, mass + tol, side='right')


def gather(root, rows, workers=None, timings=None, fps=True):
    """Sorted unique pool rows -> (smiles list, packed fp uint8[n, 256]): stored fp for curated rows, RDKit
    (casmi.features.fp, same generator) for PubChem rows. Reads only the parquet row groups holding the rows (a 5 ppm
    window spans 1-2 of ~65k rows; whole-bin reads took ~1 h on Kaggle's /kaggle/input). timings: dict += read/fp s.
    fps=False: SMILES only (fp stays zero, no RDKit work)."""
    import json
    import time
    from multiprocessing import Pool as Workers

    import pyarrow as pa
    import pyarrow.parquet as pq

    from casmi.features.fp import fingerprints
    idx = json.load(open(f'{root}/manifest.json'))['index']
    boff = np.array(idx['bin_offsets'])
    which = np.searchsorted(boff, rows, side='right') - 1
    smiles, fp = [None] * len(rows), np.zeros((len(rows), 256), np.uint8)
    tm = {'read': 0.0, 'fp': 0.0}
    with Workers(workers) as wk:
        for bi in np.unique(which):
            t0 = time.time()
            sel = np.flatnonzero(which == bi)
            pf = pq.ParquetFile(f'{root}/pool/{idx["bins"][bi]}')
            rg_off = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
            local = rows[sel] - boff[bi]
            rg = np.searchsorted(rg_off, local, side='right') - 1
            need = np.unique(rg)
            cat_off = np.cumsum(np.r_[0, rg_off[need + 1] - rg_off[need]])  # need[k] starts at cat_off[k] once read
            pos = local - rg_off[rg] + cat_off[np.searchsorted(need, rg)]
            t = pf.read_row_groups(need.tolist(), columns=['smiles', 'fp']).take(pa.array(pos))
            tm['read'] += time.time() - t0
            t0 = time.time()
            smi, stored = t['smiles'].to_pylist(), t['fp'].to_pylist()
            todo = [i for i, x in enumerate(stored) if x is None]
            for i, x in enumerate(stored):
                smiles[sel[i]] = smi[i]
                if x is not None:
                    fp[sel[i]] = np.frombuffer(x, np.uint8)
            chunks = [[smi[i] for i in todo[k:k + 5000]] for k in range(0, len(todo), 5000)]
            if chunks and fps:
                fp[sel[todo]] = np.concatenate(wk.map(fingerprints, chunks))
            tm['fp'] += time.time() - t0
    if timings is not None:
        for k, v in tm.items():
            timings[k] = timings.get(k, 0.0) + round(v, 2)
    return smiles, fp


def formula_score(formulas, nformulas):
    """Query spectra of one molecule (formulas int16[N,K,10], nformulas[N]) -> {formula_key: mean_s 1/(1+rank_s)}."""
    keys = formula_key(formulas)
    d = {}
    for s, n in enumerate(nformulas):
        seen = set()
        for r in range(int(n)):
            k = int(keys[s, r])
            if k not in seen:
                seen.add(k)
                d[k] = d.get(k, 0.0) + 1 / (1 + r) / len(nformulas)
    return d


def rank(pool, mass, fscores, lib_cos, ppm=PPM, top=25, strip_train=None, cos_t=COS_T):
    """-> pool row indices (best first). lib_cos: {ik14 bytes: cosine} for window structures with library spectra.
    strip_train: ik14 bytes whose train source bit is ignored (local val: val structures are not in the test-time
    train set unless the library has their spectra)."""
    if not np.isfinite(mass):
        return np.array([], np.int64)
    lo, hi = pool.window(mass, ppm)
    rows = np.arange(lo, hi)
    keys, src = pool.ik14[lo:hi], pool.src[lo:hi].copy()
    cos = np.array([lib_cos.get(k, -1.0) for k in keys], np.float32)  # -1 = no library spectra
    if strip_train is not None:
        src[np.isin(keys, strip_train)] &= ~np.uint8(TRAIN)
        keep = (src > 0) | (cos >= 0)
        rows, keys, src, cos = rows[keep], keys[keep], src[keep], cos[keep]
    fk = pool.fkey[rows]
    u, inv = np.unique(fk, return_inverse=True)
    fs = np.array([fscores.get(int(x), 0.0) for x in u], np.float32)[inv]
    tier = cos >= cos_t
    curated = ((src & CURATED_EXT) > 0) | (cos >= 0)
    cid = pool.cid[rows].astype(np.int64)
    cid[cid == 0] = 1 << 40
    o = np.lexsort((cid, -pool.level[rows], ~curated, -fs, -np.where(tier, cos, 0), ~tier))[:top]
    return rows[o]
