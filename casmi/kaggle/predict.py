"""P3.2 inference shared by the Kaggle notebook and the local dry run: test.parquet -> submission.csv.
Features are computed on the fly with the same casmi.features functions the shards used (parity)."""
import csv
import time

import numpy as np
import pyarrow.parquet as pq

from casmi.features.clean import P, clean_peaks
from casmi.features.formula import neutral_mass
from casmi.retrieval.library import Library, bin_rows

MAX_RANK = 25  # == casmi.metric.MAX_RANK; not imported because metric needs RDKit at import time
LIB_T = 0.7  # D-034: only library hits this close are forced above the model ranking (scripts/lib_blend.py)
P6_LIB_T, P6_BLEND = 2.0, (1.0, 4.0, 0.05)  # D-034 #4b: no forced tier; c * cosine + t * Tanimoto + b * curated/library
P6_POP = 0.05  # D-036: + P6_POP * pool popularity.npy (ln1p substances + ln1p PubMed); 0 = off, needs the file
P6_SRC = 7  # D-035 probe: P6 ranks only train | COCONUT | LOTUS pool rows (src bits 1|2|4); 0 = every row incl. PubChem-only
FALLBACK_PPM = (5.0, 20.0, 100.0)  # widen the window before giving up
FALLBACK_SMILES = 'C'


def load_test(path):
    t = pq.read_table(path, columns=['molecule_id', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct']).to_pydict()
    n = len(t['molecule_id'])
    mz, inten, npk = np.zeros((n, P), np.float32), np.zeros((n, P), np.float32), np.zeros(n, np.int64)
    nm = np.full(n, np.nan)
    for i in range(n):
        m, x, _ = clean_peaks(t['ms2_mzs'][i], t['ms2_normalized_intensities'][i], t['precursor_mz'][i])
        mz[i, :len(m)], inten[i, :len(m)], npk[i] = m, x, len(m)
        nm[i] = neutral_mass(t['precursor_mz'][i], t['adduct'][i])
    return t['molecule_id'], mz, inten, npk, nm


def predict(test_path, lib_path, out_path):
    """-> (timings dict, {molecule_id: [smiles...]})."""
    tm, t0 = {}, time.time()
    lib = Library.load(lib_path)
    tm['load_library'] = time.time() - t0
    t0 = time.time()
    ids, mz, inten, npk, nm = load_test(test_path)
    q = bin_rows(mz, inten, npk)
    tm['features'] = time.time() - t0
    t0 = time.time()
    rows = {}
    for mid in ids:
        rows.setdefault(mid, [])
    for i, mid in enumerate(ids):
        rows[mid].append(i)
    out = {}
    for mid, r in rows.items():
        m = nm[r][np.isfinite(nm[r])]
        mass = np.median(m) if len(m) else np.nan
        smi = []
        for ppm in FALLBACK_PPM:
            _, smi, _ = lib.rank(q[r], mass, ppm=ppm, top=MAX_RANK)
            if smi:
                break
        out[mid] = smi or [FALLBACK_SMILES]
    tm['rank'] = time.time() - t0
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['molecule_id', 'smiles'])
        for mid, smi in out.items():
            w.writerow([mid, ';'.join(smi)])
    return {k: round(v, 2) for k, v in tm.items()}, out


TEST_COLS = ['molecule_id', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'collision_energy_ev',
             'instrument_type', 'ionization_mode']


def predict_encoder(test_path, lib_path, pool_dir, ckpt_path, vocab_path, out_path, ppm=5.0, spb=256, workers=None):
    """P5.4: library hits with cosine >= LIB_T (D-034) first, then pool_v1 window candidates by soft Tanimoto to the
    Encoder's predicted fingerprint (scripts/eval_encoder.py with a = b = 0). A P6 Contrastive checkpoint instead scores
    every window candidate with P6_BLEND and forces no library hits (needs src.npy + fkey.npy in the pool dataset)."""
    import json

    import torch

    from casmi.data.batch import featurize_rows, to_batch
    from casmi.features.formula import fkey_counts
    from casmi.models.contrastive import Contrastive
    from casmi.models.encoder import Encoder, unpack_fp
    from casmi.retrieval.pool import CURATED_EXT, Pool, gather
    tm, t0 = {}, time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = json.load(open(vocab_path))
    sd = {k.replace('_orig_mod.', ''): v for k, v in torch.load(ckpt_path, map_location=dev)['model'].items()}
    contrastive = any(k.startswith('proj.') for k in sd)  # P6 checkpoint (D-034 #4b)
    model = (Contrastive if contrastive else Encoder)({k: len(v) for k, v in vocab.items()}).to(dev).eval()
    model.load_state_dict(sd)
    lib_t = P6_LIB_T if contrastive else LIB_T
    lib = Library.load(lib_path)
    pool = Pool(pool_dir, ('mass', 'ik14', 'src', 'fkey') + (('popularity',) if P6_POP else ()) if contrastive else ('mass', 'ik14'))  # what casmi26-pool ships
    tm['load'] = time.time() - t0
    t0 = time.time()
    t = pq.read_table(test_path, columns=TEST_COLS).to_pylist()
    arrs = featurize_rows(t, vocab)
    mols = {}
    for i, r in enumerate(t):
        mols.setdefault(r['molecule_id'], []).append(i)
    ids = list(mols)
    q = bin_rows(arrs['mz'], arrs['inten'].astype(np.float32), arrs['npeaks'])
    tm['features'] = time.time() - t0
    t0 = time.time()
    prob, emb, batch, n = {}, {}, [], 0
    with torch.no_grad():
        for j, mid in enumerate(ids + [None]):
            if mid is None or (batch and n + len(mols[mid]) > spb):
                idx = np.concatenate([mols[ids[k]] for k in batch])
                seg = np.repeat(np.arange(len(batch)), [len(mols[ids[k]]) for k in batch])
                b = {k: v.to(dev) for k, v in to_batch(arrs, idx, seg).items()}
                with torch.autocast(dev, torch.bfloat16, enabled=dev == 'cuda' and torch.cuda.is_bf16_supported()):
                    o = model(b)
                for i, (k, p) in enumerate(zip(batch, torch.sigmoid(o[0].float()).cpu())):
                    prob[ids[k]] = p
                    if contrastive:
                        emb[ids[k]] = o[2][i].float().cpu()
                batch, n = [], 0
            if mid is not None:
                batch.append(j)
                n += len(mols[mid])
    tm['model'] = time.time() - t0
    t0 = time.time()
    mass, win = {}, {}
    for mid, r in mols.items():
        m = arrs['neutral_mass'][r]
        m = m[np.isfinite(m)]
        mass[mid] = np.median(m) if len(m) else np.nan
        win[mid] = pool.window(mass[mid], ppm) if len(m) else (0, 0)
    rows = np.unique(np.concatenate([np.arange(lo, hi) for lo, hi in win.values()] + [np.zeros(0, np.int64)]))
    smiles, cfp = gather(pool_dir, rows, workers, tm)  # adds tm['read'], tm['fp']
    tm['candidates'] = time.time() - t0
    t0 = time.time()
    out = {}
    for mid, r in mols.items():
        c, s, has = lib.scores(q[r], mass[mid], ppm) if np.isfinite(mass[mid]) else ([], [], [])
        cand = [(10 + float(x), lib.ik14[i], lib.smiles[i]) for i, x, h in zip(c, s, has) if h and x >= lib_t]
        lo, hi = win[mid]
        if hi > lo:
            a = np.searchsorted(rows, lo)
            bits = unpack_fp(torch.from_numpy(cfp[a:a + hi - lo]))
            p = prob[mid]
            inter = bits @ p
            score = (inter / (bits.sum(1) + p.sum() - inter).clamp(min=1e-6)).numpy()
            if contrastive:  # scripts/eval_encoder.py P6.3 blend, a = 0
                wc, wt, wb = P6_BLEND
                with torch.no_grad():
                    counts = torch.from_numpy(fkey_counts(pool.fkey[lo:hi]))
                    cosv = (model.mol(bits.to(dev), counts.to(dev)) @ emb[mid].to(dev)).cpu().numpy()
                in_lib = {lib.ik14[i] for i, h in zip(c, has) if h}
                cur = np.array([(int(pool.src[lo + i]) & CURATED_EXT) > 0 or pool.ik14[lo + i].decode() in in_lib
                                for i in range(hi - lo)], np.float32)
                score = wc * cosv + wt * score + wb * cur
                if P6_POP:
                    score = score + P6_POP * pool.popularity[lo:hi].astype(np.float32)
                if P6_SRC:
                    score = np.where(pool.src[lo:hi] & P6_SRC, score, -np.inf)
            top = np.argsort(-score)[:4 * MAX_RANK]
            top = top[np.isfinite(score[top])]
            cand += [(float(score[i]), pool.ik14[lo + i].decode(), smiles[a + i]) for i in top]
        seen, smi = set(), []
        for _, k, s_ in sorted(cand, key=lambda x: -x[0]):
            if k not in seen:  # ponytail: plain vs tautomer InChIKey14 duplicates survive (D-025); re-key if it matters
                seen.add(k)
                smi.append(s_)
            if len(smi) == MAX_RANK:
                break
        if not smi:
            for wide in FALLBACK_PPM[1:]:
                _, smi, _ = lib.rank(q[r], mass[mid], ppm=wide, top=MAX_RANK)
                if smi:
                    break
        out[mid] = smi or [FALLBACK_SMILES]
    tm['rank'] = time.time() - t0
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['molecule_id', 'smiles'])
        for mid, smi in out.items():
            w.writerow([mid, ';'.join(smi)])
    return {k: round(v, 2) for k, v in tm.items()}, out


def validate(sub_path, test_path):
    """Format check: header, every test molecule_id exactly once, 1..25 ';'-joined non-empty SMILES."""
    want = set(pq.read_table(test_path, columns=['molecule_id'])['molecule_id'].to_pylist())
    with open(sub_path, newline='') as f:
        r = list(csv.reader(f))
    assert r[0] == ['molecule_id', 'smiles'], r[0]
    ids = [row[0] for row in r[1:]]
    assert len(ids) == len(set(ids)), 'duplicate molecule_id'
    assert set(ids) == want, f'missing {len(want - set(ids))}, extra {len(set(ids) - want)}'
    for row in r[1:]:
        g = row[1].split(';')
        assert 1 <= len(g) <= MAX_RANK and all(s.strip() for s in g), row[0]
    return len(ids)
