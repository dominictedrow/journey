"""P5.4 val: rank fixed val candidate pools (scripts/val_candidates.py) with a trained Encoder.
Score = library tier (cosine >= COS_T, as P4.5) first, then soft Tanimoto(predicted bit probabilities, candidate fp)
+ a * formula agreement + b * curated NP source, swept over (a, b). Honesty rules as scripts/eval_pool.py (D-026).
P6.3: a Contrastive checkpoint (has proj.*) scores c * cosine(spectrum emb, candidate emb) + t * Tanimoto + a * fs + b * cur,
swept over (c, t, a, b). Keys in eval_pool.json are 'c,t,a,b' for both kinds (P5: c = 0, t = 1).
Usage: python scripts/eval_encoder.py runs/<name>/last.pt -> runs/<name>/eval_pool.json"""
import itertools
import json
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import RDLogger

from casmi.data.loader import make_loader
from casmi.data.splits import VAL
from casmi.features.formula import fkey_counts, formula_key
from casmi.models.contrastive import Contrastive
from casmi.models.encoder import Encoder, unpack_fp
from casmi.retrieval.library import Library, bin_rows
from casmi.retrieval.pool import COS_T, CURATED_EXT, TRAIN, Pool, formula_score
from scripts.pool_recall import plain_key

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
VC = f'{POOL}/valcand'
A, B = (0.0, 0.1, 0.3), (0.0, 0.05, 0.1)
CT = ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0), (1.0, 8.0), (0.0, 1.0))  # P6.3 (cosine, Tanimoto)
RDLogger.DisableLog('rdApp.*')


@torch.no_grad()
def predict(model, split):
    """-> (float32 [M, 2048] predicted bit probabilities, [M, E] spectrum embeddings or None) in mols.parquet order."""
    out, emb = {}, {}
    for b in make_loader(ROOT, split, workers=2, shuffle=False, max_spec_per_mol=None):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        with torch.autocast('cuda', torch.bfloat16):
            o = model(b)
        for j, (i, p) in enumerate(zip(b['mol_idx'].tolist(), torch.sigmoid(o[0].float()).cpu().numpy())):
            out[i] = p
            if len(o) > 2:
                emb[i] = o[2][j].float().cpu().numpy()
    return np.stack([out[i] for i in range(len(out))]), np.stack([emb[i] for i in range(len(emb))]) if emb else None


def main():
    ck = sys.argv[1]
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    sd = {k.replace('_orig_mod.', ''): v for k, v in torch.load(ck, map_location='cuda')['model'].items()}  # compiled enc
    contrastive = any(k.startswith('proj.') for k in sd)
    model = (Contrastive if contrastive else Encoder)({k: len(v) for k, v in vocab.items()}).cuda().eval()
    model.load_state_dict(sd)
    grid = list(itertools.product(CT if contrastive else ((0.0, 1.0),), A, B))
    sp = pq.read_table(f'{ROOT}/splits.parquet', columns=['ik14', 'split']).to_pydict()
    novel = [k for k, s in zip(sp['ik14'], sp['split']) if s == 'v_novel']
    strip = np.array(sorted(k.encode() for k, s in zip(sp['ik14'], sp['split']) if s in VAL), 'S14')
    lib, pool = Library(ROOT, ('train',), exclude_ik14=novel), Pool(POOL)
    rows, cfp = np.load(f'{VC}/rows.npy'), np.load(f'{VC}/fp.npy', mmap_mode='r')
    res, detail = {}, {'grid': np.array([f'{wc},{wt},{wa},{wb}' for (wc, wt), wa, wb in grid])}
    for split in VAL:
        d = f'{ROOT}/{split}'
        prob, emb = predict(model, split)
        prob = torch.from_numpy(prob).cuda()
        off = np.load(f'{d}/mol_offsets.npy')
        mols = pq.read_table(f'{d}/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
        q = bin_rows(*(np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks')))
        nm = np.load(f'{d}/neutral_mass.npy')
        forms, nform = np.load(f'{d}/formulas.npy', mmap_mode='r'), np.load(f'{d}/nformulas.npy')
        wins = np.load(f'{VC}/{split}_windows.npy')
        rr = {g: [] for g in grid}
        ranks = np.full((len(mols['ik14']), len(grid)), -1, np.int32)  # uncapped 0-based rank; -1 = truth not in window
        nwin = np.zeros(len(mols['ik14']), np.int32)
        for j, (k, smi) in enumerate(zip(mols['ik14'], mols['smiles'])):
            lo, hi = wins[j]
            a, b = off[j], off[j + 1]
            m = nm[a:b]
            m = np.median(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan
            if hi <= lo:
                for v in rr.values():
                    v.append(0.0)
                continue
            c, s, has = lib.scores(q[a:b], m)
            lib_cos = {lib.ik14[i].encode(): float(x) for i, x, h in zip(c, s, has) if h}
            keys, src = pool.ik14[lo:hi], pool.src[lo:hi].copy()
            cos = np.array([lib_cos.get(x, -1.0) for x in keys], np.float32)
            src[np.isin(keys, strip)] &= ~np.uint8(TRAIN)
            keep = np.flatnonzero((src > 0) | (cos >= 0))
            r0 = np.searchsorted(rows, lo)
            bits = unpack_fp(torch.from_numpy(np.asarray(cfp[r0 + keep])).cuda())
            p = prob[j]
            inter = bits @ p
            tani = (inter / (bits.sum(1) + p.sum() - inter).clamp(min=1e-6)).cpu().numpy()
            cosv = 0.0
            if emb is not None:
                with torch.no_grad():
                    cf = torch.from_numpy(fkey_counts(pool.fkey[lo + keep])).cuda()
                    cosv = (model.mol(bits, cf) @ torch.from_numpy(emb[j]).cuda()).cpu().numpy()
            fsd = formula_score(forms[a:b], nform[a:b])
            fk = pool.fkey[lo + keep]
            u, inv = np.unique(fk, return_inverse=True)
            fs = np.array([fsd.get(int(x), 0.0) for x in u], np.float32)[inv]
            cur = (((src[keep] & CURATED_EXT) > 0) | (cos[keep] >= 0)).astype(np.float32)
            tier = cos[keep] >= COS_T
            truth = {k.encode(), plain_key(smi)}
            hit = np.array([x in truth for x in keys[keep]])
            nwin[j] = len(keep)
            for g, (((wc, wt), wa, wb), v) in enumerate(rr.items()):
                score = np.where(tier, 10 + cos[keep], wc * cosv + wt * tani + wa * fs + wb * cur)
                if hit.any():
                    best = score[hit].max()
                    rank = int((score > best).sum())
                    ranks[j, g] = rank
                    v.append(1 / (rank + 1) if rank < 25 else 0.0)
                else:
                    v.append(0.0)
        res[split] = {f'{wc},{wt},{wa},{wb}': round(float(np.mean(v)), 4) for ((wc, wt), wa, wb), v in rr.items()}
        detail[f'{split}_ranks'], detail[f'{split}_nwin'] = ranks, nwin
        if contrastive:  # P6.4: true molecules in candidate-embedding space
            mfp, mf = np.load(f'{d}/mol_fp.npy'), np.load(f'{d}/mol_formula.npy')
            with torch.no_grad():
                detail[f'{split}_molemb'] = model.mol(unpack_fp(torch.from_numpy(mfp).cuda()), torch.from_numpy(
                    fkey_counts(formula_key(np.maximum(mf, 0)))).cuda()).cpu().numpy()
        print(split, json.dumps(res[split]), flush=True)
    res['weighted'] = {ab: round(float(np.mean([res[s][ab] for s in ('v_lib', 'v_pool', 'v_novel')])), 4)
                       for ab in res['v_lib']}
    print('weighted', json.dumps(res['weighted']), flush=True)
    json.dump(res, open(ck.rsplit('/', 1)[0] + '/eval_pool.json', 'w'), indent=1)
    np.savez(ck.rsplit('/', 1)[0] + '/eval_detail.npz', **detail)


if __name__ == '__main__':
    main()
