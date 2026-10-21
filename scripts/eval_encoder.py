"""P5.4 val: rank fixed val candidate pools (scripts/val_candidates.py) with a trained Encoder.
Score = library tier (cosine >= COS_T, as P4.5) first, then soft Tanimoto(predicted bit probabilities, candidate fp)
+ a * formula agreement + b * curated NP source, swept over (a, b). Honesty rules as scripts/eval_pool.py (D-026).
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
from casmi.models.encoder import Encoder, unpack_fp
from casmi.retrieval.library import Library, bin_rows
from casmi.retrieval.pool import COS_T, CURATED_EXT, TRAIN, Pool, formula_score
from scripts.pool_recall import plain_key

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
VC = f'{POOL}/valcand'
A, B = (0.0, 0.1, 0.3), (0.0, 0.05, 0.1)
RDLogger.DisableLog('rdApp.*')


@torch.no_grad()
def predict(model, split):
    """-> float32 [M, 2048] predicted bit probabilities in mols.parquet order."""
    out = {}
    for b in make_loader(ROOT, split, workers=2, shuffle=False, max_spec_per_mol=None):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        with torch.autocast('cuda', torch.bfloat16):
            fp, _ = model(b)
        for i, p in zip(b['mol_idx'].tolist(), torch.sigmoid(fp.float()).cpu().numpy()):
            out[i] = p
    return np.stack([out[i] for i in range(len(out))])


def main():
    ck = sys.argv[1]
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    model = Encoder({k: len(v) for k, v in vocab.items()}).cuda().eval()
    sd = torch.load(ck, map_location='cuda')['model']
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()})  # enc was torch.compile'd
    sp = pq.read_table(f'{ROOT}/splits.parquet', columns=['ik14', 'split']).to_pydict()
    novel = [k for k, s in zip(sp['ik14'], sp['split']) if s == 'v_novel']
    strip = np.array(sorted(k.encode() for k, s in zip(sp['ik14'], sp['split']) if s in VAL), 'S14')
    lib, pool = Library(ROOT, ('train',), exclude_ik14=novel), Pool(POOL)
    rows, cfp = np.load(f'{VC}/rows.npy'), np.load(f'{VC}/fp.npy', mmap_mode='r')
    res = {}
    for split in VAL:
        d = f'{ROOT}/{split}'
        prob = torch.from_numpy(predict(model, split)).cuda()
        off = np.load(f'{d}/mol_offsets.npy')
        mols = pq.read_table(f'{d}/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
        q = bin_rows(*(np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks')))
        nm = np.load(f'{d}/neutral_mass.npy')
        forms, nform = np.load(f'{d}/formulas.npy', mmap_mode='r'), np.load(f'{d}/nformulas.npy')
        wins = np.load(f'{VC}/{split}_windows.npy')
        rr = {ab: [] for ab in itertools.product(A, B)}
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
            fsd = formula_score(forms[a:b], nform[a:b])
            fk = pool.fkey[lo + keep]
            u, inv = np.unique(fk, return_inverse=True)
            fs = np.array([fsd.get(int(x), 0.0) for x in u], np.float32)[inv]
            cur = (((src[keep] & CURATED_EXT) > 0) | (cos[keep] >= 0)).astype(np.float32)
            tier = cos[keep] >= COS_T
            truth = {k.encode(), plain_key(smi)}
            hit = np.array([x in truth for x in keys[keep]])
            for (wa, wb), v in rr.items():
                score = np.where(tier, 10 + cos[keep], tani + wa * fs + wb * cur)
                if hit.any():
                    best = score[hit].max()
                    rank = int((score > best).sum())
                    v.append(1 / (rank + 1) if rank < 25 else 0.0)
                else:
                    v.append(0.0)
        res[split] = {f'{wa},{wb}': round(float(np.mean(v)), 4) for (wa, wb), v in rr.items()}
        print(split, json.dumps(res[split]), flush=True)
    res['weighted'] = {ab: round(float(np.mean([res[s][ab] for s in ('v_lib', 'v_pool', 'v_novel')])), 4)
                       for ab in res['v_lib']}
    print('weighted', json.dumps(res['weighted']), flush=True)
    json.dump(res, open(ck.rsplit('/', 1)[0] + '/eval_pool.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
