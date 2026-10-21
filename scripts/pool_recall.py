"""P4.3 recall report for pool_v1: per val split, share of molecules whose answer sits in the +-5/10 ppm window
around the query neutral mass (median over its spectra), plus window sizes. 'ext' ignores the train source
(src bit 1), i.e. whether COCONUT/LOTUS/PubChem alone contain the answer (the kind-2 path; for V-novel this is
the only honest number, since train structures include V-novel by construction).
Answer matched on tautomer ik14 or plain InChIKey14 (PubChem rows carry plain keys, D-025).
Usage: python scripts/pool_recall.py -> reports/pool_recall.json"""
import json

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

from casmi.data.splits import VAL

V1, POOL = 'data/processed/v1', 'data/processed/pool_v1'
RDLogger.DisableLog('rdApp.*')


def plain_key(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14].encode() if m else b''


def main():
    mass = np.load(f'{POOL}/mass.npy', mmap_mode='r')
    keys = np.load(f'{POOL}/ik14.npy', mmap_mode='r')
    ext = (np.load(f'{POOL}/src.npy', mmap_mode='r') & 14) > 0
    rep = {}
    for split in VAL:
        d = f'{V1}/{split}'
        off = np.load(f'{d}/mol_offsets.npy')
        nm = np.load(f'{d}/neutral_mass.npy')
        mols = pq.read_table(f'{d}/mols.parquet', columns=['ik14', 'smiles']).to_pydict()
        res = {}
        for ppm in (5, 10):
            hit, hit_ext, n, n_ext = [], [], [], []
            for j, (k, smi) in enumerate(zip(mols['ik14'], mols['smiles'])):
                m = nm[off[j]:off[j + 1]]
                m = np.median(m[np.isfinite(m)]) if np.isfinite(m).any() else np.nan
                if not np.isfinite(m):
                    hit.append(False), hit_ext.append(False), n.append(0), n_ext.append(0)
                    continue
                lo, hi = np.searchsorted(mass, m * (1 - ppm * 1e-6)), np.searchsorted(mass, m * (1 + ppm * 1e-6), 'right')
                w, e = keys[lo:hi], ext[lo:hi]
                found = np.isin(w, [k.encode(), plain_key(smi)])
                hit.append(found.any()), hit_ext.append((found & e).any()), n.append(hi - lo), n_ext.append(int(e.sum()))
            res[f'{ppm}ppm'] = {
                'recall_any': round(float(np.mean(hit)), 4), 'recall_ext': round(float(np.mean(hit_ext)), 4),
                'cands_median': float(np.median(n)), 'cands_p90': float(np.percentile(n, 90)),
                'cands_max': int(np.max(n)), 'ext_cands_median': float(np.median(n_ext))}
        rep[split] = res
        print(split, json.dumps(res), flush=True)
    json.dump(rep, open('reports/pool_recall.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
