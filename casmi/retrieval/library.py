"""P3.1 library baseline: candidates = structures within +-PPM of the query neutral mass; score = mean over query
spectra of the best binned-sqrt cosine to any reference spectrum of the candidate; ties -> closest mass."""
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

BIN = 0.02     # ponytail: fixed-width bins; a peak near a bin edge can split from its match. Tolerance matching if P7 needs it
MAX_MZ = 2000.5
NBINS = int(MAX_MZ / BIN) + 1
PPM = 5.0


def bin_rows(mz, inten, npeaks):
    """Padded [N,P] peaks -> L2-normalized CSR [N, NBINS] (duplicate bins summed)."""
    mz, inten, npeaks = np.asarray(mz), np.asarray(inten, np.float32), np.asarray(npeaks)
    valid = np.arange(mz.shape[1])[None] < npeaks[:, None]
    rows = np.broadcast_to(np.arange(len(mz))[:, None], mz.shape)[valid]
    cols = np.minimum(np.rint(mz[valid] / BIN), NBINS - 1).astype(np.int32)
    m = sp.csr_matrix((inten[valid], (rows, cols)), shape=(len(mz), NBINS), dtype=np.float32)
    m.sum_duplicates()
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    return sp.diags(1 / np.maximum(norm, 1e-12)).astype(np.float32) @ m


class Library:
    """refs: shard split dirs whose spectra are reference spectra (sorted by molecule within each dir).
    exclude_ik14: structures removed from the candidate pool (V-novel for local val)."""

    def __init__(self, root, ref_splits=('train',), exclude_ik14=(), cache_dir=None):
        cache_dir = cache_dir or f'{root}/cache'
        key = hashlib.sha256(json.dumps([BIN, list(ref_splits), sorted(exclude_ik14)]).encode()).hexdigest()[:12]
        path = f'{cache_dir}/library_{key}.npz'
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.mass, self.ik14, self.smiles = z['mass'], z['ik14'], z['smiles']
            self.ref_off = z['ref_off']
            self.refs = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=(int(z['nref']), NBINS))
            return
        s = pq.read_table(f'{root}/structures.parquet', columns=['smiles', 'ik14', 'exact_mass']).to_pydict()
        ex, seen, cand = set(exclude_ik14), set(), []
        for smi, k, m in zip(s['smiles'], s['ik14'], s['exact_mass']):
            if k not in seen and k not in ex and m is not None:
                seen.add(k)
                cand.append((m, k, smi))  # ponytail: first SMILES per ik14 stands for the structure
        cand.sort()
        self.mass = np.array([c[0] for c in cand])
        self.ik14 = np.array([c[1] for c in cand])
        self.smiles = np.array([c[2] for c in cand])
        pos = {k: i for i, k in enumerate(self.ik14)}
        mats, owner = [], []
        for split in ref_splits:
            d = f'{root}/{split}'
            off = np.load(f'{d}/mol_offsets.npy')
            ks = pq.read_table(f'{d}/mols.parquet', columns=['ik14'])['ik14'].to_pylist()
            mz, inten, npk = (np.load(f'{d}/{k}.npy', mmap_mode='r') for k in ('mz', 'inten', 'npeaks'))
            for a in range(0, len(npk), 200_000):
                b = min(a + 200_000, len(npk))
                mats.append(bin_rows(mz[a:b], inten[a:b], npk[a:b]))
            owner.append(np.repeat([pos.get(k, -1) for k in ks], np.diff(off)))
        owner = np.concatenate(owner)
        refs = sp.vstack(mats, format='csr')
        keep = np.flatnonzero(owner >= 0)
        order = keep[np.argsort(owner[keep], kind='stable')]  # group reference spectra by candidate
        self.refs = refs[order]
        self.ref_off = np.searchsorted(owner[order], np.arange(len(cand) + 1))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, mass=self.mass, ik14=self.ik14, smiles=self.smiles, ref_off=self.ref_off,
                 data=self.refs.data, indices=self.refs.indices, indptr=self.refs.indptr, nref=self.refs.shape[0])

    def candidates(self, mass, ppm=PPM):
        tol = mass * ppm * 1e-6
        return np.arange(np.searchsorted(self.mass, mass - tol), np.searchsorted(self.mass, mass + tol, side='right'))

    def rank(self, q, mass, top=25):
        """q: CSR [n_query_spectra, NBINS] of one molecule; mass: neutral mass -> (ik14[], smiles[], score[])."""
        c = self.candidates(mass) if np.isfinite(mass) else np.array([], int)
        if len(c) == 0:
            return [], [], []
        lo, hi = self.ref_off[c], self.ref_off[c + 1]
        score = np.zeros(len(c), np.float32)
        r0, r1 = lo.min(), hi.max()  # candidates are mass-contiguous, so their refs are one contiguous block
        if r1 > r0:
            sims = (q @ self.refs[r0:r1].T).toarray()  # [nq, nref]
            for j in range(len(c)):
                if hi[j] > lo[j]:
                    score[j] = sims[:, lo[j] - r0:hi[j] - r0].max(1).mean()
        o = np.lexsort((np.abs(self.mass[c] - mass), -score))[:top]
        return self.ik14[c[o]].tolist(), self.smiles[c[o]].tolist(), score[o].tolist()
