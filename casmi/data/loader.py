"""P2.6 shard loader: block-shuffled molecule batches with length bucketing over data/processed/v1/{split}/ memmaps.
Reads contiguous molecule blocks (never random rows), shuffles blocks and molecules within a block,
sorts molecules by peak count so a batch pads to a similar length, then shuffles batch order."""
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from casmi.features.formula import fkey_counts, formula_key

SPEC_KEYS = ('mz', 'inten', 'peakfeat', 'npeaks', 'prec', 'neutral_mass', 'ce', 'iso', 'adduct', 'instrument',
             'polarity', 'lib', 'formulas', 'nformulas', 'formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe')
LONG = ('adduct', 'instrument', 'polarity', 'lib', 'npeaks', 'nformulas')  # embedding ids + counts (torch lacks uint16)
PAD_TO = 8


class ShardDataset(IterableDataset):
    """Yields dicts of tensors: per spectrum (B rows, peaks trimmed to L), 'seg' int64[B] = molecule row in the batch,
    per molecule 'mol_fp' uint8[M,256], 'mol_formula' int16[M,10], 'mol_idx' int64[M] (row in the split's mols.parquet).
    max_spec_per_mol=None keeps every spectrum (use for val); shuffle=False gives a fixed order.
    libs: keep only spectra whose lib id (vocab.json ingest_lib) is in this set; molecules left empty are skipped."""

    def __init__(self, root, split, spec_per_batch=256, max_spec_per_mol=16, block_mols=2048, shuffle=True, seed=0,
                 libs=None, hardneg=None):
        self.libs = None if libs is None else np.array(sorted(libs))
        self.hardneg = hardneg  # P6: scripts/hard_negatives.py dir (rows match this split's molecules) -> neg_* keys
        self.dir = f'{root}/{split}'
        self.spb, self.cap, self.block, self.shuffle, self.seed = spec_per_batch, max_spec_per_mol, block_mols, shuffle, seed
        self.off = np.load(f'{self.dir}/mol_offsets.npy')
        self.epoch = 0

    def _arr(self, k):
        return np.load(f'{self.dir}/{k}.npy', mmap_mode='r')

    def __iter__(self):
        # each (persistent) worker copy advances its own epoch counter identically -> same block permutation everywhere
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        n_mol = len(self.off) - 1
        blocks = np.arange(0, n_mol, self.block)
        if self.shuffle:
            rng.shuffle(blocks)
        rng = np.random.default_rng((self.seed, self.epoch, wid))
        arrs = {k: self._arr(k) for k in SPEC_KEYS + ('mol_fp', 'mol_formula')}
        if self.hardneg:
            arrs.update({f'neg_{k}': np.load(f'{self.hardneg}/{k}.npy', mmap_mode='r') for k in ('fp', 'fkey', 'n')})
        for b0 in blocks[wid::nw]:
            yield from self._block(arrs, b0, min(b0 + self.block, n_mol), rng)

    def _block(self, arrs, m0, m1, rng):
        s0, s1 = self.off[m0], self.off[m1]
        loc = self.off[m0:m1 + 1] - s0
        blk = {k: np.asarray(arrs[k][s0:s1]) for k in SPEC_KEYS}  # one contiguous read per array
        fp, mf = np.asarray(arrs['mol_fp'][m0:m1]), np.asarray(arrs['mol_formula'][m0:m1])
        if self.hardneg:
            neg = {k: np.asarray(arrs[f'neg_{k}'][m0:m1]) for k in ('fp', 'fkey', 'n')}
        mols, keep = [], []
        for j in range(m1 - m0):
            idx = np.arange(loc[j], loc[j + 1])
            if self.libs is not None:
                idx = idx[np.isin(blk['lib'][idx], self.libs)]
                if len(idx) == 0:
                    continue
            if self.cap and len(idx) > self.cap:
                idx = rng.choice(idx, self.cap, replace=False)
            mols.append(idx)
            keep.append(j)
        keep = np.array(keep, np.int64)
        order = rng.permutation(len(mols)) if self.shuffle else np.arange(len(mols))
        maxnp = np.array([blk['npeaks'][idx].max() for idx in mols])
        order = order[np.argsort(maxnp[order], kind='stable')]  # bucket by length, random within equal length
        batches, cur, n = [], [], 0
        for j in order:
            if cur and n + len(mols[j]) > self.spb:
                batches.append(cur)
                cur, n = [], 0
            cur.append(j)
            n += len(mols[j])
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        for bm in batches:
            idx = np.concatenate([mols[j] for j in bm])
            seg = np.repeat(np.arange(len(bm)), [len(mols[j]) for j in bm])
            L = max(PAD_TO, -(-int(blk['npeaks'][idx].max()) // PAD_TO) * PAD_TO)
            out = {}
            for k in SPEC_KEYS:
                a = blk[k][idx]
                if k in ('mz', 'inten', 'peakfeat'):
                    a = a[:, :L]
                out[k] = torch.from_numpy(a.astype(np.int64) if k in LONG else np.ascontiguousarray(a))
            out['seg'] = torch.from_numpy(seg)
            mj = keep[bm]  # block-local molecule rows
            out['mol_fp'] = torch.from_numpy(fp[mj])
            out['mol_formula'] = torch.from_numpy(mf[mj])
            out['mol_idx'] = torch.from_numpy(mj + m0)
            if self.hardneg:
                out['neg_fp'] = torch.from_numpy(neg['fp'][mj])
                out['neg_formula'] = torch.from_numpy(fkey_counts(neg['fkey'][mj]))
                out['n_neg'] = torch.from_numpy(neg['n'][mj].astype(np.int64))
                out['pos_formula'] = torch.from_numpy(fkey_counts(formula_key(np.maximum(mf[mj], 0))))
            yield out


def make_loader(root, split, workers=8, **kw):
    ds = ShardDataset(root, split, **kw)
    return DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
