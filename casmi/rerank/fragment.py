"""P7.1 in-house fragmenter: break 1-2 bonds of a candidate's heavy-atom graph (MetFrag-style, no rearrangements),
keep each connected piece once, and match piece m/z (+-H shifts) to MS2 peaks."""
import itertools

import numpy as np
from rdkit import Chem

from casmi.features.formula import MASS, PROTON
from casmi.features.peaks import _tol

H = MASS[1]
SHIFTS = np.arange(-2, 3)  # hydrogen transfers tolerated per fragment


def fragments(smiles, max_cuts=2):
    """-> (neutral masses float64[n], cuts int8[n]) of the unique pieces left after breaking <= max_cuts bonds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(0), np.zeros(0, np.int8)
    n = mol.GetNumAtoms()
    am = np.array([_mono(a) + a.GetTotalNumHs() * H for a in mol.GetAtoms()])
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    seen = {}
    for k in range(1, max_cuts + 1):
        for cut in itertools.combinations(range(len(edges)), k):
            for piece in _components(n, edges, set(cut)):
                if piece != (1 << n) - 1:
                    seen.setdefault(piece, k)
    if not seen:
        return np.zeros(0), np.zeros(0, np.int8)
    bits = np.array([[(p >> i) & 1 for i in range(n)] for p in seen], bool)
    return bits.astype(np.float64) @ am, np.fromiter(seen.values(), np.int8, len(seen))


_MONO = {}


def _mono(atom):
    s = atom.GetSymbol()
    if s not in _MONO:
        _MONO[s] = Chem.GetPeriodicTable().GetMostCommonIsotopeMass(s)
    return _MONO[s]


def _components(n, edges, cut):
    """Connected components (as atom bitmasks) of the graph minus the cut edges."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (a, b) in enumerate(edges):
        if i not in cut:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comp = {}
    for i in range(n):
        r = find(i)
        comp[r] = comp.get(r, 0) | (1 << i)
    return comp.values()


def explain(mz, inten, masses, charge=1):
    """Fraction of intensity and count of peaks matched by any fragment ion (+-2 H, [frag+H]+ / [frag-H]-)."""
    if len(masses) == 0 or len(mz) == 0:
        return 0.0, 0
    ion = np.sort(((masses[:, None] + SHIFTS * H) + charge * PROTON).ravel())
    j = np.clip(np.searchsorted(ion, mz), 1, len(ion) - 1)
    hit = np.minimum(np.abs(ion[j] - mz), np.abs(ion[j - 1] - mz)) <= _tol(mz)
    return float(inten[hit].sum() / max(inten.sum(), 1e-12)), int(hit.sum())


MAX_SPEC = 8  # spectra per molecule scored by mol_feats


def spectra(mz, inten, npeaks, polarity, idx, neg_id):
    """Spectrum rows idx of shard-format arrays (first MAX_SPEC) -> [(mz, inten, charge)] for mol_feats. Shared by
    scripts/frag_features.py (cached shards) and casmi/kaggle/predict.py (featurize_rows on the fly): parity."""
    out = []
    for s in list(idx)[:MAX_SPEC]:
        k = int(npeaks[s])
        out.append((np.asarray(mz[s, :k], np.float64), np.asarray(inten[s, :k], np.float64), -1 if polarity[s] == neg_id else 1))
    return out


def mol_feats(args):
    """(candidate smiles, spectra()) -> float32 [n_cand, 4]: expl, expl1, peak_frac, log_nfrag (spectrum-intensity
    weighted over spectra; None smiles -> 0)."""
    smis, spec = args
    out = np.zeros((len(smis), 4), np.float32)
    w = np.array([y.sum() for _, y, _ in spec])
    w = w / max(w.sum(), 1e-12)
    for i, smi in enumerate(smis):
        if smi is None:
            continue
        m, c = fragments(smi)
        for (x, y, ch), wi in zip(spec, w):
            e, n = explain(x, y, m, ch)
            out[i, 0] += wi * e
            out[i, 1] += wi * explain(x, y, m[c == 1], ch)[0]
            out[i, 2] += wi * n / max(len(x), 1)
        out[i, 3] = np.log1p(len(m))
    return out
