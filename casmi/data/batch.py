"""Inference batches from raw spectra (Kaggle test rows), mirroring what scripts/build_shards.py stores and
casmi.data.loader yields, so the Encoder sees identical inputs (float16 storage of inten/peakfeat included)."""
import numpy as np
import torch

from casmi.data.loader import LONG, PAD_TO
from casmi.features import TOP_FORMULAS, featurize
from casmi.features.clean import P
from casmi.features.formula import EL
from casmi.features.peaks import F

K = TOP_FORMULAS
VOCAB = {'adduct': 'adduct', 'instrument_type': 'instrument', 'ionization_mode': 'polarity', 'ingest_lib': 'lib'}


def featurize_rows(rows, vocab):
    """rows: list of dicts with ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct, collision_energy_ev,
    instrument_type, ionization_mode -> dict of per-spectrum numpy arrays (shard dtypes, peaks padded to P)."""
    n = len(rows)
    o = {'mz': np.zeros((n, P), np.float32), 'inten': np.zeros((n, P), np.float16),
         'peakfeat': np.zeros((n, P, F), np.float16), 'npeaks': np.zeros(n, np.uint16), 'prec': np.zeros(n),
         'neutral_mass': np.zeros(n), 'ce': np.zeros((n, 2), np.float32), 'iso': np.zeros((n, 6), np.float32),
         'formulas': np.zeros((n, K, len(EL)), np.int16), 'nformulas': np.zeros(n, np.uint8),
         **{k: np.zeros((n, K), np.float32) for k in ('formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe')},
         **{name: np.zeros(n, np.uint16) for name in VOCAB.values()}}
    for i, r in enumerate(rows):
        f = featurize(r['ms2_mzs'], r['ms2_normalized_intensities'], r['precursor_mz'], r['adduct'])
        m = len(f['mz'])
        o['mz'][i, :m], o['inten'][i, :m], o['peakfeat'][i, :m], o['npeaks'][i] = f['mz'], f['inten'], f['peakfeat'], m
        o['prec'][i], o['neutral_mass'][i], o['iso'][i] = r['precursor_mz'], f['neutral_mass'], f['iso']
        ce = r.get('collision_energy_ev')
        o['ce'][i] = (min(ce), max(ce)) if ce else (np.nan, np.nan)
        for col, name in VOCAB.items():
            o[name][i] = vocab[col].get(r.get(col), 0)
        k = len(f['formulas'])
        o['nformulas'][i] = k
        for name in ('formulas', 'formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe'):
            o[name][i, :k] = f[name]
    return o


def to_batch(arrs, idx, seg):
    """Spectra idx (grouped by molecule; seg = molecule row in the batch) -> loader-format tensor dict (no targets)."""
    L = max(PAD_TO, -(-int(arrs['npeaks'][idx].max()) // PAD_TO) * PAD_TO)
    out = {}
    for k, a in arrs.items():
        a = a[idx]
        if k in ('mz', 'inten', 'peakfeat'):
            a = a[:, :L]
        out[k] = torch.from_numpy(a.astype(np.int64) if k in LONG else np.ascontiguousarray(a))
    out['seg'] = torch.from_numpy(np.asarray(seg, np.int64))
    out['mol_fp'] = torch.zeros(int(seg.max()) + 1, 256, dtype=torch.uint8)  # Encoder reads M from here
    return out
