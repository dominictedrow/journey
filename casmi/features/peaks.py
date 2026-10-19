"""P2.4 per-peak physics features. Precise masses stay in mz float32 / prec float64; peakfeat holds
low-precision descriptors only (it is stored as float16)."""
import numpy as np
from numba import njit

from casmi.features.formula import MASS, PROTON

# name -> monoisotopic neutral loss (Da)
LOSSES = {
    'H2O': 18.010565, 'CO': 27.994915, 'CO2': 43.989829, 'NH3': 17.026549, 'CH3': 15.023475,
    'CH2O': 30.010565, 'CH3OH': 32.026215, 'C2H2O': 42.010565, 'HCOOH': 46.005479, 'HCN': 27.010899,
    'C2H4': 28.031300, 'H2S': 33.987721, 'SO3': 79.956815, 'HCl': 35.976678, 'HF': 20.006229,
    'HBr': 79.926161, 'H3PO4': 97.976895, 'C3H9N': 59.073499, 'C3H2O3': 86.000394,
    'hexose': 162.052824, 'deoxyhexose': 146.057909, 'pentose': 132.042259, 'glucuronide': 176.032088,
    'caffeoyl': 162.031694, 'coumaroyl': 146.036779, 'galloyl': 152.010959,
}
LOSS_MASS = np.array(sorted(LOSSES.values()))
FEATS = ('md', 'kmd', 'loss_md', 'loss_id', 'pair_loss', 'is_prec', 'sub_hit', 'sub_ppm',
         'sub_C', 'sub_H', 'sub_N', 'sub_O', 'sub_SP', 'sub_X', 'sub_rdbe')
F = len(FEATS)
TOP_K_PAIRS = 16
SUB_MAX_COMBOS = 200_000  # ponytail: skip subformula search for huge formulas (>~C100); raise if P7 wants them


def _tol(m):
    return np.maximum(0.005, m * 10e-6)


def loss_id(loss, tol):
    """1-based index into LOSS_MASS of the closest table loss within tol, else 0."""
    j = np.clip(np.searchsorted(LOSS_MASS, loss), 1, len(LOSS_MASS) - 1)
    near = np.where(np.abs(LOSS_MASS[j - 1] - loss) < np.abs(LOSS_MASS[j] - loss), j - 1, j)
    return np.where(np.abs(LOSS_MASS[near] - loss) <= tol, near + 1, 0)


@njit(cache=True)
def _subformula(mz, tol, form, shift, masses):
    """Best subformula of neutral `form` (int[10]) per peak: ion mass = f + shift. -> int16[n,10], err ppm, hit."""
    n = len(mz)
    best = np.zeros((n, 10), np.int16)
    err = np.full(n, np.inf)
    radix = form + 1
    radix[1] = 1  # H is solved per peak, not enumerated
    total = 1
    for r in radix:
        total *= r
    cur = np.zeros(10, np.int64)
    for _ in range(total):
        rest = 0.0
        for e in range(10):
            rest += cur[e] * masses[e]
        heavy = cur[0] + cur[2] + cur[3] + cur[4] + cur[5] + cur[6] + cur[7] + cur[8] + cur[9]
        if heavy > 0:
            for i in range(n):
                h = int(np.round((mz[i] - shift - rest) / masses[1]))
                if h < 0 or h > form[1]:
                    continue
                d = mz[i] - shift - rest - h * masses[1]
                if abs(d) <= tol[i] and abs(d) < abs(err[i]):
                    x = cur[6] + cur[7] + cur[8] + cur[9]
                    if 2 * cur[0] + 2 + cur[2] + cur[4] - h - x >= -1:  # fragment RDBE >= -0.5
                        err[i] = d
                        for e in range(10):
                            best[i, e] = cur[e]
                        best[i, 1] = h
        for e in range(10):  # mixed-radix increment
            cur[e] += 1
            if cur[e] < radix[e]:
                break
            cur[e] = 0
    return best, err


def peak_features(mz, inten, prec, formula=None, charge=1):
    """Per-peak features float32[n, F] (column order FEATS) for a cleaned spectrum.
    formula: int[10] top-ranked neutral precursor formula (EL order) or None; charge: +1 / -1.
    Fragment ions assumed [f+H]+ / [f-H]-."""
    # ponytail: protonation/deprotonation only; Na/K-retaining fragments of [M+Na]+ go unannotated
    mz = np.asarray(mz, np.float64)
    n = len(mz)
    out = np.zeros((n, F), np.float32)
    if n == 0:
        return out
    tol = _tol(mz)
    out[:, 0] = mz - np.round(mz)
    km = mz * 14.0 / 14.01565
    out[:, 1] = np.round(km) - km
    loss = prec - mz
    out[:, 2] = loss - np.round(loss)
    out[:, 3] = loss_id(loss, _tol(prec))
    k = np.argsort(-np.asarray(inten), kind='stable')[:TOP_K_PAIRS]
    d = np.abs(mz[k, None] - mz[None, k])
    out[k, 4] = (loss_id(d, _tol(prec)) > 0).sum(1)
    out[:, 5] = np.abs(loss) <= tol
    if formula is not None and np.prod(np.asarray(formula, np.int64)[[0, 2, 3, 4, 5, 6, 7, 8, 9]] + 1) <= SUB_MAX_COMBOS:
        best, err = _subformula(mz, tol, np.asarray(formula, np.int64), charge * PROTON, MASS)
        hit = np.isfinite(err)
        out[:, 6] = hit
        out[hit, 7] = err[hit] / mz[hit] * 1e6
        b = best.astype(np.float32)
        out[:, 8:12] = b[:, [0, 1, 2, 3]]
        out[:, 12] = b[:, 4] + b[:, 5]
        out[:, 13] = b[:, 6:].sum(1)
        out[hit, 14] = b[hit, 0] + 1 + 0.5 * (b[hit, 2] + b[hit, 4] - b[hit, 1] - out[hit, 13])
    return out
