"""P2.4 precursor physics: adduct -> neutral mass, formula decomposition (CHNOPS+F/Cl/Br/I),
Seven-Golden-Rules/SENIOR/RDBE filters, isotope-ratio score (D-018), ranking."""
import os
import re
from functools import lru_cache

import numpy as np
from numba import njit

EL = ('C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I')
MASS = np.array([12.0, 1.00782503207, 14.0030740048, 15.99491461956, 30.97376163,
                 31.97207100, 18.99840322, 34.96885268, 78.9183371, 126.904473])
EXTRA_MASS = {'Na': 22.9897692809, 'K': 38.96370668, 'Li': 7.01600455, 'Ca': 39.96259098}
ELECTRON = 0.00054858
PROTON = MASS[1] - ELECTRON
CORE = np.array([2, 4, 5, 6, 7, 8, 9])            # N P S F Cl Br I, enumerated into a sorted table
CORE_MAX = np.array([20, 3, 4, 12, 6, 4, 4])       # covers p99 of train structures
# Seven Golden Rules element/C ratio caps for N P S F Cl Br I
RATIO = np.array([1.3, 0.3, 0.8, 1.5, 0.8, 0.8, 0.8])
ISO_AB1 = np.array([0.010816, 0.000115, 0.003654, 0.000381, 0, 0.007895, 0, 0, 0, 0])  # M+1/M per atom
ISO_AB2 = np.array([0, 0, 0, 0.002005, 0, 0.044740, 0, 0.319780, 0.972780, 0])         # M+2/M per atom
PRIOR_PATH = os.path.join(os.path.dirname(__file__), 'formula_prior.npz')
N_BINS = {'rdbe': 41, 'hc': 32, 'oc': 13}

_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')


def parse_formula(f):
    """'C6H12O6' -> {'C': 6, ...}; None if anything is left unparsed."""
    parts = _TOKEN.findall(f or '')
    if not f or ''.join(e + n for e, n in parts) != f:
        return None
    out = {}
    for e, n in parts:
        out[e] = out.get(e, 0) + int(n or 1)
    return out


def _mass(d):
    idx = {e: i for i, e in enumerate(EL)}
    return sum(n * (MASS[idx[e]] if e in idx else EXTRA_MASS[e]) for e, n in d.items())


@lru_cache(maxsize=4096)
def parse_adduct(adduct):
    """'[M+CH2O2-H]-' -> (n_M, signed charge, mass delta of adduct atoms); None if unsupported."""
    m = re.fullmatch(r'\[(\d*)M((?:[+-]\d*[A-Za-z0-9]+)*)\](\d*)([+-])', adduct or '')
    if not m:
        return None
    n = int(m.group(1) or 1)
    z = int(m.group(3) or 1) * (1 if m.group(4) == '+' else -1)
    delta = 0.0
    for sign, mult, f in re.findall(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)', m.group(2)):
        d = parse_formula(f)
        if d is None or any(e not in EL and e not in EXTRA_MASS for e in d):
            return None
        delta += (1 if sign == '+' else -1) * int(mult or 1) * _mass(d)
    return n, z, delta


def neutral_mass(prec_mz, adduct):
    """Monoisotopic mass of M; NaN if the adduct can't be parsed."""
    a = parse_adduct(adduct)
    if a is None:
        return float('nan')
    n, z, delta = a
    return (prec_mz * abs(z) - delta + z * ELECTRON) / n


@lru_cache(maxsize=1)
def core_table():
    """All N/P/S/halogen count combos within CORE_MAX, sorted by mass -> (mass float64[n], counts int8[n,7])."""
    grids = np.meshgrid(*[np.arange(k + 1) for k in CORE_MAX], indexing='ij')
    cnt = np.stack([g.ravel() for g in grids], 1).astype(np.int8)
    mass = cnt @ MASS[CORE]
    o = np.argsort(mass, kind='stable')
    return mass[o], np.ascontiguousarray(cnt[o])


@njit(cache=True)
def _decompose(mass, tol, core_mass, core_cnt, require_even, masses, ratio):
    out = []
    cmax = min(150, int(mass / 12.0) + 1)
    for c in range(1, cmax + 1):
        for o in range(0, min(40, int(1.2 * c)) + 1):
            rem0 = mass - 12.0 * c - o * masses[3]
            if rem0 < -tol:
                break
            hmin = int(np.ceil(0.2 * c))
            hmax = min(250, int(3.1 * c))
            for h in range(hmin, hmax + 1):
                r = rem0 - h * masses[1]
                if r < -tol:
                    break
                lo = np.searchsorted(core_mass, r - tol)
                hi = np.searchsorted(core_mass, r + tol, 'right')
                for k in range(lo, hi):
                    cc = core_cnt[k]
                    ok = True
                    for j in range(7):
                        if cc[j] > ratio[j] * c:
                            ok = False
                            break
                    if not ok:
                        continue
                    n, p, s = int(cc[0]), int(cc[1]), int(cc[2])
                    x = int(cc[3]) + int(cc[4]) + int(cc[5]) + int(cc[6])
                    twice_rdbe = 2 * c + 2 + n + p - h - x
                    if twice_rdbe < 0 or (require_even and twice_rdbe % 2 == 1):
                        continue
                    # SENIOR: total valence >= 2*(atoms-1); P as 3, S as 2
                    atoms = c + h + o + n + p + s + x
                    if 4 * c + h + 2 * o + 3 * n + 3 * p + 2 * s + x < 2 * (atoms - 1):
                        continue
                    err = mass - (12.0 * c + h * masses[1] + o * masses[3] + core_mass[k])
                    out.append((float(c), float(h), float(o), float(k), err))
    return out


def decompose(mass, ppm=10.0, require_even=True):
    """-> counts int16[n,10] in EL order, mass error in ppm float64[n]. Empty arrays if none."""
    if not mass > 0:
        return np.zeros((0, 10), np.int16), np.zeros(0)
    core_mass, core_cnt = core_table()
    hits = _decompose(float(mass), mass * ppm * 1e-6, core_mass, core_cnt, require_even, MASS, RATIO)
    if not hits:
        return np.zeros((0, 10), np.int16), np.zeros(0)
    a = np.array(hits)
    cnt = np.zeros((len(a), 10), np.int16)
    cnt[:, 0], cnt[:, 1], cnt[:, 3] = a[:, 0], a[:, 1], a[:, 2]
    cnt[:, CORE] = core_cnt[a[:, 3].astype(np.int64)]
    return cnt, a[:, 4] / mass * 1e6


def iso_theory(cnt):
    """Approximate theoretical M+1/M and M+2/M for formulas int[n,10]."""
    r1 = cnt @ ISO_AB1
    return r1, cnt @ ISO_AB2 + 0.5 * (cnt[:, 0] * ISO_AB1[0]) ** 2


def iso_score(cnt, iso):
    """Gaussian log-likelihood of observed M+1/M (and M+2/M) vs theory; 0 where peaks are absent.
    iso = clean.iso_features output."""
    # ponytail: fixed relative sigma; MS2 isolation distorts ratios — fit sigma on val if it matters
    s = np.zeros(len(cnt))
    if not iso[3]:
        return s
    r1, r2 = iso_theory(cnt)
    if iso[4]:
        s += -0.5 * ((iso[1] / iso[0] - r1) / (0.25 * r1 + 0.02)) ** 2
    if iso[5]:
        s += -0.5 * ((iso[2] / iso[0] - r2) / (0.25 * r2 + 0.02)) ** 2
    return s


def rdbe(cnt):
    """Ring/double-bond equivalents for formulas int[n,10]."""
    c = cnt.astype(np.float64)
    return c[:, 0] + 1 + 0.5 * (c[:, 2] + c[:, 4] - c[:, 1] - c[:, 6:].sum(1))


def prior_bins(cnt):
    """-> (rdbe bin, H/C bin (0.1 wide), O/C bin (0.1 wide)) int arrays for formulas int[n,10] with C > 0."""
    c = np.maximum(cnt[:, 0], 1)
    return (np.clip(np.floor(rdbe(cnt)), 0, N_BINS['rdbe'] - 1).astype(np.int64),
            np.clip(cnt[:, 1] / c * 10, 0, N_BINS['hc'] - 1).astype(np.int64),
            np.clip(cnt[:, 3] / c * 10, 0, N_BINS['oc'] - 1).astype(np.int64))


@lru_cache(maxsize=1)
def formula_prior():
    """Log-priors built by scripts/build_formula_prior.py (train split only)."""
    with np.load(PRIOR_PATH) as z:
        return {k: z[k] for k in z.files}


def prior_score(cnt):
    p = formula_prior()
    r, hc, oc = prior_bins(cnt)
    return p['count'][np.arange(7), cnt[:, CORE]].sum(1) + p['rdbe'][r] + p['hc'][hc] + p['oc'][oc]


def rank_formulas(mass, iso=None, ppm=10.0, sigma_ppm=2.0, w_iso=0.0, require_even=True, top=16):
    """Decompose + rank. -> (counts int16[k,10], score float32[k], ppm float32[k]), best first.
    sigma/w_iso tuned on train-split enveda spectra (S06): top1 0.707, top16 0.999; isotope term didn't help."""
    cnt, err = decompose(mass, ppm, require_even)
    if not len(cnt):
        return cnt, np.zeros(0, np.float32), np.zeros(0, np.float32)
    score = -0.5 * (err / sigma_ppm) ** 2 + prior_score(cnt)
    if iso is not None:
        score += w_iso * iso_score(cnt, iso)
    o = np.argsort(-score, kind='stable')[:top]
    return cnt[o], score[o].astype(np.float32), err[o].astype(np.float32)


def formula_vector(f):
    """'C6H12O6' -> int16[10] in EL order; None if it has elements outside EL."""
    d = parse_formula(f)
    if d is None or any(e not in EL for e in d):
        return None
    return np.array([d.get(e, 0) for e in EL], np.int16)
