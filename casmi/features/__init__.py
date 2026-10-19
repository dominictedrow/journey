"""Single entry point for spectrum features: shard building (P2.5) and Kaggle inference both call featurize()."""
import numpy as np

from casmi.features.clean import clean_peaks
from casmi.features.formula import iso_score, neutral_mass, parse_adduct, rank_formulas, rdbe
from casmi.features.peaks import peak_features

TOP_FORMULAS = 16


def featurize(mzs, intens, prec, adduct):
    """Raw spectrum -> dict(mz float32[n], inten float32[n], peakfeat float32[n,F], iso float32[6],
    neutral_mass float64, formulas int16[k,10], formula_score/ppm/iso_score/rdbe float32[k]); k <= TOP_FORMULAS."""
    mz, inten, iso = clean_peaks(mzs, intens, prec)
    a = parse_adduct(adduct)
    m = neutral_mass(prec, adduct)
    # [M]+ / [M]- are charged molecules: odd-electron-count formulas allowed
    cnt, score, ppm = rank_formulas(m, iso, require_even=not (a and a[2] == 0.0), top=TOP_FORMULAS)
    charge = 1 if a is None or a[1] > 0 else -1
    return {
        'mz': mz, 'inten': inten, 'iso': iso, 'neutral_mass': m,
        'peakfeat': peak_features(mz, inten, prec, cnt[0] if len(cnt) else None, charge),
        'formulas': cnt, 'formula_score': score, 'formula_ppm': ppm,
        'formula_iso_score': iso_score(cnt, iso).astype(np.float32), 'formula_rdbe': rdbe(cnt).astype(np.float32),
    }
