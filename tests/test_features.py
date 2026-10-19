import numpy as np

from casmi.features import featurize
from casmi.features.formula import EL, PROTON, decompose, formula_vector, neutral_mass, parse_adduct
from casmi.features.peaks import FEATS, LOSSES, peak_features

GLUCOSE = 180.063388


def test_adducts():
    assert abs(neutral_mass(GLUCOSE + PROTON, '[M+H]+') - GLUCOSE) < 1e-6
    assert abs(neutral_mass(GLUCOSE - PROTON, '[M-H]-') - GLUCOSE) < 1e-6
    assert abs(neutral_mass(2 * GLUCOSE + 22.989221, '[2M+Na]+') - GLUCOSE) < 1e-5
    assert abs(neutral_mass(GLUCOSE + 46.005479 - PROTON, '[M+CH2O2-H]-') - GLUCOSE) < 1e-5
    assert parse_adduct('[M+2H]2+')[1] == 2
    assert parse_adduct('[Cat]2+') is None and np.isnan(neutral_mass(300.0, None))


def test_decompose_finds_glucose():
    cnt, ppm = decompose(GLUCOSE, ppm=5)
    assert (cnt == formula_vector('C6H12O6')).all(1).any()
    assert np.all(np.abs(ppm) <= 5)
    assert len(EL) == cnt.shape[1]


def test_peak_features():
    prec = GLUCOSE + PROTON
    mz = np.array([prec - LOSSES['H2O'], prec])
    f = peak_features(mz, np.array([1.0, 0.5]), prec, formula_vector('C6H12O6'), 1)
    col = {k: i for i, k in enumerate(FEATS)}
    assert f[0, col['loss_id']] > 0 and f[1, col['is_prec']] == 1
    assert f[0, col['sub_hit']] == 1 and abs(f[0, col['sub_ppm']]) < 1
    assert f[0, [col['sub_C'], col['sub_H'], col['sub_O']]].tolist() == [6, 10, 5]  # C6H10O5 + H+
    assert f[0, col['pair_loss']] == 1


def test_featurize_shapes():
    prec = GLUCOSE + PROTON
    out = featurize([60.0, 145.0495, prec - LOSSES['H2O'], prec], [0.2, 0.4, 1.0, 0.5], prec, '[M+H]+')
    assert out['peakfeat'].shape == (4, len(FEATS)) and out['formulas'].shape[1] == 10
    assert (out['formulas'] == formula_vector('C6H12O6')).all(1).any()
