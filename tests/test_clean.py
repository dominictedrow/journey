import numpy as np

from casmi.features.clean import ISO, clean_peaks, keep_row


def test_clean_peaks():
    prec = 300.0
    mz = [10.0, 50.0, 100.0, 150.0, 200.0, prec, prec + ISO, prec + 2 * ISO, 400.0]
    it = [0.5, 0.0004, 0.25, 0.04, 1.0, 0.6, 0.12, 0.02, 2.0]
    m, i, iso = clean_peaks(mz, it, prec)
    # m/z < 20 and > prec+0.5 dropped; 0.0004/1.0 below floor; base peak re-normalized after the cut
    assert m.tolist() == [100.0, 150.0, 200.0, prec]
    np.testing.assert_allclose(i, np.sqrt([0.25, 0.04, 1.0, 0.6]), rtol=1e-6)
    np.testing.assert_allclose(iso, [0.6, 0.12, 0.02, 1, 1, 1], rtol=1e-6)


def test_top_p_and_empty():
    m, i, iso = clean_peaks(np.arange(100, 300, dtype=float), np.linspace(0.01, 1, 200), 500.0, p=5)
    assert len(m) == 5 and m.min() == 295 and i.max() == 1
    assert iso[3:].sum() == 0
    m, i, _ = clean_peaks([600.0], [1.0], 500.0)
    assert len(m) == 0


def test_keep_row():
    assert keep_row(300.0, 5.0, 10)
    assert keep_row(300.0, float('nan'), 10)
    assert not keep_row(300.0, 25.0, 10)
    assert not keep_row(2.4e6, 1.0, 10)
    assert not keep_row(300.0, 1.0, 2)
