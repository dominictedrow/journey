"""P2.2 spectrum cleaning, shared by shard building (P2.5) and Kaggle inference (parity test).
Thresholds from reports/data_audit.md (P1.5)."""
import numpy as np

ISO = 1.003355          # 13C - 12C
P = 128                 # peaks kept per spectrum
NOISE_FLOOR = 0.001     # x base peak, applied after the above-precursor cut
ABOVE_PREC = 0.5        # Da; peaks above precursor + this are dropped (isotope evidence kept in iso features)
MIN_MZ = 20.0
MAX_PPM = 20.0          # train row filter only; test has no ppm column
PREC_RANGE = (50.0, 2000.0)
MIN_PEAKS = 3


def iso_features(mz, inten, prec):
    """[I_M, I_M+1, I_M+2, has_M, has_M+1, has_M+2]; intensities relative to the raw base peak."""
    # ponytail: M+2 assumed at 2*ISO; 34S/37Cl/18O M+2 sit ~0.01 Da lower, inside the 0.01 Da tol only at low mass
    tol = max(0.01, prec * 10e-6)
    out = np.zeros(6, np.float32)
    for k in range(3):
        hit = np.abs(mz - prec - k * ISO) <= tol
        if hit.any():
            out[k] = inten[hit].max()
            out[3 + k] = 1
    return out


def clean_peaks(mz, inten, prec, p=P):
    """-> (mz float32[n], sqrt intensity float32[n] with max 1, iso float32[6]); n <= p, sorted by m/z."""
    mz = np.asarray(mz, np.float64)
    inten = np.asarray(inten, np.float64)
    iso = iso_features(mz, inten, prec)  # before the above-precursor cut, which removes M+1/M+2
    keep = (mz >= MIN_MZ) & (mz <= prec + ABOVE_PREC) & (inten > 0)
    mz, inten = mz[keep], inten[keep]
    if len(inten):
        inten = inten / inten.max()
        keep = inten >= NOISE_FLOOR
        mz, inten = mz[keep], inten[keep]
    if len(inten) > p:
        top = np.argpartition(-inten, p - 1)[:p]
        mz, inten = mz[top], inten[top]
    order = np.argsort(mz)
    return mz[order].astype(np.float32), np.sqrt(inten[order]).astype(np.float32), iso


def keep_row(prec, ppm, n_clean):
    """Train-only row filter. Never apply to test: every test spectrum must be scored."""
    ppm_ok = ppm is None or np.isnan(ppm) or abs(ppm) <= MAX_PPM
    return PREC_RANGE[0] <= prec <= PREC_RANGE[1] and ppm_ok and n_clean >= MIN_PEAKS
