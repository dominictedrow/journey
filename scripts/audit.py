"""P1.2 streaming data audit. Prints stats to stdout (redirect to logs/audit.log) and saves
per-spectrum arrays to logs/audit_arrays.npz for plotting. Never loads train whole."""
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ISO = 1.003355  # 13C - 12C


def q(a, ps=(0, 1, 5, 50, 95, 99, 100)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return {p: round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else {}


def top(c, n=40):
    tot = sum(c.values())
    return [(k, v, f'{100 * v / tot:.2f}%') for k, v in c.most_common(n)]


def row_minmax_and_flags(mzs, prec):
    """Per-row min/max m/z, peaks above precursor, and whether precursor / M+1 isotope peaks are present."""
    off = mzs.offsets.to_numpy()
    flat = mzs.values.to_numpy(zero_copy_only=False)
    n = np.diff(off)
    nz = n > 0
    starts = off[:-1][nz]
    mn = np.full(len(n), np.nan); mx = np.full(len(n), np.nan)
    mn[nz] = np.minimum.reduceat(flat, starts)
    mx[nz] = np.maximum.reduceat(flat, starts)
    row = np.repeat(np.arange(len(n)), n)
    p = prec[row]
    tol = np.maximum(0.01, p * 10e-6)
    above = np.bincount(row, weights=(flat > p + 0.5), minlength=len(n))
    has_prec = np.bincount(row, weights=(np.abs(flat - p) <= tol), minlength=len(n)) > 0
    has_iso = np.bincount(row, weights=(np.abs(flat - p - ISO) <= tol), minlength=len(n)) > 0
    return mn, mx, above, has_prec, has_iso, n


def row_max(lst):
    off = lst.offsets.to_numpy(); flat = lst.values.to_numpy(zero_copy_only=False)
    out = np.full(len(off) - 1, np.nan); nz = np.diff(off) > 0
    out[nz] = np.maximum.reduceat(flat, off[:-1][nz]); return out


def audit(path, train):
    pf = pq.ParquetFile(path)
    cats = ['ingest_lib', 'instrument_type', 'adduct', 'ionization_mode', 'collision_energy_orig_units'] + (['adduct_orig'] if train else [])
    cnt = {c: Counter() for c in cats}
    ce_len, ce_first, prec_mz, err_ppm = Counter(), [], [], []
    npk, mn_all, mx_all, above_all, hp_all, hi_all, inten_max = [], [], [], [], [], [], []
    per_struct, lib_structs, nulls = Counter(), defaultdict(set), Counter()
    cols = [c for c in pf.schema_arrow.names if c not in ('normalized_smiles', 'inchikey', 'molecular_formula', 'spectrum_id', 'molecule_id')]
    for b in pf.iter_batches(batch_size=65536, columns=cols):
        for c in cats:
            if c in b.schema.names:
                cnt[c].update(b.column(c).to_pylist())
        for c in b.schema.names:
            nulls[c] += b.column(c).null_count
        ce = b.column('collision_energy_ev')
        ce_len.update(pc.list_value_length(ce).fill_null(-1).to_pylist())
        ce_first.append(pc.list_element(pc.if_else(pc.greater(pc.list_value_length(ce).fill_null(0), 0), ce, None), 0).to_numpy(zero_copy_only=False))
        prec = b.column('precursor_mz').to_numpy(zero_copy_only=False)
        prec_mz.append(prec)
        if train:
            err_ppm.append(b.column('precursor_error_ppm').to_numpy(zero_copy_only=False))
            ik = b.column('inchikey14').to_pylist()
            per_struct.update(ik)
            for lib, k in zip(b.column('ingest_lib').to_pylist(), ik):
                lib_structs[lib].add(k)
        mn, mx, above, hp, hi, n = row_minmax_and_flags(b.column('ms2_mzs'), prec)
        npk.append(n); mn_all.append(mn); mx_all.append(mx); above_all.append(above); hp_all.append(hp); hi_all.append(hi)
        inten_max.append(row_max(b.column('ms2_normalized_intensities')))
    cat = np.concatenate
    npk, mn_all, mx_all, above_all, hp_all, hi_all, prec_mz = map(cat, (npk, mn_all, mx_all, above_all, hp_all, hi_all, prec_mz))
    ce_first = cat(ce_first).astype(float)
    tag = 'TRAIN' if train else 'TEST'
    print(f'\n===== {tag} {path}: {pf.metadata.num_rows} rows =====')
    print('nulls:', {k: v for k, v in nulls.items() if v})
    for c in cats:
        print(f'\n[{c}] unique={len(cnt[c])}', top(cnt[c], 40 if c != 'adduct_orig' else 15))
    print('\n[collision_energy_ev list length]', sorted(ce_len.items())[:20])
    print('[CE first eV quantiles]', q(ce_first), 'nan frac', round(float(np.isnan(ce_first).mean()), 4))
    print('[precursor_mz quantiles]', q(prec_mz))
    if train:
        e = cat(err_ppm)
        print('[precursor_error_ppm quantiles]', q(e), '| abs:', q(np.abs(e)), 'nan frac', round(float(np.isnan(e).mean()), 4))
        for t in (5, 10, 20, 50):
            print(f'   |err| <= {t} ppm: {100 * float((np.abs(e) <= t).mean()):.2f}%')
    print('[peaks/spectrum quantiles]', q(npk, (0, 1, 50, 90, 95, 99, 99.9, 100)), 'zero-peak spectra', int((npk == 0).sum()))
    print('[min m/z quantiles]', q(mn_all), '\n[max m/z quantiles]', q(mx_all))
    print('[spectra with peaks > precursor+0.5]', f'{100 * float((above_all > 0).mean()):.2f}%', '| peaks above prec quantiles', q(above_all, (50, 95, 99, 100)))
    print('[precursor peak present (10ppm/0.01Da)]', f'{100 * float(hp_all.mean()):.2f}%', '| M+1 isotope peak present', f'{100 * float(hi_all.mean()):.2f}%')
    print('[max normalized intensity per spectrum]', q(cat(inten_max)))
    if train:
        sps = np.array(list(per_struct.values()))
        print(f'\n[structures (inchikey14)] {len(per_struct)}', '| spectra/structure quantiles', q(sps, (50, 90, 99, 100)), '| singletons', int((sps == 1).sum()))
        print('[structures per library]', sorted(((k, len(v)) for k, v in lib_structs.items()), key=lambda x: -x[1]))
        libs = list(lib_structs)
        only = {l: len(lib_structs[l] - set().union(*(lib_structs[m] for m in libs if m != l))) for l in libs}
        print('[structures unique to library]', sorted(only.items(), key=lambda x: -x[1]))
    np.savez_compressed(f'logs/audit_{tag.lower()}.npz', npk=npk, mn=mn_all, mx=mx_all, above=above_all, prec=prec_mz, ce=ce_first, **({'err': cat(err_ppm)} if train else {}))


if __name__ == '__main__':
    which = sys.argv[1:] or ['test', 'train']
    for w in which:
        audit(f'data/{w}.parquet', w == 'train')
