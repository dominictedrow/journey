"""P2.5: featurize train.parquet -> data/processed/v1/{train,v_lib,v_pool,v_scaf,v_novel}/ .npy memmaps sorted by molecule.
Pass 1: one worker per row group writes unsorted temp arrays. Pass 2: per split, scatter into molecule order.
Usage: python scripts/build_shards.py [--smoke N]   (smoke: first N rows of 2 row groups -> data/processed/_smoke)"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap

from casmi.data.splits import VAL, spectrum_role
from casmi.features import TOP_FORMULAS, featurize
from casmi.features.clean import P, PREC_RANGE, keep_row
from casmi.features.formula import EL, formula_vector
from casmi.features.peaks import F

SRC = 'data/train.parquet'
V1 = 'data/processed/v1'
SPLITS = ('train',) + VAL
MULTIMER = re.compile(r'^\[\d+M')  # [2M..], [3M..], [4M..] (D-020)
VOCAB_COLS = ('adduct', 'instrument_type', 'ionization_mode', 'ingest_lib')
COLS = ['normalized_smiles', 'precursor_mz', 'precursor_error_ppm', 'ms2_mzs', 'ms2_normalized_intensities',
        'collision_energy_ev'] + list(VOCAB_COLS)
FP_BITS = 2048
K = TOP_FORMULAS
SPEC = {  # per-spectrum arrays: name -> (dtype, trailing shape)
    'mz': (np.float32, (P,)), 'inten': (np.float16, (P,)), 'peakfeat': (np.float16, (P, F)), 'npeaks': (np.uint16, ()),
    'prec': (np.float64, ()), 'neutral_mass': (np.float64, ()), 'ce': (np.float32, (2,)), 'iso': (np.float32, (6,)),
    'adduct': (np.uint16, ()), 'instrument': (np.uint8, ()), 'polarity': (np.uint8, ()), 'lib': (np.uint8, ()),
    'formulas': (np.int16, (K, len(EL))), 'nformulas': (np.uint8, ()), 'formula_score': (np.float32, (K,)),
    'formula_ppm': (np.float32, (K,)), 'formula_iso_score': (np.float32, (K,)), 'formula_rdbe': (np.float32, (K,)),
    'src_row': (np.int32, ()),  # row in train.parquet, for the parity test (P2.7)
}
KEYS = {'mol': (np.int32, ()), 'dest': (np.int8, ())}  # temp only: global molecule id, index into SPLITS
META = dict(zip(VOCAB_COLS, ('adduct', 'instrument', 'polarity', 'lib')))

G = {}


def init(vocab):
    s = pq.read_table(f'{V1}/structures.parquet', columns=['smiles', 'ik14'])
    sp = pq.read_table(f'{V1}/splits.parquet', columns=['ik14', 'split'])
    mol_id = {k: i for i, k in enumerate(sp['ik14'].to_pylist())}
    G['mol'] = {smi: mol_id.get(k, -1) for smi, k in zip(s['smiles'].to_pylist(), s['ik14'].to_pylist())}
    G['split'] = sp['split'].to_pylist()
    G['vocab'] = vocab


def work(job):
    rg, limit, tmp = job
    pf = pq.ParquetFile(SRC)
    cap = limit or pf.metadata.row_group(rg).num_rows
    base = sum(pf.metadata.row_group(j).num_rows for j in range(rg))
    d = f'{tmp}/rg{rg:02d}'
    os.makedirs(d, exist_ok=True)
    out = {k: open_memmap(f'{d}/{k}.npy', 'w+', dt, (cap,) + shp) for k, (dt, shp) in {**SPEC, **KEYS}.items()}
    stats, i, seen = Counter(), 0, 0
    for b in pf.iter_batches(batch_size=4096, row_groups=[rg], columns=COLS):
        c = {n: b.column(n).to_pylist() for n in COLS}
        for r in range(b.num_rows):
            if limit and seen >= limit:
                break
            seen += 1
            a, prec = c['adduct'][r], c['precursor_mz'][r]
            if a and MULTIMER.match(a):
                stats['drop_multimer'] += 1
                continue
            m = G['mol'].get(c['normalized_smiles'][r], -1)
            split = G['split'][m] if m >= 0 else None
            role = spectrum_role(split, c['ingest_lib'][r].startswith('enveda')) if split else None
            if role is None:
                stats['drop_unused' if split else 'drop_no_mol'] += 1
                continue
            if role == 'train' and not PREC_RANGE[0] <= prec <= PREC_RANGE[1]:
                stats['drop_prec'] += 1
                continue
            f = featurize(c['ms2_mzs'][r], c['ms2_normalized_intensities'][r], prec, a)
            n = len(f['mz'])
            # queries mirror test: never filtered (keep_row is train-only)
            if role == 'train' and not keep_row(prec, c['precursor_error_ppm'][r], n):
                stats['drop_keep_row'] += 1
                continue
            out['mz'][i, :n] = f['mz']
            out['inten'][i, :n] = f['inten']
            out['peakfeat'][i, :n] = f['peakfeat']
            out['npeaks'][i] = n
            out['prec'][i] = prec
            out['neutral_mass'][i] = f['neutral_mass']
            ce = c['collision_energy_ev'][r]
            out['ce'][i] = (min(ce), max(ce)) if ce else (np.nan, np.nan)
            out['iso'][i] = f['iso']
            for col, name in META.items():
                out[name][i] = G['vocab'][col].get(c[col][r], 0)
            k = len(f['formulas'])
            out['nformulas'][i] = k
            for name in ('formulas', 'formula_score', 'formula_ppm', 'formula_iso_score', 'formula_rdbe'):
                out[name][i, :k] = f[name]
            out['src_row'][i] = base + seen - 1
            out['mol'][i] = m
            out['dest'][i] = SPLITS.index('train' if role == 'train' else split)
            i += 1
            stats['kept_' + SPLITS[out['dest'][i - 1]]] += 1
        if limit and seen >= limit:
            break
    for a in out.values():
        a.flush()
    return rg, i, stats


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles), FP_BITS // 8), np.uint8)
    for j, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[j] = np.packbits(gen.GetFingerprintAsNumPy(mol))
    return out


def merge(out_root, tmp, counts, pool):
    sp = pq.read_table(f'{V1}/splits.parquet', columns=['ik14', 'split'])
    ik14 = sp['ik14'].to_pylist()
    s = pq.read_table(f'{V1}/structures.parquet', columns=['smiles', 'ik14', 'formula'])
    first = {}  # ponytail: one representative SMILES per ik14 (first in structures.parquet); fp ignores stereo anyway
    for smi, k, fo in zip(s['smiles'].to_pylist(), s['ik14'].to_pylist(), s['formula'].to_pylist()):
        first.setdefault(k, (smi, fo))
    rgs = sorted(counts)
    files = {}
    for di, name in enumerate(SPLITS):
        keys = {rg: (np.load(f'{tmp}/rg{rg:02d}/mol.npy', mmap_mode='r')[:counts[rg]],
                     np.load(f'{tmp}/rg{rg:02d}/dest.npy', mmap_mode='r')[:counts[rg]]) for rg in rgs}
        rows = {rg: np.flatnonzero(dst == di) for rg, (mol, dst) in keys.items()}
        mol = np.concatenate([keys[rg][0][rows[rg]] for rg in rgs])
        n = len(mol)
        if n == 0:
            continue
        order = np.argsort(mol, kind='stable')  # stable: keeps file order within a molecule
        pos = np.empty(n, np.int64)
        pos[order] = np.arange(n)
        d = f'{out_root}/{name}'
        os.makedirs(d, exist_ok=True)
        for k, (dt, shp) in SPEC.items():
            out = open_memmap(f'{d}/{k}.npy', 'w+', dt, (n,) + shp)
            start = 0
            for rg in rgs:
                r = rows[rg]
                out[pos[start:start + len(r)]] = np.load(f'{tmp}/rg{rg:02d}/{k}.npy', mmap_mode='r')[r]
                start += len(r)
            out.flush()
            del out
        mols, mol_idx = np.unique(mol, return_inverse=True)
        np.save(f'{d}/mol_idx.npy', mol_idx[order].astype(np.int32))
        np.save(f'{d}/mol_offsets.npy', np.append(np.searchsorted(mol[order], mols), n).astype(np.int64))
        smiles = [first[ik14[m]][0] for m in mols]
        forms = [first[ik14[m]][1] for m in mols]
        fv = [formula_vector(fo) if fo else None for fo in forms]
        np.save(f'{d}/mol_formula.npy', np.stack([v if v is not None else np.full(len(EL), -1, np.int16) for v in fv]))
        chunks = [smiles[j:j + 2000] for j in range(0, len(smiles), 2000)]
        np.save(f'{d}/mol_fp.npy', np.concatenate(pool.map(fingerprints, chunks)))
        pq.write_table(pa.table({'ik14': [ik14[m] for m in mols], 'smiles': smiles, 'formula': forms}),
                       f'{d}/mols.parquet', compression='zstd')
        files[name] = {'spectra': n, 'molecules': len(mols),
                       'sha256': {fn: sha(f'{d}/{fn}') for fn in sorted(os.listdir(d))}}
        print(name, n, len(mols), flush=True)
    return files


def main():
    smoke = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == '--smoke' else 0
    out_root = 'data/processed/_smoke' if smoke else V1
    if not smoke and os.path.exists(f'{V1}/train'):
        print(f'{V1}/train exists; shards are frozen (regenerating is a gate)')
        return
    if smoke:
        shutil.rmtree(out_root, ignore_errors=True)
    tmp = f'{out_root}/_tmp'
    t0 = time.time()
    t = pq.read_table(SRC, columns=list(VOCAB_COLS))
    vocab = {c: {v: i + 1 for i, v in enumerate(sorted(x for x in set(t[c].to_pylist()) if x is not None))}
             for c in VOCAB_COLS}  # 0 = unknown / missing (test values unseen in train)
    del t
    os.makedirs(out_root, exist_ok=True)
    json.dump(vocab, open(f'{out_root}/vocab.json', 'w'), indent=1)
    nrg = pq.ParquetFile(SRC).metadata.num_row_groups
    jobs = [(rg, smoke, tmp) for rg in (range(2) if smoke else range(nrg))]
    counts, stats = {}, Counter()
    with Pool(14, initializer=init, initargs=(vocab,)) as pool:
        for rg, n, st in pool.imap_unordered(work, jobs):
            counts[rg] = n
            stats.update(st)
            print(f'rg {rg} kept {n} ({time.time() - t0:.0f}s)', flush=True)
        t1 = time.time()
        files = merge(out_root, tmp, counts, pool)
    shutil.rmtree(tmp)
    info = {
        'step': 'P2.5', 'decisions': ['D-020'], 'script': 'scripts/build_shards.py',
        'config': {'P': P, 'F': F, 'top_formulas': K, 'fp': f'morgan r2 {FP_BITS} packed', 'drop': 'multimer [nM..]',
                   'train_filter': 'keep_row', 'query_filter': 'none', 'sort': 'mol (splits.parquet order), then file order'},
        'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'stats': dict(stats), 'splits': files, 'vocab.json': sha(f'{out_root}/vocab.json'),
        'seconds': {'featurize': round(t1 - t0), 'merge': round(time.time() - t1)},
    }
    mp = f'{V1}/manifest.json'
    if smoke:
        json.dump(info, open(f'{out_root}/manifest.json', 'w'), indent=2)
    else:
        m = json.load(open(mp))
        m['shards'] = info
        json.dump(m, open(mp, 'w'), indent=2)
    print(json.dumps({'stats': dict(stats), 'seconds': info['seconds']}, indent=1))


if __name__ == '__main__':
    main()
