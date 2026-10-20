"""P4.2 candidate pool (D-024) -> data/processed/pool_v1/.
ingest <source>: train | coconut | lotus | pubchem -> raw/<source>_<chunk>.parquet (each chunk sorted by mass)
merge: per 25 Da bin, dedup on tautomer-canonical ik14 -> pool/bin_<lo>.parquet sorted by mass (+ manifest.json)
Filters: all sources must parse; PubChem only (D-024): single component, net charge 0, no isotopes, elements in EL, 150–1,200 Da.
Usage: python scripts/build_pool.py ingest pubchem | merge"""
import glob
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.features.formula import EL, parse_formula

OUT = 'data/processed/pool_v1'
EXT = 'data/external'
SRC = {'train': 1, 'coconut': 2, 'lotus': 4, 'pubchem': 8}
PRIORITY = {'train': 0, 'coconut': 1, 'lotus': 2, 'pubchem': 3}  # whose SMILES is kept on ik14 collision
PUBCHEM_MASS = (150.0, 1200.0)
CHUNK = 20_000
BIN = 25.0
ELS = set(EL)


def worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def process(job):
    """job: (source, chunk_id, rows) with rows = list of (smiles, cid, level, np_likeness[, ik14]) -> rows kept.
    A precomputed ik14 (train: P2.1) skips the tautomer step, which costs ~2 ms median / ~36 ms mean on NPs."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from casmi.metric import ik14
    taut = ik14.__wrapped__  # skip the lru_cache: every structure is seen once
    source, cid_chunk, rows = job
    out = {k: [] for k in ('ik14', 'smiles', 'mass', 'formula', 'cid', 'level', 'np_likeness', 'taut')}
    pc = source == 'pubchem'  # D-024 filters apply to PubChem only; curated sources just need to parse

    def one(smi, pre):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        if pc and ('.' in smi or Chem.GetFormalCharge(m) != 0 or any(a.GetIsotope() for a in m.GetAtoms())):
            return None
        f = rdMolDescriptors.CalcMolFormula(m)
        d = parse_formula(f)
        if pc and (d is None or not set(d) <= ELS):
            return None
        mass = rdMolDescriptors.CalcExactMolWt(m)
        if pc and not PUBCHEM_MASS[0] <= mass <= PUBCHEM_MASS[1]:
            return None
        # D-025: PubChem gets the plain InChIKey14 (~0.2 ms vs ~12 ms); inference re-keys top candidates by tautomer
        k = pre[0] if pre else (Chem.MolToInchiKey(m)[:14] if pc else taut(smi))
        return (k, mass, f) if k else None

    for smi, cid, level, npl, *pre in rows:
        try:
            r = one(smi, pre)
        except Exception:  # RDKit raises on a few malformed PubChem records (e.g. KekulizeException in InChI)
            r = None
        if r is None:
            continue
        k, mass, f = r
        for col, v in zip(out, (k, smi, mass, f, cid, level, npl, not pc)):
            out[col].append(v)
    t = pa.table({
        'ik14': out['ik14'], 'smiles': out['smiles'], 'mass': pa.array(out['mass'], pa.float64()),
        'formula': out['formula'], 'cid': pa.array(out['cid'], pa.int64()), 'level': pa.array(out['level'], pa.int8()),
        'np_likeness': pa.array(out['np_likeness'], pa.float32()), 'taut': pa.array(out['taut'], pa.bool_()),
        'src': pa.array([SRC[source]] * len(out['ik14']), pa.uint8()),
        'prio': pa.array([PRIORITY[source]] * len(out['ik14']), pa.uint8()),
    }).sort_by('mass')
    pq.write_table(t, f'{OUT}/raw/{source}_{cid_chunk:05d}.parquet', row_group_size=8192)
    return len(rows), t.num_rows


def num(x, cast=float):
    try:
        return cast(x)
    except (TypeError, ValueError):
        return None


def rows_train():
    t = pq.read_table('data/processed/v1/structures.parquet', columns=['smiles', 'ik14']).to_pydict()
    yield from ((s, None, None, None, k) for s, k in zip(t['smiles'], t['ik14']))


def rows_coconut():
    z = zipfile.ZipFile(f'{EXT}/coconut/coconut_csv-10-2026.zip')
    with z.open(z.infolist()[0]) as f:
        r = pl.read_csv(f.read(), columns=['canonical_smiles', 'annotation_level', 'np_likeness'], infer_schema=False)
    for s, lv, npl in r.iter_rows():
        yield s, None, num(lv, int), num(npl)


def rows_lotus():
    r = pl.read_csv(f'{EXT}/lotus/260413_frozen_metadata.csv.gz', columns=['structure_inchikey', 'structure_smiles', 'structure_cid'],
                    infer_schema=False).unique('structure_inchikey')
    for _, s, cid in r.iter_rows():
        yield s, num(cid, int), None, None


def rows_pubchem():
    with gzip.open(f'{EXT}/pubchem/CID-SMILES.gz', 'rt') as f:
        for line in f:
            cid, _, smi = line.rstrip('\n').partition('\t')
            if '.' not in smi:  # cheap text pre-filter before pickling to workers
                yield smi, int(cid), None, None


def chunks(source, it):
    buf, i = [], 0
    for r in it:
        buf.append(r)
        if len(buf) == CHUNK:
            yield source, i, buf
            buf, i = [], i + 1
    if buf:
        yield source, i, buf


def ingest(source):
    os.makedirs(f'{OUT}/raw', exist_ok=True)
    done = {os.path.basename(p) for p in glob.glob(f'{OUT}/raw/{source}_*.parquet')}  # resumable
    jobs = (j for j in chunks(source, globals()[f'rows_{source}']()) if f'{source}_{j[1]:05d}.parquet' not in done)
    t0, n_in, n_out = time.time(), 0, 0
    with Pool(14, initializer=worker_init, maxtasksperchild=200) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(process, jobs)):
            n_in, n_out = n_in + a, n_out + b
            if i % 20 == 0:
                print(f'{source}: {n_in:,} in, {n_out:,} kept ({time.time() - t0:.0f}s)', flush=True)
    print(f'{source} done: {n_in:,} in, {n_out:,} kept ({time.time() - t0:.0f}s) [skipped {len(done)} done chunks]', flush=True)


def merge():
    files = sorted(glob.glob(f'{OUT}/raw/*.parquet'))
    os.makedirs(f'{OUT}/pool', exist_ok=True)
    for p in files:  # chunks written before D-025 lack 'taut'; they all carry tautomer keys
        if 'taut' not in pq.read_schema(p).names:
            t = pq.read_table(p)
            pq.write_table(t.append_column('taut', pa.array([True] * t.num_rows, pa.bool_())), p, row_group_size=8192)
    lazy = pl.scan_parquet(files)
    lo_hi = lazy.select(pl.col('mass').min().alias('lo'), pl.col('mass').max().alias('hi')).collect().row(0)
    stats, sha = {}, {}
    for lo in np.arange(np.floor(lo_hi[0] / BIN) * BIN, lo_hi[1] + BIN, BIN):
        b = (lazy.filter((pl.col('mass') >= lo) & (pl.col('mass') < lo + BIN))
             .sort(['ik14', 'prio', 'cid'], nulls_last=True)
             .group_by('ik14', maintain_order=True)
             .agg(pl.col('smiles', 'mass', 'formula').first(), pl.col('src').bitwise_or(), pl.col('cid').min(),
                  pl.col('level').max(), pl.col('np_likeness').max(), pl.col('taut').max())
             .sort('mass').collect())
        if b.height == 0:
            continue
        p = f'{OUT}/pool/bin_{int(lo):04d}.parquet'
        b.write_parquet(p, row_group_size=65536)
        stats[int(lo)] = b.height
        sha[os.path.basename(p)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        print(f'bin {int(lo)}: {b.height:,}', flush=True)
    src = pl.scan_parquet(f'{OUT}/pool/*.parquet').select(
        [(pl.col('src') & v > 0).sum().alias(k) for k, v in SRC.items()] + [pl.len().alias('total')]).collect().row(0, named=True)
    json.dump({'step': 'P4.2', 'decisions': ['D-024'], 'script': 'scripts/build_pool.py', 'by_source': src,
               'bins': stats, 'sha256': sha, 'config': {'pubchem_mass': PUBCHEM_MASS, 'bin_da': BIN, 'elements': EL},
               'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()},
              open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(src))


if __name__ == '__main__':
    ingest(sys.argv[2]) if sys.argv[1] == 'ingest' else merge()
