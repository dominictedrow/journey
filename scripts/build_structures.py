"""P2.1: one row per unique train SMILES -> data/processed/v1/structures.parquet.
Columns: smiles, ik14 (tautomer-canonical, casmi.metric.ik14), ik14_raw (train inchikey14),
formula, exact_mass, scaffold (Murcko, stereo stripped), n_spectra. Skips if output exists."""
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

sys.path.insert(0, '.')
from casmi.metric import ik14  # noqa: E402

OUT = 'data/processed/v1/structures.parquet'


def describe(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, None, None, None, None
    Chem.RemoveStereochemistry(mol)
    try:
        scaf = MurckoScaffoldSmiles(mol=mol)
    except (ValueError, RuntimeError):
        scaf = None
    return smi, ik14(smi), CalcMolFormula(mol), CalcExactMolWt(mol), scaf


def main():
    if os.path.exists(OUT):
        print(f'{OUT} exists; reuse it (delete is a gate)')
        return
    t0 = time.time()
    counts, raw = Counter(), {}
    for b in pq.ParquetFile('data/train.parquet').iter_batches(batch_size=65536, columns=['normalized_smiles', 'inchikey14']):
        for smi, ik in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            counts[smi] += 1
            raw.setdefault(smi, ik)
    smiles = sorted(s for s in counts if s)
    print(f'scan {time.time() - t0:.0f}s: {len(smiles)} unique smiles, {counts.get(None, 0)} null', flush=True)

    t1 = time.time()
    with Pool(os.cpu_count() - 2) as p:
        rows = sorted(p.imap_unordered(describe, smiles, chunksize=256))
    print(f'rdkit {time.time() - t1:.0f}s', flush=True)

    cols = list(zip(*rows))
    tbl = pa.table({
        'smiles': cols[0], 'ik14': cols[1], 'ik14_raw': [raw[s] for s in cols[0]],
        'formula': cols[2], 'exact_mass': pa.array(cols[3], pa.float64()), 'scaffold': cols[4],
        'n_spectra': pa.array([counts[s] for s in cols[0]], pa.int32()),
    })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pq.write_table(tbl, OUT, compression='zstd')
    n = tbl.num_rows
    stats = {
        'rows': n,
        'ik14_null': tbl['ik14'].null_count, 'scaffold_null': tbl['scaffold'].null_count,
        'unique_ik14': len(set(tbl['ik14'].to_pylist()) - {None}),
        'unique_ik14_raw': len(set(tbl['ik14_raw'].to_pylist()) - {None}),
        'ik14_ne_raw': sum(a is not None and a != b for a, b in zip(tbl['ik14'].to_pylist(), tbl['ik14_raw'].to_pylist())),
        'unique_scaffold': len(set(tbl['scaffold'].to_pylist()) - {None, ''}),
        'acyclic (scaffold empty)': tbl['scaffold'].to_pylist().count(''),
    }
    manifest = {
        'step': 'P2.1', 'script': 'scripts/build_structures.py',
        'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'files': {'structures.parquet': hashlib.sha256(open(OUT, 'rb').read()).hexdigest()},
        'structures': stats, 'seconds': round(time.time() - t0),
    }
    mp = os.path.join(os.path.dirname(OUT), 'manifest.json')
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    m['structures'] = manifest
    json.dump(m, open(mp, 'w'), indent=2)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
