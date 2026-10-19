"""P1.3: are local test spectra copies of train spectra? Matches on (precursor_mz, num_peaks,
first 3 m/z) rounded to 4 dp. Writes logs/test_in_train.csv (spectrum_id -> train inchikey14/smiles)."""
import csv
from collections import defaultdict

import pyarrow.parquet as pq


def key(prec, mzs):
    return (round(prec, 4), len(mzs), tuple(round(m, 4) for m in mzs[:3]))


test = pq.read_table('data/test.parquet', columns=['molecule_id', 'spectrum_id', 'precursor_mz', 'ms2_mzs']).to_pylist()
want = {key(r['precursor_mz'], r['ms2_mzs']): r for r in test}
hits = defaultdict(set)
pf = pq.ParquetFile('data/train.parquet')
for b in pf.iter_batches(batch_size=65536, columns=['precursor_mz', 'ms2_mzs', 'inchikey14', 'normalized_smiles', 'ingest_lib']):
    for prec, mzs, ik, smi, lib in zip(*(b.column(c).to_pylist() for c in b.schema.names)):
        k = key(prec, mzs)
        if k in want:
            hits[k].add((ik, smi, lib))

mol_hit = defaultdict(set)
with open('logs/test_in_train.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['molecule_id', 'spectrum_id', 'n_matches', 'inchikey14', 'normalized_smiles', 'ingest_lib'])
    for k, r in want.items():
        for ik, smi, lib in sorted(hits.get(k, ())) or [('', '', '')]:
            w.writerow([r['molecule_id'], r['spectrum_id'], len(hits.get(k, ())), ik, smi, lib])
            if ik:
                mol_hit[r['molecule_id']].add(ik)
n_mol = len({r['molecule_id'] for r in test})
print(f'test spectra matched in train: {sum(1 for k in want if k in hits)}/{len(want)} (unique keys; {len(test)} rows)')
print(f'molecules with >=1 matched spectrum: {len(mol_hit)}/{n_mol}; with a single consistent ik14: {sum(len(v) == 1 for v in mol_hit.values())}')
print('libs of matches:', sorted({lib for v in hits.values() for _, _, lib in v}))
