"""P7 gap check: run the exact Kaggle path (casmi.kaggle.predict.predict_encoder) on val query spectra shaped like
test.parquet, with a train-only library, and score against the truth. Tells a Kaggle-path bug apart from a
val/hidden-test mismatch (local test.parquet is a copy of train, so the dry run only exercises library hits).
Usage: python scripts/val_as_test.py <ckpt> [split=v_pool] [n_mols=400]"""
import csv
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.kaggle.predict import TEST_COLS, predict_encoder, validate
from casmi.metric import score
from casmi.retrieval.library import Library

ROOT, POOL = 'data/processed/v1', 'data/processed/pool_v1'
ck = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else 'v_pool'
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
d = f'{ROOT}/{split}'
mi, srow = np.load(f'{d}/mol_idx.npy'), np.load(f'{d}/src_row.npy')
smiles = pq.read_table(f'{d}/mols.parquet', columns=['smiles'])['smiles'].to_pylist()
pick = np.random.default_rng(0).choice(len(smiles), min(n, len(smiles)), replace=False)
take = np.flatnonzero(np.isin(mi, pick))
rows, mids = srow[take], mi[take]

pf = pq.ParquetFile('data/train.parquet')
starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
cols = [c for c in TEST_COLS if c != 'molecule_id']
parts = []
for g in range(pf.num_row_groups):
    m = (rows >= starts[g]) & (rows < starts[g + 1])
    if m.any():
        t = pf.read_row_group(g, columns=cols).take(pa.array(rows[m] - starts[g]))
        parts.append(t.append_column('molecule_id', pa.array([f'm{x}' for x in mids[m]])))
test = f'{ROOT}/cache/valtest_{split}.parquet'
pq.write_table(pa.concat_tables(parts), test)

sp = pq.read_table(f'{ROOT}/splits.parquet', columns=['ik14', 'split']).to_pydict()
novel = [k for k, s in zip(sp['ik14'], sp['split']) if s == 'v_novel']
if split == 'v_np':  # D-044: v_np was copied out of the train shard, which still holds it; keep its spectra out of the library
    novel += pq.read_table(f'{d}/mols.parquet', columns=['ik14'])['ik14'].to_pylist()
lib = Library(ROOT, ('train',), exclude_ik14=novel)  # what scripts/eval_encoder.py ranks against
out = f'logs/valtest_{split}.csv'
tm, _ = predict_encoder(test, lib.path, POOL, ck, f'{ROOT}/vocab.json', out)
truth = {f'm{i}': smiles[i] for i in pick}
sub = {r['molecule_id']: r['smiles'] for r in csv.DictReader(open(out))}
print(split, ck, json.dumps(tm), 'rows', validate(out, test), 'spectra', len(rows))
print(f'Kaggle-path MRR@25 on {len(pick)} {split} molecules: {score(truth, sub):.4f}', flush=True)
