"""S04: report-derived labels (our Claude reader + vetted public label sets), blend, gold calibration, folds.

Pipeline (each step is a module with its own CLI):
    public_sets.py  download + license/leak-check the public label sets → one standard frame per reader
    reader.py       our Claude API reader (Message Batches, resumable raw-JSON cache, cost ledger)
    blend.py        readers → soft label p + weight w per (study, finding)
    calibrate.py    tiny per-finding bin → P(gold) map for our reader, fitted out-of-fold on the 58
    folds.py        5-fold StratifiedGroupKFold, groups = normalized report hash (+ site proxy when S01 part 2 exists)
    build_labels.py orchestrates → labels_v1.parquet, folds_v1.csv, docs/labels.md
"""
