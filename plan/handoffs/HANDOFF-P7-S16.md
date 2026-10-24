# HANDOFF P7-S16 — 2026-10-24

## Done this session
- Gate 3 asked early (np_chain still training) → approved, D-036.
- `scripts/pubchem_popularity.sh` (resumable curl of PubChem FTP CID-SID.gz 2.0 GB + CID-PMID.gz 0.3 GB into
  data/external/pubchem/) → `scripts/pubchem_popularity.py` → `data/processed/pool_v1/popularity.npy`
  (f16, same row order as cid.npy, ln1p(n_sid) + ln1p(n_pmid), 0 if cid == 0); sha + stats in manifest['popularity'].
- `scripts/np_probe.py` now also sweeps `blend + lam * popularity` (rerank inside the saved top-K only).

## Results / numbers
- popularity: nonzero 89,973,312 / 90,043,950 rows, mean 1.16. Full build ~5 min incl. download.
- λ sweep on runs/p6_ft/top200_v_np.npz (⚠ LEAKY: p6_ft trained on the V-np molecules, so near ceiling):
  | lam | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
  |---|---|---|---|---|---|---|
  | all cand | 0.886 | 0.939 | 0.946 | 0.918 | 0.875 | 0.789 |
  | curated only | 0.931 | 0.958 | 0.955 | 0.931 | 0.893 | 0.817 |
  Blend scale is cos + 4·Tanimoto (~0–5), popularity ~0–20, so useful λ is small.

## Decisions & approvals
- D-036 (gate 3): PubChem popularity prior. Logged.

## Traps discovered
- CID-PMID rows are (cid, pmid, type): a pmid under 2 types counts twice (ponytail comment in script).
- The prior only reranks the top-200 chosen by blend; a truth outside the top-200 can't be rescued (truth-in-top-200 0.98 on p6_ft).

## Background jobs
- np_chain (logs/np_chain.log), at np_p5_pre step 48k/50k at 13:55; np_p6_pre 30k (~1.15 h) + np_p6_ft 10k (~0.4 h) remain → ETA ~15:45.
  Check `tail -2 logs/np_chain.log`, `nvidia-smi`. If it died: `bash scripts/bg.sh np_chain logs/np_chain.sh`.

## Next session starts with
- When np_p6_ft is done: `PYTHONPATH=. python scripts/eval_encoder.py runs/np_p6_ft/last.pt --top 200 1,4,0,0.05 --splits v_np`
  then `PYTHONPATH=. python scripts/np_probe.py runs/np_p6_ft/top200_v_np.npz` → honest V-np with/without curated filter and the λ sweep.
- If λ helps honestly: add popularity to the Kaggle predict path (casmi.kaggle.predict) → gate 7 (casmi26-pool version
  with popularity.npy) + gate 6 (Submission #6 = #5 + λ·popularity).

## Open questions for Dom
- none
