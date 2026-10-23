# HANDOFF P6-S15 — 2026-10-22

## Done this session (catch-up: S14 and the P6 sessions after it left no handoff and never updated STATE)
- Reconstructed from git log, logs/, runs/, reports/. S14 (commit 19ce08f): Submission #3 public LB 0.184, tag
  `p5-encoder`, D-028 + D-029 logged. P6 code was left uncommitted "until validated" and is committed now.
- Repo now lives at `~/envenda` (`~/casmi` no longer exists; Dom moved it, gate 8 not logged at the time). Fixed
  `cd ~/casmi` in scripts/{bg,train_retry,run_pool,download_pool}.sh and the path lines in plan/CLAUDE.md and START-HERE.
- P6 code (D-029): `casmi/models/contrastive.py` (Contrastive = P5 Encoder + 512->256 proj; MolEncoder MLP over
  2048 fp + log1p element counts -> 1024 -> 256, L2; InfoNCE learnable temperature; focal fp + formula aux),
  `Encoder.molecules()` split out of forward, loader `hardneg=` (neg_fp/neg_formula/n_neg/pos_formula),
  `formula.fkey_counts` (inverse of formula_key), `scripts/hard_negatives.py` (P6.2, 32 same-formula/±5 ppm decoys),
  `scripts/model_negatives.py` (D-030, top-64 wrong pool_v1 candidates by p6_ft blend), `scripts/cluster_mrr.py`
  (P6.4 K-means), eval_encoder.py sweeps (c, t, a, b), pool.gather reads only needed row groups + timings.
- Tests: tests/test_contrastive.py, tests/test_gather.py (+ test_batch.py) pass, 4/4.

## Results / numbers (honest val MRR@25, best blend per run)
| ckpt | V-lib | V-pool | V-scaf | V-novel | weighted | blend c,t,a,b |
|---|---|---|---|---|---|---|
| p5_ft | 0.877 | 0.0286 | 0.0264 | 0.0279 | 0.311 | 0,1,0,0 |
| p6_ft (init p6_pre 30k all libs, 10k enveda) | 0.877 | 0.0316 | 0.0287 | 0.0301 | 0.3129 | 1,4,0,0.05 |
| p6_mm (D-030 model-mined negs, 10k) | 0.877 | 0.032 | 0.0287 | 0.0294 | 0.3128 | 1,4,0.1,0 |
- Cosine alone (1,0,0,0) 0.3123 ≈ Tanimoto alone (0,1,0,0) 0.3117: the contrastive head adds about +0.001 on its own.
- Kind-2 (4,000 mols, p6_mm): truth in window 0.97, median rank 57 of 6,593, R@1/10/25/100/1000 =
  0.005/0.087/0.244/0.668/0.96. Weakest clusters (MRR 0.02): N-heteroaryl amides, sulfonamides. reports/p6_clusters_*.md
- Timings: p6_pre 1.15 h, p6_ft 0.4 h, p6_mm 0.41 h; hard_negatives 9.8 min; model_negatives 39.5 min (fp 22 min).
  Profile 1,817 spec/s, data-wait 0.5%, 6.8 GB.

## Decisions & approvals
- D-029 (gate 1, P6 architecture) and D-030 (rule zero, model-mined negatives) are both in DECISIONS.md.

## Traps discovered
- **Generalisation gap is the whole problem.** With p6_ft scoring its own train molecules, truth is top-1 in 84%
  (top-25 99.4%), but val kind-2 top-1 is 0.5%. Training NCE goes to ~0.005–0.08, so harder negatives can't help
  while the model memorises train molecules. Next levers: regularisation/augmentation, or a reranker that uses spectrum physics.
- Old logs print `/home/dom/casmi/...` paths; ignore them.

## Later in S15
- D-032: P6.5 Submission #4 skipped, tag `p6-contrastive`; main + all tags p2..p6 pushed (remote had only been at P1).
- D-033 (P7.1 gate): in-house fragmenter `casmi/rerank/fragment.py` (`fragments(smiles)` -> unique 1-2-bond-cut
  piece masses; `explain(mz, inten, masses, charge)` -> (explained intensity frac, n peaks), ±2 H, [frag±H]±).
  `scripts/bench_fragment.py` on 60 V-pool mols x 50 random window decoys: 2.5 ms/cand mean (p99 7.4, max 14),
  133 frags/cand, so top-50 x 400 takes ~15 s on 4 cores. Truth explains more than a decoy 92.6%, mean gap +0.357.
- Traps: vocab key is `ionization_mode` (negative 1, positive 2), not `polarity`. Val spectra are stored as
  mz/inten [n_spec, 128] padded + npeaks. Fragment ions assume [frag+H]+ / [frag-H]- regardless of the precursor adduct.

## Background jobs
- none

## Next session starts with
- P7.2: per val molecule, take the P6 (p6_ft, blend 1,4,0,0.05) top-100 from eval_encoder's scoring, compute
  fragment features (explained frac, n explained, min cuts, loss-table hits) + P2.4 isotope score, and measure
  truth-vs-top-100 separation. Random decoys are easy; the top-100 is the real test. Consider 3 cuts / top-200 (cheap).
- P7.3 needs a **separate** held-out fold (not V-sets); designing it may hit gate 2.

## Open questions for Dom
- none
