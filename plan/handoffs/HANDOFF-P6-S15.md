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
- **CORRECTED later in S15: the gap below was the P4.5 library tier, not memorisation.** COS_T = 0 forces every
  library structure in the window (~50/mol) above all pool candidates, so a kind-2 truth can never rank top-25.
  Without the tier, p6_ft is V-pool 0.70 / weighted 0.743 (see "Later in S15"). Original (wrong) note kept for the record:
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

- P7.2 (S15 cont.): `eval_encoder.py --top 200 c,t,a,b` -> runs/<run>/top200.npz (top-200 by blend + 50 best
  library-cosine rows; tiered score, raw blend, libcos, hit). `scripts/frag_features.py` -> frag200.npz: fragment
  features help a little (V-pool 0.0316 -> 0.0343 with + 1 x expl1, under the tier). `scripts/lib_blend.py` sweeps
  forced-tier threshold T and beta * libcos:
  | model | rule | V-lib | V-pool | V-scaf | V-novel | weighted | lb_mix (18% lib) |
  |---|---|---|---|---|---|---|---|
  | p5_ft | tier T=0 (Kaggle now) | 0.877 | 0.0286 | 0.0264 | 0.0279 | 0.311 | 0.181 (LB 0.184) |
  | p5_ft | T=0.7, beta 0 | 0.841 | 0.595 | 0.565 | 0.580 | 0.672 | 0.633 |
  | p6_ft | T=2 (no tier), beta 0 | 0.829 | 0.700 | 0.663 | 0.701 | 0.743 | 0.724 |
  Leak checks done: stored vs recomputed-from-SMILES fp identical for truths and decoys; with the tier, val predicts
  the LB (0.181 vs 0.184). Any beta > 0 hurts pool badly (wrong library structures often have cosine ~0.4-0.8).
- Trap: `pgrep -f <pattern>` inside a waiting `bash -lc` loop matches the waiter itself; wait on the PID from bg.sh.

- Submissions (S15 end, 2026-10-24): #4a (p5_ft, LIB_T 0.7, kernel v4) public LB **0.184** = unchanged from #3;
  #4b (p6_ft no tier, kernel v5, datasets re-versioned with fkey.npy) LB **0.207** (new best). Local dry run of #4b:
  400 rows, known-answer 0.933. Commits 45f6a16 + a886d1e pushed.
- **Val no longer predicts the LB.** Val wtd 0.672 / 0.743 vs LB 0.184 / 0.207; lb_mix predicted 0.63 / 0.72.
  #4a moved val +0.36 with LB +0.000, so the library tier was NOT what capped the hidden test, and V-pool/V-novel
  (truth always in pool_v1, ~0.97 in window) do not look like the hidden test molecules. Check first: hidden-test
  truth coverage in pool_v1 (truth not in pool => MRR 0 however we rank) and how much of the LB is library-tier hits.
  Any change to how val is built is gate 2.
- Trap: `nohup ... &` inside `wsl.exe bash -lc` dies when the call returns; use scripts/bg.sh (setsid).

- **Gap diagnosis (S15 cont., 2026-10-24, Dom: "diagnose the gap").**
  1. Not a Kaggle-path bug: `scripts/val_as_test.py` runs predict_encoder on 400 V-pool molecules shaped like
     test.parquet with a train-only library: MRR 0.714 (val 0.70). (The local dry run only exercises library hits.)
  2. V-sets contain no natural products (`scripts/truth_src.py`): truth in COCONUT/LOTUS = V-lib 9.6%, V-pool 0%,
     V-scaf 0.3%, V-novel 0%; ~97% are PubChem-only screening compounds. In V-lib, NP truths get no-tier MRR
     0.52 vs 0.87 for PubChem truths. Local test.parquet molecules are also 0% curated (99% PubChem-only).
     The curated bit is fine: 249/250 enveda-np-examples structures are curated.
  3. Public 0.45-0.46 pipelines (dmitriigluzdov "Data Behind Spectra to Structures"; read only, in ~/ext_nb): main
     search over train + COCONUT (~690k) ranked by fp/ranking models + a PubChem popularity prior (ln(1+substances)
     + ln(1+PubMed)); raw PubChem only injected at fixed ranks; they benchmark on enveda-np-examples (250 NPs).
  4. Sizes: 26,155 of 274,203 train structures are curated; the train ∪ curated pool is 724,237 structures.
  Reading: the hidden test is NP-heavy, our val is not, so val rewards ranking screening compounds among PubChem
  decoys. Next: Dom's call on (a) a gate-6 probe: #4b with candidates restricted to train ∪ curated, and
  (b) a gate-2 NP val fold built from curated enveda structures.

- **D-035 results (S15 cont.).** (a) Submission #5 (kernel v6, P6_SRC = 7) LB **0.248** (+0.041 over #4b):
  restricting P6 to train/COCONUT/LOTUS helps, as the gap diagnosis predicted. (b) v_np built (196 mols, 891 enveda
  spectra, commit 5af4cc2); np retrain chain running. First run died at step 12.4k of np_p5_pre with WSL CUDA
  "unknown error"; restarted, resumed from state.pt at 12,001. If it dies again: `bash scripts/bg.sh np_chain
  logs/np_chain.sh` (bg.sh takes a bare log name; finished stages resume at their last step).

## Background jobs
- np_chain (logs/np_chain.sh -> logs/np_chain.log; check `tail -2`, `nvidia-smi`). When np_p6_ft is done:
  `python scripts/eval_encoder.py runs/np_p6_ft/last.pt --top 200 1,4,0,0.05 --splits v_np` then
  `PYTHONPATH=. python scripts/np_probe.py runs/np_p6_ft/top200_v_np.npz` (no-tier `blend`, not tiered `score`).

## Next session starts with
- If the tier-fix gate is not answered yet: re-ask it (see STATE pending_gates).
- P7.2: per val molecule, take the P6 (p6_ft, blend 1,4,0,0.05) top-100 from eval_encoder's scoring, compute
  fragment features (explained frac, n explained, min cuts, loss-table hits) + P2.4 isotope score, and measure
  truth-vs-top-100 separation. Random decoys are easy; the top-100 is the real test. Consider 3 cuts / top-200 (cheap).
- P7.3 needs a **separate** held-out fold (not V-sets); designing it may hit gate 2.

## Open questions for Dom
- none
