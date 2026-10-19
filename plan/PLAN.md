# PLAN.md — CASMI 2026 multi-session build

Every phase lists steps, exit criteria and gates. Phases that produce a new model end in a Kaggle submission (🛑 gate). This plan is the agreed route. If a step turns out wrong, buggy or inefficient once you see real data, **stop and ask to reroute** (`plan/CLAUDE.md`, rule zero). Don't force a bad step, and don't silently work around it. Approved reroutes are edited into this file and marked `(rerouted D-xxx)`.

**Strategy in one line:** a transformer encoder over peak tokens with physics features embeds a molecule's spectra. It ranks candidates from an offline structure pool inside a tight mass/formula window, a fragmentation-aware reranker refines that ranking, and a formula-constrained de novo decoder covers molecules no database contains.

**Why this shape:** test molecules come in three kinds (see `plan/COMPETITION.md`). Kind 1 has public spectra, so library matching wins. Kind 2 is in PubChem/COCONUT without spectra, so spectrum→structure retrieval wins. Kind 3 is truly novel, so only generation can score. Retrieval is where most of the points are; generation is the edge public notebooks lack.

---

## P0 — Environment, repo, metric
- **P0.1** Run `pwd`. The repo currently lives at `C:\Users\blazi\kaggle\Envenda` (`/mnt/c/...` from WSL) with data in `data/`, and WSL is not installed. 🛑 Propose installing WSL and moving it to `~/casmi` (ext4). The Windows-drive bridge is many times slower for small reads and would starve the GPU.
- **P0.2** Set up a conda env: Python 3.12, a PyTorch build supporting Blackwell (sm_120, CUDA 12.8+), `rdkit==2026.03.3`, pyarrow, polars, numpy, numba, scikit-learn, lightgbm, wandb, kaggle, selfies, pytest. Export `environment.lock.yml`. Verify both GPUs from torch.
- **P0.3** Create the repo layout at the root: `casmi/{data,features,models,retrieval,rerank,denovo,kaggle}/`, `scripts/`, `tests/`, `reports/`, `runs/`, `logs/`. Already present: `plan/` (docs + `plan/handoffs/`), `.gitignore`, git repo with `origin` → `dominictedrow/envenda` (private).
- **P0.4** Write `casmi/metric.py` and `tests/test_metric.py`. Look for the official metric code on the competition page or Code tab. If it's not fetchable, ask Dom to paste it. Tests include the glucose stereo/tautomer example from the Evaluation page.
- **P0.5** Set up Kaggle CLI auth. Data is already in `data/` (not `data/raw/`): `train.parquet` 3,033,286,496 B, `test.parquet` 4,848,729 B, `sample_submission.csv` 43,619 B. Ask Dom for the daily submission limit and the number of final picks, then write both into `plan/COMPETITION.md`.
- **P0.6** Run `wandb login` and a smoke run that logs one metric.
- **Exit:** env locked, metric tests green, data on ext4, tag `p0-env`. No submission.

## P1 — Data audit
- **P1.1** From parquet metadata (`data/train.parquet`, `data/test.parquet`, `data/sample_submission.csv`): schema, row groups, compression, row counts for train and test, and sample_submission format.
- **P1.2** Streaming stats, with one column group at a time:
  - `ingest_lib`, instrument, adduct, CE, polarity, precursor error ppm
  - peaks per spectrum (p50/p95/p99/max), m/z range
  - spectra per structure, structures per library
  - whether any MS1/isotope-envelope information exists
- **P1.3** Verify every claim in `plan/COMPETITION.md` and correct the wrong ones.
- **P1.4** If no isotope data exists, 🛑 confirm the replacement plan for Dom's isotope idea: formula from precursor mass decomposition, scored by fragment-subformula consistency.
- **P1.5** Write `reports/data_audit.md` (≤150 lines; plots saved as PNG, not displayed). Include proposals for: max peaks per spectrum `P`, precursor-error cutoff, noise-peak floor, and adducts to support.
- **Exit:** audit written, COMPETITION.md verified, tag `p1-audit`. No submission.

## P2 — Structures, splits, physics features, shards
- **P2.1** Build the structure table `data/processed/v1/structures.parquet`, one row per unique SMILES: tautomer-canonical InChIKey14, formula, exact mass, Murcko scaffold. Use multiprocessing and cache it.
- **P2.2** Spectrum cleaning per the audit:
  - drop rows over the ppm cutoff
  - drop peaks above precursor + 0.5 Da and below the noise floor
  - keep the top-`P` peaks by intensity
  - sqrt intensity, normalized to base peak
- **P2.3** 🛑 Validation design (proposal). All grouping is by InChIKey14, so no structure leaks. Query spectra in every val set are **timsTOF (enveda-\*) only**, matching test. The three val sets mirror the three test kinds:
  - **V-lib:** the structure keeps spectra from other libraries in the reference set (kind 1).
  - **V-pool:** the structure is removed from all training spectra but stays in the candidate pool (kind 2). This is the main metric.
  - **V-novel:** the structure is removed from training and from the pool, so it is scorable only by generation (kind 3).
  - Also a **scaffold-held-out** slice of V-pool, as a harder generalization check.
  - Reported score: each set separately, plus a weighted mean (default equal weights; revisit if the class mix becomes known).
- **P2.4** Physics feature cache in `casmi/features/`. Each piece is fast and vectorized (numba), with a target of ≤10 ms per spectrum.
  - Neutral mass from precursor m/z and adduct.
  - Formula candidates: decomposition within ppm over CHNOPS+F/Cl/Br/I, filtered by Seven-Golden-Rules/SENIOR/RDBE, ranked.
  - Per-peak features: neutral loss to precursor, mass defect, Kendrick mass defect (CH₂), best subformula of the top precursor formula with its ppm error, and a flag for matches to a common neutral-loss table (H₂O, CO, CO₂, NH₃, hexose 162.0528, deoxyhexose 146.0579, pentose 132.0423, glucuronide 176.0321, etc.).
  - Pairwise peak differences for the top-K peaks, matched against the loss table.
- **P2.5** Shard writer `data/processed/v1/{split}/`: memmaps sorted by molecule.
  - Per spectrum: `mz float32[N,P]`, `inten float16[N,P]`, `peakfeat float16[N,P,F]`, `npeaks uint16[N]`, `meta` (adduct, CE, instrument, polarity ids; precursor m/z float64), `mol_idx int32[N]`.
  - Per molecule: spectrum offsets, packed Morgan fingerprint target `uint8[M,256]`, formula vector `int16[M,E]`.
  - Estimate disk size first. 🛑 if over 40 GB.
- **P2.6** Loader and benchmark: block-shuffled molecule batches with length bucketing. Benchmark samples/s against a dummy GPU consumer; the loader must outrun the model by at least 3×.
- **P2.7** Write `tests/test_parity.py` (cached vs on-the-fly features).
- **Exit:** v1 shards and manifest, val sets frozen, loader benchmark in the handoff, tag `p2-data`. No submission.

## P3 — Baseline + Kaggle pipeline → Submission #1
- **P3.1** Library baseline: neutral mass ±5 ppm against train structures, ranked by the mean over query spectra of the best cosine (binned, sqrt) to any reference spectrum of the candidate.
- **P3.2** Kaggle packaging in `casmi/kaggle/`:
  - a script that builds dataset bundles (code wheel, offline pip wheels incl. pinned RDKit, artifacts)
  - a notebook template: install offline, then features on the fly, then rank, then `submission.csv`
  - per-stage timing logs and a format validator
- **P3.3** Local val on V-lib / V-pool / V-novel. Expect V-pool and V-novel near 0; that's the point.
- **P3.4** 🛑 Submit. Log the result in `plan/LEADERBOARD.md`.
- **Exit:** the end-to-end pipeline is proven on Kaggle, tag `p3-baseline`.

## P4 — Offline candidate pool → Submission #2
- **P4.1** 🛑 Choose sources (proposal):
  - train structures
  - COCONUT (natural products)
  - a PubChem subset: 150–1,200 Da, allowed elements, neutral single component, no isotopes
  - optional LOTUS/HMDB
  - Report estimated size first.
- **P4.2** Standardize each structure, then tautomer-canonical InChIKey14 dedup. Store formula, exact mass and packed Morgan fingerprint, all **sorted by mass** with a binary-search index. Keep source flags and any popularity prior available offline (e.g., number of sources containing it, an NP flag). Build this on the 5060 or CPU while P5 prep happens.
- **P4.3** Recall report: share of V-pool molecules whose answer sits inside a ±5/10 ppm window, plus the candidate-count distribution per window. If formula filtering is applied, report recall after the filter too.
- **P4.4** 🛑 Upload the pool as a Kaggle dataset.
- **P4.5** Baseline + pool: library score when available, otherwise prior plus formula rank. 🛑 Submit.
- **Exit:** tag `p4-pool`.

## P5 — Transformer encoder v1 (Dom's architecture) → Submission #3
**Model:**
- **Tokens:** one per peak (≤P), plus one precursor token per spectrum. There is no positional encoding: peaks are a set, and m/z is the position. Context is about 64–256 tokens. Attention uses 8 heads (head dim 64), not thousands; heads split the model width, while context length is the token count.
- **Peak embedding:** an MLP over the concatenation of:
  - multi-scale sinusoidal encoding of m/z, with wavelengths from about 0.001 to 1,000 Da, which keeps exact-mass precision that z-scoring would destroy
  - sinusoidal encoding of the neutral loss
  - z-scored sqrt intensity
  - z-scored cached physics features
  - a subformula-annotation embedding
- **Precursor token:** sinusoidal encoding of precursor mass; element-count vectors of the top formula candidates with their scores; embeddings for adduct, CE, instrument and polarity.
- **Encoder:** standard pre-LN transformer encoder blocks. Start at d=512, 8 layers, about 30–60M parameters.
- **Molecule pooling:** attention pooling over all spectra of a molecule, giving one molecule embedding.
- **Head:** 2048-bit Morgan fingerprint prediction (focal BCE), with a formula-consistency auxiliary loss.

**Ranking:** candidates in the ppm window. Score = Tanimoto between the predicted fingerprint and the candidate fingerprint, plus λ·prior. Tune λ on V-pool.

**Steps:**
- **P5.1** Implement it and run a 200-step profile gate (see `plan/CLAUDE.md`).
- **P5.2** Pretrain on all libraries with the instrument embedding.
- **P5.3** Fine-tune on enveda-\* (timsTOF) spectra. Each run over 4 h is 🛑.
- **P5.4** Val, then 🛑 submit.
- **Exit:** tag `p5-encoder`.

## P6 — Contrastive retrieval + hard negatives → Submission #4
- **P6.1** Molecule encoder: an MLP over the fingerprint plus formula, so candidate embeddings can be precomputed offline for the whole pool. A GNN encoder is a later 🛑 option.
- **P6.2** InfoNCE loss between spectrum and molecule embeddings. Hard negatives: 32 precomputed same-formula / ±5 ppm decoys per training molecule, built on the 5060.
- **P6.3** Final score is a learned or tuned blend of the contrastive cosine, fingerprint Tanimoto and prior.
- **P6.4** Clustering analysis (Dom's K-means idea): K-means on molecule embeddings, then MRR per cluster. Find weak chemical regions and use the result for hard-negative mining and cluster-balanced sampling. Report in `reports/`.
- **P6.5** Val, then 🛑 submit.
- **Exit:** tag `p6-contrastive`.

## P7 — Fragmentation-aware reranker → Submission #5
- **P7.1** 🛑 Choose the fragmenter. Options: MetFrag (Java, offline jar), or an in-house RDKit bond-breaking fragmenter (1–2 bond cleavages with H-shift tolerance; fragment masses matched to peaks). Pick based on a runtime benchmark: top-50 candidates × 400 molecules must fit about 3 h on Kaggle CPU.
- **P7.2** Fragmentation-tree features: fraction of intensity explained, number of explained peaks, which bonds broke and their weakest-bond plausibility (Dom's weak-bond idea), and loss-table hits.
- **P7.3** LightGBM LambdaRank reranker over model scores, fragment features, formula rank, mass error and prior. Train it on a **separate** held-out fold, never on V-sets.
- **P7.4** Val, then 🛑 submit.
- **Exit:** tag `p7-rerank`.

## P8 — De novo generation for novel molecules → Submission #6 (🛑 architecture gate: adds a decoder)
- **P8.1** Fingerprint→SELFIES decoder pretrained on millions of pool structures. No spectra are needed, so data is abundant and generalization is strong.
- **P8.2** Condition it on the encoder memory and the predicted fingerprint. Use formula-constrained beam search that masks tokens exceeding remaining atom counts; SELFIES keeps every output valid.
- **P8.3** Canonicalize and dedupe, score with the P6/P7 scorer, and insert generated candidates only where retrieval confidence is low. Calibrate the threshold on V-pool vs V-novel.
- **P8.4** Optional 🛑: RL fine-tuning of the decoder, with reward = reranker score + exact formula match.
- **P8.5** Val, then 🛑 submit.
- **Exit:** tag `p8-denovo`.

## P9 — Scale, ensemble, finalize (submissions as needed)
- Scale whichever component val says is the bottleneck (🛑 per >2× size). Train seeds on Kaggle GPUs in parallel with local runs.
- Ensemble with reciprocal-rank fusion across seeds and models. Use per-adduct calibration and a runtime audit.
- Loop: find the biggest-loss val slice, fix it, then submit.
- Deadlines: **Dec 7** entry/team-merge deadline (rules must already be accepted); **Dec 14, 23:59 UTC** final. 🛑 Final picks: choose by local val first and LB second.
