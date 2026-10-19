# CLAUDE.md — project rules for the CASMI 2026 build

## Repo layout (as of 2026-10-18)
- Root `~/casmi` in WSL Ubuntu-24.04 (ext4); the Windows copy `C:\Users\blazi\kaggle\Envenda` is stale. Git branch `main`, remote `origin` = https://github.com/dominictedrow/envenda (private).
- `plan/` — these docs plus `plan/handoffs/`. `data/` — raw competition files (gitignored): `train.parquet` (3.03 GB, 2,539,608 rows, 21 row groups, ZSTD), `test.parquet` (4.8 MB, 1,213 rows), `sample_submission.csv` (400 ids + header).
- Code dirs `casmi/`, `scripts/`, `tests/`, `reports/`, `runs/`, `logs/` at the repo root (P0.3). Env: conda `casmi` in `~/miniforge3` (Python 3.12, torch 2.11+cu128, RDKit 2026.03.3).
- Machine: Windows 11, system Python 3.14, `nvidia-smi` shows only the RTX 5070; WSL2 Ubuntu-24.04 with 12 GB RAM cap (`.wslconfig`).

## Mode and goal
- **Build mode.** No tutor mode, no attempt-first cycles, no explain-back. Dom learns elsewhere; here the job is leaderboard score.
- **The goal is prize money (top 5).** Use any library, pretrained model or public data the rules allow. Optimize for final private-LB MRR@25.
- `plan/COMPETITION.md` is the fact sheet. If you learn something that contradicts it, fix it and note the fix in the handoff.

## 🛑 Permission gates — STOP and ask before doing any of these
When you hit a gate, write ≤10 lines covering what you want to do, why, its cost (GPU-hours, disk, runtime), 1–2 alternatives, and whether it's reversible. Then wait for "yes". If the session must end first, record it in `pending_gates`.

**Rule zero: any deviation from `plan/PLAN.md` is a gate.** `plan/PLAN.md` is the agreed route. If you find a bug, flaw, inefficiency, wrong assumption or better approach *in the plan itself*, don't silently work around it. Stop and ask to reroute, using this format:
- **Plan says:** (PLAN step id + what it specifies)
- **Problem found:** (evidence: numbers, error, benchmark, data fact)
- **Proposed reroute:** (what you'd do instead, which later steps it affects)
- **Cost of rerouting vs. not:** (time, GPU-hours, expected val impact)
- **Alternatives:** (including "keep the plan as written")

This covers skipping, reordering, merging or adding steps; changing a step's method, tool, data or target; and changing exit criteria. Once Dom approves, update `plan/PLAN.md` in the same session, mark the edit `(rerouted D-xxx)`, and log it in `plan/DECISIONS.md`.

The gates below are specific cases that always need permission, even when the plan already calls for them:
1. Any architecture change: adding or removing a model component, changing the head type, changing the token design, or adding a decoder.
2. Changing the validation split or val metric weighting.
3. Adding or removing a data source (training data or candidate pool).
4. Regenerating processed shards or caches (a new `data/processed/vN`), or deleting any cache, checkpoint or dataset.
5. A single training run expected over 4 GPU-hours, or scaling model size by more than 2×.
6. **Every Kaggle submission** (submissions are scarce).
7. Uploading or overwriting a Kaggle dataset.
8. Moving the repo or data, or changing the environment's Python, CUDA or RDKit version.
9. Anything touching git history (rebase, force push, deleting branches or tags).
10. Choosing the 2 final submissions.

Things that need no permission, as long as they stay inside what the current PLAN step specifies: fixing bugs in *your code*, refactors, tests, reports, profiling, hyperparameter sweeps under 4 GPU-hours total, and short (<30 min) smoke runs. Implementation details the plan leaves open (exact function names, file splits, minor constants) are yours to choose; note them in the handoff. If unsure whether something is a deviation, treat it as one and ask.

## Reading rules (protect the context window)
- Never `cat` a file over 150 lines. Use `head`, `tail`, `grep -n` or `view` with a line range.
- Never print a DataFrame over 20 rows or an array over 50 elements. Print shapes, dtypes and `describe()`.
- All job output goes to `logs/*.log`. Inspect with `tail -n 30` or `grep`.
- Inspect parquet with `pyarrow.parquet.ParquetFile(...).metadata/schema`, not by loading it.
- Don't re-read a file you already read this session unless it changed.
- Summaries you'll need later go into the handoff, not into repeated reads.

## Data I/O rules (protect the GPU from starvation)
- **Never load `data/train.parquet` whole** (16 GB RAM). Stream it with `ParquetFile.iter_batches(columns=[...])`, projecting only the columns you need.
- Processed data is written **once** to versioned dirs `data/processed/v{N}/`, with a `manifest.json` (config, row counts, sha256 of each array, git sha).
- Hot-path training data lives in **fixed-dtype numpy memmaps**, sorted by molecule. The loader reads contiguous chunks and does a block shuffle (shuffle chunk order plus within-chunk order), never random row access.
- All data, caches and checkpoints live on the WSL ext4 filesystem, never under `/mnt/c` (resolved in P0.1: repo at `~/casmi`).
- Caches are keyed by a hash of the config that produced them. Matching hash means reuse; never recompute.
- No disk writes inside the training step loop. Checkpoint every N minutes plus at val improvement, keeping only `best.pt` and `last.pt`.
- Metrics go to W&B every 50 steps. Never call `.item()` or `.cpu()` per step.

## Compute-bottleneck rules (GPU must be the bottleneck)
- Before any run over 30 min, do a **200-step profile**. Report samples/s, GPU util (`nvidia-smi dmon`), and the share of step time spent waiting on data. Target data-wait under 5%. If it's higher, fix the loader before training.
- Local RTX 5070: bf16 autocast, TF32 on, `torch.compile`, SDPA attention (flash kernels), fused AdamW, `pin_memory=True`, `non_blocking=True`, `persistent_workers=True`, and length bucketing by peak count so padding stays small.
- Kaggle GPUs (T4/P100): fp16 + GradScaler, no bf16.
- RTX 5060 (second GPU): side jobs only, such as feature caching, candidate-pool fingerprints and embeddings, and small ablations. No DDP across mismatched GPUs.
- Candidate fingerprints are stored as packed bits (uint8). Tanimoto is computed on GPU with popcount, in batches.
- Validation runs on fixed, precomputed candidate pools per val molecule; never rebuild windows during training.

## Train/inference parity
- All feature code lives in `casmi/features/`. Training reads cached outputs; Kaggle inference **computes features on the fly** with the same functions.
- `tests/test_parity.py` compares cached and on-the-fly features for 200 random spectra. It must pass before every submission.

## Metric
- RDKit is pinned to **2026.03.3**, matching the official scorer.
- `casmi/metric.py` mirrors the official matching: tautomer canonicalization, then InChIKey first block, then MRR@25 with the first-correct-hit rule. Unit-tested.
- **Local validation is the truth.** The public LB covers only part of about 400 molecules. Never tune on the LB, and always report local val next to the LB.

## Kaggle submission protocol
1. Run the parity test and the format check: every test `molecule_id` exactly once, ≤25 `;`-joined valid SMILES, best first.
2. Do a local dry run of the notebook on the sample test, with per-stage timings. Projected hidden-test runtime must be ≤6 h, leaving headroom under the 9 h limit.
3. 🛑 Ask Dom: commit sha, local val, expected LB change, submissions left today.
4. Push the notebook with `kaggle kernels push`. Submit via CLI if supported; otherwise give Dom exact click steps.
5. Log the row in `plan/LEADERBOARD.md` immediately (score pending), and fill in the score when it returns.

## Git and tracking
- Commit every session. Tag at each phase end. Risky experiments go on `exp/<name>` branches.
- Never commit data, weights or Kaggle files (rules forbid it). Root `.gitignore` covers `data/ checkpoints/ wandb/ logs/ __pycache__/ *.pyc .env`.
- W&B project `enveda-casmi26`, run name `P<phase>-<short>-<gitsha7>`. Also append one line per run to `runs/runs.csv`. W&B runs offline on Kaggle.
