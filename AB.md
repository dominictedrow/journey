# Grand plan: Track A on Kaggle T4 ×2, Track B on the Windows 5070 (2026-10-07)

## Context
No submission exists yet, the Oct 8 gate is tomorrow, and the final deadline is Oct 22. There is one local GPU and two tracks.
Dom wants the two tracks in parallel:
- **Track A** (pretrained CoAtNet) trains unattended on Kaggle GPUs.
- **Track B** (Dom's ViT: MAE pretraining, then fine-tuning) runs on the 5070, where it can get attention.

Blockers found:
- The 113 GiB train cache exists only on `C:/knee_cache/cache_v1_train`.
- A Kaggle notebook can save at most ~20.9 GB of output.
- No Kaggle training kernel exists.
- Track A must be shown bug-free and running at the expected speed before it spends quota.

Research basis (4 agents, 2026-10-07):
- **Kaggle limits:** a private dataset can be up to 200 GB, and so can the total private quota (search snippet, UNVERIFIED). The dataset root allows at most 50 top-level files. The weekly GPU quota is 30 h and resets on Saturday 00:00 UTC (3rd-party). A batch session is capped at 12 h (3rd-party; 9 h is the safe plan). Attaching a kernel's output as another kernel's input is UNVERIFIED.
- **Speed:** a Track A study needs ~6.6 MB of reads per ~0.9 s of GPU time, so both GPUs together need only ~15 MB/s. The dataset mount is very unlikely to be the bottleneck; T4 compute and clocks are. Run one process per GPU, not DDP. Measure `channels_last` and compile; don't assume either helps. Log T4 clocks and power.
- **MAE:** the config is close to standard. Shrink the decoder (depth 4 → 2). 75% masking is safe. Run more epochs if the probe is still rising. At fine-tune, use layer decay 0.65–0.75 and drop path 0.1–0.2, with no mixup.
- **Code:** the cache index stores shard *numbers*, not paths (`dataset.py:213`), so the cache moves anywhere unchanged. The reader needs a single root, which a symlink farm in `/tmp` provides with no code change. `multi_fold --hours` and the STOP/resume path already exist. Track A's `last.pt` is ~878 MB, and a fold needs ~2 GB of run dir.

## Dom's decisions (2026-10-07, log them in docs/decisions.md)
- Cache upload: **6 flat shard datasets + 1 index dataset**.
- Track A on Kaggle: **folds 0+1 first**, then decide on epochs and whether to run folds 2–4.
- MAE budget: **up to ~24 h** on the 5070. The Track B gate slips accordingly, to ~Oct 11–12.
- 5070 order: **Track A checks (Part 2) first, then MAE.**

## Part 1. The cache on Kaggle ("a Kaggle dataset big enough to serve as the cache")
1. **Layout: 6 shard datasets plus 1 index dataset.** No archive, no `tar` mode, no code change.
   - `knee-cache-v1-s0` … `s5`: 46 `shard_XXXXX.u8` files each, flat in the root (under the 50-file limit), ~19 GB per dataset. Skip the per-shard parquets; training doesn't read them.
   - `knee-cache-v1-index`: `cache_manifest.json`, `series_index.parquet`, `study_index.parquet`, `labels_v0.parquet`, `folds_v0.csv`, `configs/*.yaml`. All small.
   - **Why six:** each part uploads in parallel, a failed upload costs one ~19 GB part instead of 113 GB, and it avoids the unknowns around single-file and `tar` handling. Total ~113 GB of the 200 GB quota.
2. **New script `scripts/stage_kaggle_cache.py`:**
   - Hard-links the shards into `C:/knee_cache/kaggle_stage/<part>/`. Hard links mean zero copy and work on NTFS on the same volume.
   - Writes each `dataset-metadata.json`, reusing the pattern in `scripts/build_kaggle_bundle.py`.
   - Prints the `kaggle datasets create -p <dir> -r skip` commands. Dom OKs the upload, then they run.
3. **Validate on Kaggle with a CPU probe kernel** (`notebooks/kaggle_cachecheck/`, uses no GPU quota):
   - Find all 7 mounts (glob pattern from `notebooks/kaggle_audit/audit.py:31-33`).
   - Build the symlink farm `/tmp/cache/{manifest, parquets, shards/*.u8 → mounts}`.
   - Byte-compare the shard sizes against the manifest.
   - Run a Track A `KneeDataset` (slots) through a DataLoader with workers 1/2/3 and report studies/s and MB/s.
   - Report `df -h`, `nproc`, `free -g`.
   - Test whether a kernel's output can be attached as input (tiny dummy output). This decides how the run dir carries across sessions (Part 3.4).

## Part 2. Track A preflight (correctness, on the 5070, ~45 min, before the MAE launch)
Run these after Dom's current process finishes. All of them use existing `train_sup` with `--set`:
1. **Overfit:** 32 studies, ~15 min. Train AUC should approach 1. Failure means a bug in labels, inputs or loss.
2. **Small real run:** 300 train / 150 val studies, 2 epochs. Loss falls, val AUC is clearly above 0.5, the LR curve is right, and `val_n_nonfinite` = 0.
3. **Resume round trip:** STOP mid-epoch, then resume. The step counter, LR and loss continue as they would have. This is the path every Kaggle session boundary uses.
4. **Inference equals training eval:** `notebooks/infer.py` on that checkpoint reproduces the trainer's val predictions. This proves the shared preprocessing.
5. **CPU checks:** fold-0 train and val share no studies or report hashes, and per-fold prevalence is sane.

Log the results as E004 in `docs/experiments.md`.

## Part 3. Track A on Kaggle (speed + the long run)
1. **Re-version `knee-src`**, which is stale (procedure in HANDOFF §4), and verify it by download.
2. **Speed profile:** push `notebooks/kaggle_profile` with `PROFILE_SET=step2` (~40 min of T4). It picks the Track A T4 variant from ckpt off / stages [0,1] × `channels_last` × compile; ckpt off fits in 11.3 GB of the T4's 16 GB. Results go into a `--set` line or a `configs/trackA_kaggle.yaml`.
3. **New training kernel `notebooks/kaggle_train/train.py` + `kernel-metadata.json`:**
   - Sources: the 7 cache datasets, `knee-src`, `knee-wheels`, and the runs source. Internet stays on for the timm/HF backbone download (or ship the weights in a dataset); this is a training kernel, not a submission.
   - Bootstrap from `audit.py` / `gpu_profile.py`, build the symlink farm, and copy the previous runs into `/kaggle/working/runs`.
   - Start an `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active -l 60` log.
   - Run: `python -m knee.train.multi_fold --config <index>/configs/trackA_coatnet.yaml --folds ... --hours 8.5 --set cache_dir=/tmp/cache run_dir=/kaggle/working/runs/trackA labels.file=... folds.file=... gold.file=<comp>/train.csv data.workers=1 <profile picks>`.
   - Delete `epochs/` snapshots beyond top-k before exit. ~2 GB per fold fits the 20.9 GB cap.
4. **Session chaining:** session N+1 mounts session N's output.
   - If Part 1.3 shows kernel-output attachment works, use that.
   - Otherwise, add `scripts/kaggle_runs_sync.ps1`: `kaggle kernels output` → `kaggle datasets version` of `knee-trackA-runs` (~4 GB, fast with Dom's upload).
   - The `check_run_identity` resume check already tolerates a re-mounted folds file (`train_sup.py:123-137`).
5. **Smoke session (~1 h):** folds 0 and 1, `--hours 0.75`. Pass when:
   - GPU util is above ~90% and the clocks are steady.
   - Step time matches the profile.
   - STOP saves `last.pt`, and the next session resumes it.
   - `already finished` logic works.
   - `val_n_retried_fp32` / `val_n_nonfinite` = 0.
6. **Long run:** 2 folds per session, `--hours` ≈ session cap − 30 min.
   - **Budget:** ~14 T4-h per fold → 5 folds ≈ 35 session-h. Three quota weeks remain: now→Oct 9, Oct 10–16 and Oct 17–22, ~90 h in total (minus profile, smoke and inference).
   - **Order:** folds 0+1 → **first submission as soon as fold 0 finishes** (S06 bundle via `scripts/build_kaggle_bundle.py`, Dom OKs) → folds 2+3 → fold 4.
   - Use the fold-0 curve to decide whether folds 2–4 can train with fewer epochs.

## Part 4. Track B on the 5070 (after Part 2)
1. **Fix the scripts:** `scripts/run_s0*.ps1` point at `.venv`; switch `$py` to the `knees` conda env's python.
2. **Bench the MAE loader:** `scripts/perf/mae_loader_bench.py --workers 4 8 12`. If it delivers under ~650 windows/s and the SSD is the limit, build the 256-px cache (`knee.data.derive_cache`; ask Dom first).
3. **Config changes from the research:**
   - `mae.dec_depth: 2`.
   - Probe on, using `labels_v0`/`folds_v0`, every 5 epochs.
   - Mask stays 0.75 and blr 1.5e-4.
   - Optional crop scale [0.5, 1.0].
4. **Pilot:** `run_s03.ps1` (300 steps) gives windows/s and hours per epoch. Set `optim.epochs` so the run fits **≤ ~24 h** (Dom, 2026-10-07). The research favours 60–100 epochs.
5. **Launch MAE detached:** `run_s03.ps1 -Launch`. Record it in the GPU calendar.
   - Watch the reconstructions and the probe AUC trend.
   - Probe flat from epoch 10 to 40 means stop and rethink.
   - Absolute probe values are expected to be low for MAE.
6. **Fine-tune fold 0** from `encoder_final.pt`: `train_sup --config configs/trackB_vits_mil.yaml` with labels/folds/gold/init `--set`. Check the strict key load.
7. **Random-init control, fold 0:** `init.encoder=null run_name=trackB_vits_mil_scratch`. This shows what MAE bought.
8. **Gate (moves to ~Oct 10–11):** compare B fold 0 with A fold 0 on OOF report-label AUC, gold-58 OOF with a bootstrap CI, the per-finding table, and the A+B rank blend.
   - If B passes, run `multi_fold --folds 1 2 3 4` overnight on the 5070.
   - If not, B continues as the learning track and joins the ensemble only on an OOF blend gain.

## Critical files
- **New:**
  - `scripts/stage_kaggle_cache.py`
  - `notebooks/kaggle_cachecheck/{probe.py,kernel-metadata.json}`
  - `notebooks/kaggle_train/{train.py,kernel-metadata.json}`
  - maybe `scripts/kaggle_runs_sync.ps1`
  - maybe `configs/trackA_kaggle.yaml`
- **Edit:**
  - `configs/mae_vits.yaml` (decoder, probe)
  - `scripts/run_s03.ps1` (python path)
  - `docs/experiments.md`
  - `HANDOFF.md` §4–5
  - GPU calendar in `docs/plan/00_MASTER_PLAN.md` §4
  - `docs/decisions.md` (Dom's answers)
- **Reuse:**
  - `src/knee/train/multi_fold.py` (`--hours`, STOP, resume)
  - `train_sup.py` (`check_run_identity`, `last.pt`)
  - `src/knee/data/dataset.py` `CacheReader` (a path-free index)
  - `notebooks/kaggle_audit/audit.py` (mount discovery, offline wheels)
  - `scripts/build_kaggle_bundle.py` (metadata + bundle)

## Verification
- `pytest -q tests` stays green (214 pass), plus `tests/test_gpu_equiv.py` on the 5070.
- Track A: the Part 2 checks pass (E004), the cache-check kernel matches the manifest byte counts, and the profile and smoke sessions meet the pass criteria in 3.5. Fold 0's `metrics.json` holds a val AUC in the expected range (~public Raptor recipe), and `infer.py` reproduces it.
- Track B: the MAE probe AUC rises over the epochs, fine-tune fold 0 logs its result in `experiments.md`, and the random-init control exists.
- First Kaggle submission scores above 0.5 on the public LB.
