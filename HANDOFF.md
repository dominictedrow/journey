# HANDOFF — read this first in every new session

You are a fresh Claude Code session on Dom's RSNA Knee project. This file tells you how to pick up
the work without re-deriving anything. It's the **entry point**; the other docs are listed in
reading order below.

## 1. Read in this order (stop when you have what your session needs)

1. `CLAUDE.md`: competition facts, rules, Dom's strategy, working rules (**never guess; mark UNVERIFIED**).
2. `HANDOFF.md` (this file): protocol + **current state** (section 4) + **next session** (section 5).
3. `docs/plan/00_MASTER_PLAN.md`: the phase/session map, both tracks, the GPU calendar, decision gates.
4. **Your session's brief**: `docs/plan/sessions/S<NN>_*.md`. It is self-contained: goal, inputs, steps, done-criteria, and what to write back here.
5. Only as your brief directs:
   - `docs/plan/01_vit_barebones.md`: the ViT skeleton (shapes, modules, where each of Dom's components sits).
   - `docs/plan/02_vit_detailed_plan.md`: the research-backed detail of each section, plus the gap log.
   - `docs/research/*.md`: cited research reports (ViT fundamentals, 3D/medical ViT, SSL + labels, RSNA winners + efficiency).
   - `docs/kaggle/*.md`: what the public Kaggle notebooks and forum show (labels, top solutions, data/DICOM facts, official pages).
   - `docs/decisions.md`: every decision Dom has made, dated. **Don't re-litigate these.**
   - `docs/experiments.md`: every run: config, data, metric, runtime, verdict.

## 2. Roles (decided by Dom, 2026-09-24)

- **Claude writes the code; Dom reviews.** Every session ends with a *Review guide for Dom* in its
  session log: what was built, where, how to read it, which design choices to check, and 2–4
  "learning notes" explaining the ML concept behind the most important piece (Dom's secondary goal
  is learning to build a ViT).
- **Sessions run on Dom's Windows box** (RTX 5070 12 GB, 16 GB RAM, data on an external NTFS SSD).
  Code lives in the private GitHub repo `dominictedrow/knees`.
- Dom answers questions. **When you hit a real gap (a fact you can't verify, or a decision that is
  Dom's to make), ask him** (AskUserQuestion) instead of guessing, then log the answer in `docs/decisions.md`.

## 3. Session protocol (every session, no exceptions)

**Start**
1. `git pull`. Read the files in section 1 up to your brief.
2. Check `docs/plan/00_MASTER_PLAN.md` → "GPU calendar": is a long job (e.g. MAE pretraining) running
   on the GPU right now? **Never kill or preempt a running job without Dom's OK.** Check with `nvidia-smi`.
3. Re-check the Kaggle Overview/Rules if your session makes a decision that depends on them
   (`kaggle competitions pages list rsna-knee-abnormality-detection --content`).

**During**
- Keep the train and inference preprocessing **the same code** (one module, imported by both).
- Log every training run to `docs/experiments.md` (template inside). Numbers without a log entry don't count.
- Long jobs: run them detached with checkpoint/resume, write logs to disk, and record in the GPU calendar.
- Anything uncertain → `UNVERIFIED` + a TODO, or ask Dom.

**End (write-back, before you stop)**
1. Update **section 4 (Current state)** and **section 5 (Next session)** of this file.
2. Tick the session's checklist in `00_MASTER_PLAN.md`; update the GPU calendar.
3. Write `docs/sessions_log/S<NN>_<date>.md`: what was done, results, **Review guide for Dom**,
   open questions, surprises.
4. Git: `git status` + `git diff` → stage **specific files by name** → scan for credentials → commit
   (message explains *why*) → push only if Dom has authorized pushes for this session. Never commit
   data, DICOMs, caches, weights, `kaggle.json`, `.env`, or API keys (see `.gitignore`).

## 4. Current state (update every session)

_Last updated: 2026-09-26, session S04 part 1 (on the Mac; no data drive, no GPU)._

- **S01 code** (part 1): `src/knee/audit/` (env check, CSV audit, header scan, series/study summary, decode bench, reports),
  `scripts/run_s01.ps1`. **S01 part 2 (running it on Windows) has not happened yet.**
- **S02 code** (part 1): `src/knee/data/` = the shared preprocessing library. `preprocess.preprocess_study()` is the one
  entry point for the cache build *and* the Kaggle notebook (canonical frame: every knee stored as a right knee in a
  fixed display per plane; 130 mm crop → 384 px; per-series 1–99 window → uint8; slots). `build_cache.py` (resumable
  shards, parquet indexes, manifest), `dataset.py` (`StudyWindowsDataset` for Track A, `StudyBagDataset` +
  `collate_bag` for Track B), `inspect_cache.py` (figures, loader bench), `scripts/run_s02.ps1`. Details: `docs/cache.md`.
- **S03 code** (part 1): `src/knee/model/` (Track B ViT, hand-written: conv stem, sincos/slice/series-type embeddings,
  manual+SDPA attention, pre-norm blocks, `SliceViT`, `AttnMILHead`/`FindingQueryHead`/`ClsStudyHead`, `KneeViT`; Track A
  `trackA.py`; GPU `sanity.py`), `src/knee/ssl/` (MAE + resumable `pretrain_mae.py` + linear probe), `src/knee/train/`
  (**`train_sup.py` = the one supervised trainer for both tracks**, optim/LLRD, losses incl. rank bank, checkpoint/STOP
  plumbing), `src/knee/utils/metrics.py`, `src/knee/infer/predict.py` (notebook model side → `submission.csv`),
  configs `mae_vits.yaml` / `trackB_vits_mil.yaml` / `trackA_coatnet.yaml`, `scripts/run_s03.ps1`.
  Session log: `docs/sessions_log/S03_2026-09-25.md`.
- **S04 code** (part 1): `src/knee/labels/` (prompt/schema, Claude batch reader with cost ledger + $50 budget guard,
  public sets + license/leak check, blend, LOO gold calibration, folds, `build_labels.py`). **Built labels_v0 + folds_v0**
  (4 public CC0 readers, gold 0.889 [0.849, 0.924]) in `<cache_root>/labels/` on the Mac scratchpad only: rebuild on
  Windows with `python -m knee.labels.public_sets download` then `python -m knee.labels.build_labels --version v0
  --no-claude` (~2 min, CPU). QA: `docs/labels.md`. Session log: `docs/sessions_log/S04_2026-09-26.md`.
  **Claude reader pilot not run yet** (needs `ANTHROPIC_API_KEY` in `.env`); Dom chose to pilot Opus 5 *and* Sonnet 5.
- **S06 code**: `notebooks/infer.py` (Kaggle script kernel) + `scripts/build_kaggle_bundle.py` (stages the upload folder;
  never uploads), tested in `tests/test_notebook.py`.
- **Label/fold file format is fixed** (`train_sup.py` docstring): labels parquet = `StudyInstanceUID` + 12 soft columns
  (NaN = unknown) + optional `w_<label>`; folds csv = `StudyInstanceUID, fold`. S04 must write exactly that.
- Tests: 103 pytest tests (CPU) pass, 1 skipped (stub); includes train/infer **parity** at the pixel level (S02) and at the
  model-input level (S03), and an end-to-end submission test.
- **Long jobs, once launched (S03 part 2):** `scripts\run_s03.ps1 -Status` (process, last log lines, snapshots, GPU),
  `-Pause` (writes `<run_dir>\STOP` → saves `last.pt`, exits), `-Launch` (starts or resumes from `last.pt`). The same
  STOP file works for `train_sup.py` runs.
- **Measured:** the CSVs (`docs/data_audit.md` §1–4), the test stub (`docs/data_audit_test.md`, `docs/dicom_tags_test.md`),
  stub preprocessing ≈4.2 ms/slice on the M3; the canonical frame was checked by eye on real stub images.
- Stub facts still worth knowing: 100% Explicit VR LE, Toshiba files lack TR/TE/Sex/Laterality/FieldStrength, CSV fluid
  flag = header fat-sat on 14/14; decoders bit-exact on all 4 syntaxes. Dom has accepted the competition rules.
- Decided this session: cache v1 = 384 px, every slice of every series, ≈120 GB (`docs/decisions.md`).
- **Not done yet (needs the Windows box):** everything in S01 part 2; the full cache build + its measurements
  (`docs/cache.md` §4b); the train figures; S03 part 2 (VRAM table, MAE pilot, launch).
- Submissions made: none. Best public LB: n/a. Top of the LB on 2026-09-24: 0.959; public notebooks: 0.941–0.947.
- Long jobs running: none.

## 5. Next session

**Can run now, on either machine (no GPU):** the S04 part 2 Claude reader pilot. Brief `docs/plan/sessions/S04_labels_v1.md`
("Status after part 1") has the exact commands: `.env` with the key → `reader sync --n 2` → pilot batches for Opus 5 and
Sonnet 5 → `collect` → `parse` → `build_labels --version pilot_<model>` for each → show Dom gold AUC + $/report → Dom picks →
full run (the guard refuses > $50 total) → `build_labels --version v1`.

**On the Windows box, in this order:**
1. **S01 part 2**: brief `docs/plan/sessions/S01_env_and_data_audit.md` ("Status after part 1"). Set up the venv
   (`requirements.txt` now includes `anthropic`), fill `configs/paths.yaml` (`cache_root` on an internal drive with
   **≥ ~130 GB free**), run `scripts/run_s01.ps1`, and interpret the output. From its laterality table, pick the geometry
   **dead zone** (mm). It also writes the site proxy (`<audit_out>/study_train.parquet`): **re-run `build_labels` afterwards**
   so the folds stratify by site (decision 2026-09-26).
2. **S02 part 2**: brief `docs/plan/sessions/S02_preprocessing_and_cache.md` ("Status after part 1"). Run
   `scripts/run_s02.ps1 -DeadZoneMm <from step 1>` (start with `-SmokeOnly`), fill `docs/cache.md` §4b, and look at the
   train figures with Dom.
3. **S03 part 2**: brief `docs/plan/sessions/S03_vit_modules_and_mae_launch.md` ("Status after part 1"). Run
   `scripts\run_s03.ps1` (tests → VRAM table → MAE pilot), get Dom's OK on the duration, then `-Launch`.
4. **S04 part 2** if it hasn't run on the Mac (above), while MAE trains.
5. **S05** (Track A): fill `labels.file` / `folds.file` / `gold.file` in `configs/trackA_coatnet.yaml` (v1, or v0 if v1
   isn't ready), then its brief. **S06** then needs only wheels + the S05 checkpoint + Dom's OK to upload/submit.

## 6. Quick facts a fresh session tends to get wrong

- Train studies = **4,407** (forum + notebooks; the old "7,504" is wrong; confirm from `train.csv` in S01).
  Gold (image-verified) labels = **58 studies**. They're **not representative** (ACL 41% there vs ~20% overall, per the forum).
- The host confirmed that sending report text to a commercial LLM API for labeling is allowed (forum thread 733965;
  see `docs/kaggle/forum_and_official_pages.md`). External knee datasets needing a DUA (OAI, MRNet, fastMRI, KneeCoT) are effectively **not allowed**.
- The Kaggle scoring GPU is a T4-class card (Turing): **fp16 yes, bf16 no**. Training on the RTX 5070 can use bf16.
- Kaggle CLI v2.x reads `~/.config/kaggle/credentials.json` (or `~/.kaggle/kaggle.json`); the competition slug that works is
  `rsna-knee-abnormality-detection`.
- The DOMS_DRIVE on the Mac is FAT32. On Windows, keep data caches on the fastest internal drive, not the FAT32 drive.
