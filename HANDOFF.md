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
   - `docs/plan/03_performance_plan.md`: where step time goes (compute vs bandwidth vs overhead vs I/O) and the ranked optimization menu.
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

_Last updated: 2026-09-30 (on the Mac + Kaggle CPU kernels; no Windows box, no GPU)._

- **S01 code** (part 1): `src/knee/audit/`. **The S01 data audit ran on Kaggle** (2026-09-30, private kernel
  `dominictedrow/knee-audit` = `notebooks/kaggle_audit/`, 31 min): full corpus, 819,078 train files, all Explicit VR LE,
  0 errors/truncations; `docs/data_audit.md`, `docs/dicom_tags.md` (+ `_test`). Outputs copied to `<audit_out>` on the Mac
  (`study_train.parquet` = site proxy). **Dead zone = 20 mm** (decision 2026-09-30). Still Windows-only: `env_check` and
  the local decode throughput (the SSD/i5 numbers S02 needs).
- **S02 code** (part 1): `src/knee/data/` = the shared preprocessing library (`preprocess.preprocess_study()` for the cache
  build *and* the Kaggle notebook). Details: `docs/cache.md`. Open from the audit: 2,707 series have more than one IOP
  (check whether it's float noise or real multi-stack series before the cache build).
- **S03 code** (part 1): Track B ViT, MAE, `train_sup.py` (both tracks), `infer/predict.py`. Log: `docs/sessions_log/S03_2026-09-25.md`.
- **S04 labels:** labels_v0 (4 public CC0 readers, gold 0.889 [0.849, 0.924]) + **folds_v0 now stratified by site**
  (rebuilt 2026-09-30 from the Kaggle site proxy) in `<cache_root>/labels/` on the Mac. Rebuild anywhere with
  `python -m knee.labels.public_sets download` then `python -m knee.labels.build_labels --version v0 --no-claude` (needs
  `<audit_out>/study_train.parquet` for site stratification). **Our reader = Claude Code agents** (`agent_reader.py`,
  `docs/labels_agent_reader.md`). Prompt **v3** is current: gold 0.864 alone, the blend gain over the public readers is
  +0.002 [−0.005, +0.010], so **the full agent run is on hold** (Dom's go rule "beats the public readers" isn't met).
- **Kaggle side:** `docs/kaggle_env.md` (CPU notebook = 4 vCPU / 32 GB / 20.9 GB output; data at
  `/kaggle/input/competitions/<slug>`; datasets at `/kaggle/input/datasets/<user>/<slug>`; NFS reads scale to ~213 MB/s at
  16 workers). Private datasets: `dominictedrow/knee-src` (code; **re-version it when `src/knee` changes**:
  `kaggle datasets version -p <staged dir> -m ...`) and `dominictedrow/knee-wheels` (decoders; installed offline with
  `--no-deps`, never numpy). Kernels: `knee-probe`, `knee-audit`. GPU quota untouched (30 h/week).
- **S06 code**: `notebooks/infer.py` + `scripts/build_kaggle_bundle.py`; the wheels now exist (above).
- **Performance step 1** (2026-09-30, `docs/plan/03_performance_plan.md`, log `docs/sessions_log/P01_2026-09-30.md`): `precision: auto`
  (bf16 on the 5070, fp16 + GradScaler on the T4), threaded DICOM reads (`KNEE_IO_THREADS`), MAE `windows_per_read`,
  inference folds (`knee.model.fuse`), no per-micro-batch host syncs. Not measured on a GPU yet. **`knee-src` on Kaggle is
  now stale**: re-version it before the next Kaggle kernel.
- Tests: 117 pytest (CPU) pass, 2 skipped (`tests/test_perf.py` added).
- Submissions made: none. Long jobs running: none.

## 5. Next session

**Without the Windows box (Mac + Kaggle):**
- Decide what else Kaggle should do. Options: a Kaggle-side preprocessing cache (≤ 20 GB per kernel output, so a smaller
  resolution or series subset) and Track A training on T4 ×2 (30 h/week). The measured I/O makes a full read ≈ 45 min at 16
  workers. Ask Dom before spending GPU quota.
- Investigate the 2,707 multi-IOP series (a Kaggle kernel over `series_train` + a few headers).
- Full agent-reader run only if Dom overrides the go rule (diversity argument; costs subscription time, not money).

**On the Windows box, in this order:** S01 env check (`env_check` + decode bench on the SSD; skip the header scan) →
S02 part 2 (`scripts/run_s02.ps1 -DeadZoneMm 20`, copy `<audit_out>` from the Mac or re-run the audit) → S03 part 2
(VRAM table, MAE pilot, launch) → S05 on labels_v0/folds_v0 → S06 (bundle + `knee-wheels` + Dom's OK to submit).

## 6. Quick facts a fresh session tends to get wrong

- Train studies = **4,407** (forum + notebooks; the old "7,504" is wrong; confirm from `train.csv` in S01).
  Gold (image-verified) labels = **58 studies**. They're **not representative** (ACL 41% there vs ~20% overall, per the forum).
- The host confirmed that sending report text to a commercial LLM API for labeling is allowed (forum thread 733965;
  see `docs/kaggle/forum_and_official_pages.md`). External knee datasets needing a DUA (OAI, MRNet, fastMRI, KneeCoT) are effectively **not allowed**.
- The Kaggle scoring GPU is a T4-class card (Turing): **fp16 yes, bf16 no**. Training on the RTX 5070 can use bf16.
- Kaggle CLI v2.x reads `~/.config/kaggle/credentials.json` (or `~/.kaggle/kaggle.json`); the competition slug that works is
  `rsna-knee-abnormality-detection`.
- The DOMS_DRIVE on the Mac is FAT32. On Windows, keep data caches on the fastest internal drive, not the FAT32 drive.
