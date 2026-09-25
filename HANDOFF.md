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

_Last updated: 2026-09-24, session S00 (planning, on the Mac)._

- Repo: planning docs only; no code yet. The private GitHub remote exists (`dominictedrow/knees`).
- Research done: `docs/research/` (4 reports) and `docs/kaggle/` (4 reports). The plan is in `docs/plan/`.
- Data: on Dom's external SSD attached to the Windows box. **Not yet audited by any session.**
- Environment on the Windows box: **not set up yet** (Python, CUDA PyTorch for the RTX 5070, pydicom decoders).
- Submissions made: none. Best public LB: n/a. Top of the LB on 2026-09-24: 0.959; public notebooks: 0.941–0.947.
- Long jobs running: none.

## 5. Next session

**S01: Windows environment and data audit.** Brief: `docs/plan/sessions/S01_env_and_data_audit.md`.

## 6. Quick facts a fresh session tends to get wrong

- Train studies = **4,407** (forum + notebooks; the old "7,504" is wrong; confirm from `train.csv` in S01).
  Gold (image-verified) labels = **58 studies**. They're **not representative** (ACL 41% there vs ~20% overall, per the forum).
- The host confirmed that sending report text to a commercial LLM API for labeling is allowed (forum thread 733965;
  see `docs/kaggle/forum_and_official_pages.md`). External knee datasets needing a DUA (OAI, MRNet, fastMRI, KneeCoT) are effectively **not allowed**.
- The Kaggle scoring GPU is a T4-class card (Turing): **fp16 yes, bf16 no**. Training on the RTX 5070 can use bf16.
- Kaggle CLI v2.x reads `~/.config/kaggle/credentials.json` (or `~/.kaggle/kaggle.json`); the competition slug that works is
  `rsna-knee-abnormality-detection`.
- The DOMS_DRIVE on the Mac is FAT32. On Windows, keep data caches on the fastest internal drive, not the FAT32 drive.
