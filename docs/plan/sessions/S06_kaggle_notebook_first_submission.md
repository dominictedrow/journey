# S06 — Kaggle inference notebook v1 + first submission

**Track:** A (the notebook is reused by B) · **Machine:** Windows + Kaggle · **GPU:** Kaggle
**Depends on:** S05 checkpoint · **Unblocks:** the G-Oct 1 gate

## Goal
A valid scored submission, the measured runtime, and answers to the test-data open questions.

## Read first
`02_vit_detailed_plan.md` §F · `docs/kaggle/top_solutions_from_notebooks.md` §Inference (pitfalls: T4 fp16, no DataParallel,
sequential arms, never walk the competition dir) · `docs/research/rsna_winners_and_efficiency.md` §B.

## Steps
1. **Offline wheels dataset:** build wheels for exactly what the notebook needs (pylibjpeg, openjpeg, libjpeg, gdcm; timm if not in
   the Kaggle image) matching the Kaggle image's Python; upload as a private dataset with `kaggle datasets create`.
   **Ask Dom before creating any Kaggle dataset.**
2. **Weights dataset:** a private dataset with the S05 checkpoint(s) (ask Dom).
3. **Notebook** `notebooks/infer.py` (script kernel): imports the **same** `src/knee/data` code (package it as a wheel or
   ship the source in the weights dataset); prints the hardware (`nvidia-smi`, `nproc`, `free -g`) and a test-data probe
   (transfer syntaxes seen; is Fluid ≡ FatSup on `test_series.csv`?); runs per-study with a try/except fallback; fp16 with an
   fp32 retry; writes `submission.csv` with column-order and row-identity asserts; logs per-stage timings.
4. Push with `kaggle kernels push` (ask Dom), then submit to the competition (ask Dom: this uses 1 of 5 daily submissions).
5. Record the LB score, runtime, and probe results in `docs/experiments.md`; answer open items 1 and 5 in 02 §H.

## Done when
- [ ] A scored submission exists; the runtime is known; the stage timing shows where the seconds go.

## Review guide for Dom
How a Kaggle code-competition notebook works (hidden test swap, offline, runtime); reading the timing log.
