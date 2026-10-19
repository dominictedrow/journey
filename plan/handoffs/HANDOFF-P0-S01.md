# HANDOFF P0-S01 — 2026-10-18

## Done this session
- P0.1: Ubuntu-24.04 (WSL2) installed by Dom. Repo cloned to `~/casmi` (ext4) from the Windows copy; `origin` reset to GitHub. `data/` copied (sizes match plan). Windows copy at `C:\Users\blazi\kaggle\Envenda` is now **stale**, kept until P0 exit (deleting it = gate).
- `C:\Users\blazi\.wslconfig`: `memory=12GB`, `swap=16GB` (WSL default was 8 GB).
- P0.2: Miniforge at `~/miniforge3`, env `casmi` (Python 3.12, torch 2.11.0+cu128, rdkit 2026.03.3, pyarrow 26, polars 2.0, numpy 2.5, numba 0.68, sklearn, lightgbm, wandb 0.31, kaggle 2.2.4, selfies, pytest, pandas). `environment.lock.yml` exported.
- P0.3: layout `casmi/{data,features,models,retrieval,rerank,denovo,kaggle}/`, `scripts/ tests/ reports/ runs/ logs/`; `runs/runs.csv` header.
- P0.4: `casmi/metric.py` (`ik14`, `reciprocal_rank`, `score`) mirrors official `metric/casmi-mean-reciprocal-rank`; `tests/test_metric.py` 10 tests green.
- P0.6: W&B login done by Dom; smoke run `P0-smoke` (id p6drzl88) logged `smoke/metric=1.0` to project `enveda-casmi26` (hosted at forge.coreweave.com/wandb/dominictedrow-dev). Log: `logs/wandb_smoke.log`.
- P0.5: daily limit 5 (from Kaggle API) and final picks assumed 2, written to `plan/COMPETITION.md`. Kaggle CLI auth works in WSL (`~/.kaggle/credentials.json` copied from Windows). Slug: `enveda-CASMI26-molecule-id-mass-spectra`.

## Results / numbers
- Metric parity vs official scorer: 300 real train SMILES × 25 shuffled guesses → 0.13142303421602453 both (exact).
- Data copy /mnt/c → ext4: 3.04 GB in 19 s.
- torch sees 1 GPU: RTX 5070, capability (12, 0); bf16 matmul OK.

## Decisions & approvals
- D-017: WSL move approved (Dom installed Ubuntu-24.04).

## Traps discovered
- GitHub push from WSL uses the Windows GCM: `credential.helper=/mnt/c/Program Files/Git/ucrt64/bin/git-credential-manager.exe` (global git config in WSL).
- Commit messages: use `git commit -F <file>`; an apostrophe inside `wsl -e bash -lc '...'` breaks the quoting.
- Official scorer strips stereo BEFORE tautomer canonicalization; solution `smiles` can hold several accepted structures joined by `,`; unparseable guesses = miss. Train `inchikey14` column is NOT tautomer-canonical — local val truth must go through `casmi.metric.ik14`.
- `!`-prefixed commands are non-interactive (no prompts); Dom runs interactive logins in a real terminal via `wsl -e bash -lc "..."`.
- WSL VM shuts down when idle → `/tmp` is wiped between calls. Use `~/ext/` for scratch (official scorer pulled to `~/ext/metric/m.py`).
- Driving WSL from PowerShell mangles nested quotes; use the Bash tool → `wsl -e bash -lc '...'` with heredocs.
- Only the RTX 5070 is visible (no 5060) — the "5060 side jobs" in plan/CLAUDE.md can't happen until it's installed.
- `cwevals` in the env is a legit wandb dependency (CoreWeave), not a stray.
- Glucose test asserts α/β/flat-stereo glucopyranose match; the exact Evaluation-page glucose example wasn't fetched — verify against the page.

## Background jobs
- none

## Next session starts with
- Work in `~/casmi` (WSL). Activate: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate casmi`.
- P0 closed, tag `p0-env`. Start P1.1.

## Open questions for Dom
- Confirm the number of final picks (assumed 2).
