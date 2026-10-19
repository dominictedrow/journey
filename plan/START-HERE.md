# START HERE — Enveda CASMI 2026 build

> Dom pastes this whole file into a fresh Claude Code session. You (Claude Code) do not summarize it back. You boot and start working.

## Repo facts
- Repo root: `C:\Users\blazi\kaggle\Envenda` (native Windows/NTFS; `/mnt/c/Users/blazi/kaggle/Envenda` from WSL, but WSL is **not installed** yet). Remote: `origin` → https://github.com/dominictedrow/envenda (private), branch `main`.
- Plan docs live in `plan/`: `plan/CLAUDE.md`, `plan/PLAN.md`, `plan/COMPETITION.md`, `plan/LEADERBOARD.md`, `plan/DECISIONS.md`, `plan/handoffs/`. All paths below are relative to the repo root.
- Competition data is already downloaded to `data/` (`train.parquet`, `test.parquet`, `sample_submission.csv`) and is gitignored.

## Boot sequence (every session, in this order)
1. Read `plan/CLAUDE.md` in full. Its permission gates and I/O rules are binding.
2. Read the STATE block at the bottom of this file.
3. Read ONLY the handoff named in `last_handoff`. Don't open older handoffs unless it tells you to.
4. Read ONLY the `plan/PLAN.md` section for `current_phase`. Use `grep -n "^## P" plan/PLAN.md` to find the line range, then view just that range.
5. Skim `plan/LEADERBOARD.md` (last 5 rows only: `tail -n 8`) and `plan/DECISIONS.md` (`tail -n 15`).
6. Run `git status --short` and `git log --oneline -5`. If the repo doesn't match the handoff, STOP and report.
7. Post a ≤6-line kickoff: phase, step, this session's goal, and any 🛑 gates you expect to hit.
8. Begin immediately. Ask nothing unless the next action is a 🛑 gate in `plan/CLAUDE.md`.

If `pending_gates` in STATE is not `none`, your first action is to re-ask that gate question. Dom may answer it in the same message he pastes this file with.

## Session budget
- End the session at about 30% context used. Run `/context` if unsure. If you can't tell, stop after about 25 tool calls that returned substantial output, or after finishing one PLAN step, whichever comes first.
- Never start a step you can't finish or cleanly checkpoint within the remaining budget.
- Long training/processing jobs run in the background (`nohup … > logs/<name>.log 2>&1 &`). Record the PID and log path in the handoff, then end the session; don't babysit. The next session checks the log tail.

## End-of-session protocol (mandatory, in order)
1. Write `plan/handoffs/HANDOFF-<phase>-S<NN>.md` from `plan/handoffs/_TEMPLATE.md`, keeping it under 80 lines.
2. If a Kaggle score came back, add a row to `plan/LEADERBOARD.md`.
3. If Dom approved or rejected anything, add a row to `plan/DECISIONS.md`.
4. Rewrite ONLY the STATE block below (in `plan/START-HERE.md`), using `str_replace` between the markers.
5. Run `git status` + `git diff`, stage changed files by name (never `-A`/`.`; nothing under `data/`), then `git commit -m "<phase> S<NN>: <one line>"`. Pushing to `origin` needs Dom's OK. At a phase end, also `git tag <phase>-<shortname>`.
6. Last message to Dom: 3 lines of what happened, the next action, and "Paste START-HERE.md into a new session to continue."

## STATE
<!-- Claude Code rewrites everything between these markers at the end of every session. Nothing else in this file changes. -->
<!-- STATE:BEGIN -->
current_phase: P2 — Structures, splits, physics features, shards
current_step: P2.7
session_count: 8
last_handoff: plan/handoffs/HANDOFF-P2-S08.md
repo_root: ~/casmi in WSL Ubuntu-24.04 (Windows copy in Repo facts deleted, D-021). Run commands via `wsl -e bash -lc`.
background_jobs: none
best_public_lb: none
best_local_val: none (V-lib / V-pool / V-novel / weighted)
submissions_used_today: 0 (limit: 5/day)
pending_gates: none
blockers: none
next_action: P2.7 tests/test_parity.py (200 random shard spectra via src_row vs on-the-fly featurize), then P2 exit + tag p2-data. P2.5/P2.6 done (9.8 GB shards; loader 9.8x model, data-wait 0.6%).
<!-- STATE:END -->
