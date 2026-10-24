# START HERE — Enveda CASMI 2026 build

> Dom pastes this whole file into a fresh Claude Code session. You (Claude Code) do not summarize it back. You boot and start working.

## Repo facts
- Repo root: `~/envenda` in WSL Ubuntu-24.04 (`\\wsl.localhost\Ubuntu-24.04\home\dom\envenda` from Windows). Remote: `origin` → https://github.com/dominictedrow/envenda (private), branch `main`.
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
current_phase: P7 — Fragmentation-aware reranker -> Submission #5
current_step: P7.2 paused: D-035 (a) probe LB 0.248; (b) v_np retrain running
session_count: 15
last_handoff: plan/handoffs/HANDOFF-P6-S15.md
repo_root: ~/envenda in WSL Ubuntu-24.04 (moved from ~/casmi, D-031). Run commands via `wsl -d Ubuntu-24.04 -e bash -lc "cd ~/envenda; ..."`.
background_jobs: np_chain (logs/np_chain.sh -> logs/np_chain.log), see handoff
best_public_lb: 0.248 (#5 / D-035 probe, P6 on train/COCONUT/LOTUS rows only, 0e75d1c, kernel v6)
best_local_val: p6_ft no library tier: V-lib 0.829 / V-pool 0.700 / V-scaf 0.663 / V-novel 0.701 / weighted 0.743 (LB only 0.207)
submissions_used_today: 3 on 2026-10-24 (limit: 5/day)
pending_gates: none
blockers: V-np numbers need np_p6_ft (~3 h); until then val cannot rank NP-pipeline changes
next_action: when np_chain finishes, eval V-np (handoff); then ask gate 3 on a PubChem popularity prior
<!-- STATE:END -->
