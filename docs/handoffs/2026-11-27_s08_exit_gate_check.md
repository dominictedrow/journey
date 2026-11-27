# Handoff: S08 · 2026-11-27 · Phase 3 exit gate check, alert mute

**Phase:** 3 (features), step (b) build: exit gate
**Session goal:** Run the Phase 3 exit gate (`docs/components/03_features.md` §12).
**Outcome:** 🟡 partial. Criteria 1–6 pass. Criterion 7 waits on three future sessions.

## What was done
- Checked the exit gate on the live DB and the repo (Thu 11-26, ~22:30 ET):

| # | Criterion | Result |
|---|---|---|
| 1 | Migrations apply, re-run is a no-op | ✅ 10 applied through `010_survivorship` |
| 2 | Top-up series backfilled | ✅ `reports/backfill_20261126.md` |
| 3 | Full history and a registry row per feature | ✅ `features` 2016-01-01 → 2026-11-26 (3,983 asof values). 350 registry rows. 0 computed (feature, version) pairs since 11-20 lack a registry row |
| 4 | Hand-calc + no-lookahead tests, pytest, ruff | ✅ 760 passed before the change, 762 after. `ruff check` clean |
| 5 | HMM fits + stability report | ✅ `reports/hmm_stability_20261126.md`: 154 fits, 0 above threshold |
| 6 | Survivorship report | ✅ `reports/survivorship_20261126.md`: 120/137 CIKs (88%, target 80%). 17 left are the accepted gap (D-044) |
| 7 | `features_daily` unattended for 3 sessions, summary shows counts and NaN per group | ⏳ 0/3. 11-26 ran `ok` (17 min) but was Thanksgiving, so it covered crypto only (33 assets) and doesn't count |

- Scheduler check: PID 75620 started 18:43:44 PST, 21 s after `39844b9`. It was running the D-045 v2 code.
- AABA/WYND alerts: the 10 critical "possibly delisted" pushes on 11-26 had no `run_id`. They came from the manual `scripts/map_delisted_ciks.py` run, not a scheduled job. Muted them with a new `alerts.mute` config (D-046), commit `c67453b`.
- Restarted the scheduler (`launchctl kickstart -k`), Dom authorized. New PID 89957 started 19:34:35 PST. News and Fed RSS ran `ok` right away, and new runs record `git_sha c67453bb4e71`.

## Files touched
| File | Change |
|---|---|
| `configs/app.yaml` | `alerts.mute`: two `{logger: yfinance, contains: "$AABA/$WYND: possibly delisted"}` rules |
| `src/qt/common/config.py` | `AlertMuteCfg`; `AlertsCfg.mute` (default empty) |
| `src/qt/common/alerts.py` | `is_muted()`; `AlertHandler` skips muted records (they stay in the file log) |
| `src/qt/common/logging.py` | passes the mute rules to `AlertHandler` |
| `tests/test_alerts.py` | muted line is logged, not pushed; the mute is narrow |
| `DECISIONS.md` | D-046 |

## Tests
- Command: `uv run pytest -q`, `uv run ruff check .`
- Result: 762 passed, ruff clean. `ruff format --check` flags 21 files, the same count as before the change (old drift; not gated).

## Decisions made (also in DECISIONS.md)
- D-046: alert mute list for expected library ERROR lines (AABA/WYND yfinance only).

## Experiments (also in EXPERIMENTS.md)
- None.

## State of the system
- DB: migrations through 010; `features` through 2026-11-26.
- Scheduler running? Yes, `com.qt.scheduler` PID 89957 on `c67453b`.
- Git: `vibe/quant_trader`, `c67453b` plus this docs commit. **Not pushed**: the push was blocked by the permission check, so Dom pushes.

## Blockers / surprises
- 11-26 alerts with `delivered=False`: `collect_fred` (3/23 series, DGS3MO HTTP) and `collect_sosovalue`. Both were fine on later runs. Watch for a repeat during the Phase 2 gate.
- The agent can't push or restart services without Dom's explicit OK in the session (auto-mode classifier).

## Next exact task
1. Re-check criterion 7 after each `features_daily` run: Fri 11-27 (early close 13:00 ET; the job still runs 18:00 ET), Mon 11-30, Tue 12-1. Check `runs` (stage `features`, status `ok`) and the Features table in `reports/daily/<date>.md`.
2. Phase 2 gate: Congress `backfill.done`, 5 unattended days from Fri 11-27.
3. When both pass, close Phase 3 and start Phase 4 questioning.

## Questions to ask Dom next session
- DECISIONS has two `D-041` rows (append-only). Add a renumbering note?
- Carried: LLM size (Phase 8), whole vs fractional shares (Phase 6), same-session rule (Phase 5), starting NAV (Phase 5), crypto size-cap volume source (Phase 6), short-history trading threshold (Phase 5).
