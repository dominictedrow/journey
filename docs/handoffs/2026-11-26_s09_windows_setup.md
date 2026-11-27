# Handoff: S09 · 2026-11-26 · Windows setup (new home)

**Phase:** 3 (features), exit gate still waiting. This session moved the dev setup to Windows.
**Session goal:** Set up the Windows box as the new home, in the conda `ai` env (Dom: "don't make venv").
**Outcome:** 🟡 partial. Code, env, DB and tests work on Windows. Secrets, data and the scheduler are still on the Mac.

## What was done
- Cloned `vibe/quant_trader` into `C:\Users\blazi\AI_projects\Vibe_models\quant_pipeline`.
- conda `ai` env (Python 3.14.8): installed all deps from `pyproject`. `hmmlearn==0.3.3` has no cp314
  wheel, so it was built from source (MSVC Build Tools 18). Project installed editable with
  `--no-deps --ignore-requires-python` (pin was `<3.14`); after D-048 it is plain `pip install --no-deps -e .`. `pip check` clean. This upgraded pandas
  3.0.6 → 3.1.1 and pyarrow 25 → 26 in the shared `ai` env.
- `.env` created from `.env.example` with a generated local `POSTGRES_PASSWORD`. API keys are blank.
- Docker: `qt_db` and `qt_db_backup` up and healthy (fresh volumes). Migrations 001–010 applied to both;
  re-run is a no-op.
- Fixed two Windows-only bugs (D-047):
  1. `_earnings` int64 overflow (x86 wraps 2**63 to INT64_MIN, ARM saturates). All three earnings
     features were NaN on Windows. Caught by 5 hand-calc tests.
  2. `localhost` → `127.0.0.1` in `configs/db.yaml`: every connection waited 5 s for `::1`. Full
     suite went from >10 min (killed) to 2:54.
- `.env.example`: the placeholder `DATABASE_URL`s overrode `POSTGRES_*` when copied (auth failed).
  They're blank now, with the example in a comment.

## Files touched
| File | Change |
|---|---|
| `src/qt/features/price.py` | `shift(-1, fill_value=NS_INF)` keeps int64 |
| `configs/db.yaml` | host `127.0.0.1` |
| `.env.example` | DB URLs blank by default, example in comment |
| `CLAUDE.md` | Environment: Windows (conda `ai`) and Mac (legacy) |
| `pyproject.toml`, `uv.lock` | `requires-python = ">=3.13,<3.15"`; re-locked, no version changes |
| `DECISIONS.md` | D-047, D-048 |

## Tests
- Command: `python -m pytest -q`, `python -m ruff check .` (conda `ai`)
- Result: 756 passed, 6 failed, ruff clean. The 6 (5 in `test_bars.py`, 1 `test_membership` end-to-end)
  fail only on `missing secret(s): ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY`; with dummy `PK…` keys in the
  process env they pass (7 passed). So 762/762 once real keys are in `.env`.

## Decisions made (also in DECISIONS.md)
- D-047: Windows in conda `ai` (Py 3.14), int64 overflow fix, DB host 127.0.0.1.
- D-048: Python pin widened to `>=3.13,<3.15` (Dom: "Upgrade python").

## Experiments (also in EXPERIMENTS.md)
- None.

## State of the system
- Windows DB: schema at 010, **no data**. The Mac DB has the full history (features 2016 → 2026-11-26).
- Scheduler: still `com.qt.scheduler` on the Mac, on `c67453b`. The earnings bug never hit the Mac
  (ARM saturates), so its feature history is correct and needs no rebuild. No Windows scheduler: the
  installer raises `NotImplementedError` ("Phase 12").
- Git: S09 changes uncommitted on `vibe/quant_trader`.

## Blockers / surprises
- Secrets: copy the real keys from the Mac `.env` into the Windows `.env` (Dom, by hand).
- Data: Dom chose a fresh backfill on Windows (no Mac restore), next session.
- The gate (criterion 7, Phase 2's 5 days) is unattended scheduler runs; until the Windows scheduler
  exists, it has to keep running on the Mac.

## Next exact task
1. Dom: copy API keys into the Windows `.env`; re-run the suite (expect 762 passed).
2. Keep the Mac scheduler running for criterion 7 (Fri 11-27, Mon 11-30, Tue 12-1) and the Phase 2 gate.
3. Separate session (Dom): rebuild the data on Windows by backfill (no Mac restore), and build the Windows
   scheduler (pulled forward from Phase 12).

## Questions to ask Dom next session
- Carried: duplicate `D-041` note; LLM size (Phase 8), whole vs fractional shares (Phase 6), same-session
  rule (Phase 5), starting NAV (Phase 5), crypto size-cap volume source (Phase 6), short-history threshold (Phase 5).
