# CLAUDE.md: quant_trader

## Session protocol (do this first, every session)
1. **Read `START_HERE.md`.** It holds the current phase, the next exact task and the open questions.
2. Read the newest file in `docs/handoffs/`. Filenames sort by date.
3. Read the docs for the current phase:
   - `docs/plan/MASTER_PLAN.md`
   - `docs/interrogation/<component>.md` and `docs/components/<component>.md`, if they exist yet
4. Tell Dom where we are and what's next before writing any code.

## Session end (always)
1. Write a handoff to `docs/handoffs/YYYY-MM-DD_sNN_<topic>.md` using `docs/handoffs/_TEMPLATE.md`.
2. Update `START_HERE.md`: put a new checkpoint block on top and move the old one down to "Prior checkpoints".
3. If any experiment ran, add a row to `EXPERIMENTS.md` and fill in `experiments/EXP-###_<slug>/`.
4. Any design change gets an entry in `DECISIONS.md`. It is append-only.

## How phases run
Each phase runs in two steps.
1. **Questioning session.** Work through `docs/interrogation/<component>.md` with Dom one question at a time. Record his answers inline. Then write `docs/components/<component>.md`, the build spec.
2. **Build session(s).** Implement from the component doc, test, and meet the phase's exit gate in `MASTER_PLAN.md`.

`docs/spec/system_spec_v2.md` is Dom's original spec. Never edit it. Where they disagree, the component docs and DECISIONS.md win.

## Hard rules
- **Point-in-time.** A feature for decision time T may only use rows with `released_at <= T`. Training uses ALFRED vintages. Every feature gets a no-lookahead test.
- **Raw tables are append-only.** Nothing overwrites them, including the LLM. Adjusted data goes to its own feature version.
- **No random splits.** Use walk-forward with purge and embargo only. The meta learner trains on out-of-fold predictions only.
- **Timestamps.** Store in UTC; calendar logic uses America/New_York (`exchange_calendars` XNYS).
- **Configuration.** Every number lives in `configs/*.yaml`: thresholds, model sizes, tier weights.
- **Paper trading only.** The Alpaca paper URL is hard-coded, key IDs must start with `PK` and accounts with `PA`. Live trading is unreachable in code until the go-live gate passes and Dom explicitly authorizes it.
- **Default to NO TRADE** when a critical feed is stale or missing.
- **Free APIs only.** Check `docs/plan/API_REGISTRY.md`. Anything paid needs Dom's approval and a DECISIONS entry.
- **Cross-platform (this will move to Windows).**
  - Use `pathlib` only, no hard-coded `/` paths and no bash-only scripts. Scripts are Python.
  - Keep `.gitattributes` line endings.
  - Write device-agnostic torch code (cuda, then mps, then cpu). Use fp32 on MPS.

## Environment
- **Windows (home from S09, D-047):** use the conda `ai` env, no venv. Python is
  `C:\Users\blazi\miniconda3\envs\ai\python.exe` (3.14): `python -m pytest`, `python -m ruff check .`,
  `python -m qt.db.migrate`. The project is installed editable
  (`pip install --no-deps -e .`; pin widened in D-048).
- **Mac (legacy, still runs the gate):** `DOMS_DRIVE` is exFAT, which breaks venvs. Keep the venv off
  the drive: `export UV_PROJECT_ENVIRONMENT=~/.venvs/quant_trader`. Python 3.13 and uv (`uv sync`, `uv run pytest`).
- The database runs in Docker (`docker compose up -d`, TimescaleDB, named volume). Its data never sits on exFAT.
- Secrets live in `.env` (gitignored). `.env.example` lists the key names. Never print or commit key values.

## Git
- Branch `vibe/quant_trader` on `github.com/dominictedrow/journey`. It is an orphan branch with the project at the root.
- Use Conventional Commits and stage files by name, never `git add -A` or `git add .`.
- Scan for secrets before every commit.
- Every push needs Dom's OK.
