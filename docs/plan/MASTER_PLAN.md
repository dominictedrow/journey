# Master Plan

Source spec: [`docs/spec/system_spec_v2.md`](../spec/system_spec_v2.md). Changes since the spec are in [`DECISIONS.md`](../../DECISIONS.md).

## How every phase runs
1. **(a) Questioning session.** Claude works through `docs/interrogation/<NN>_<component>.md` with Dom, one question at a time, and records his answers inline. The result is `docs/components/<NN>_<component>.md`, a precise build spec with interfaces, configs, tests and acceptance criteria.
2. **(b) Build session(s).** Implement from the component doc, add unit and no-lookahead tests, and log experiments. The phase is done only when its **exit gate** passes.
3. Every session ends with a handoff doc and a START_HERE update (see `CLAUDE.md`).

A phase may take several sessions. A session should never end halfway through a change without a handoff that says exactly where it stopped.

## Phases
| # | Component | Main deliverables | Exit gate | Status |
|---|---|---|---|---|
| 0 | Framework | Repo, folder tree, docs, branch | Branch pushed, START_HERE auto-read works | ✅ done 2026-11-25 |
| 1 | Infrastructure | `docker-compose.yml` (TimescaleDB), `schema.sql` plus migrations for all spec tables, config loader (YAML + .env), `run_id`, logging, XNYS calendar helpers, uv project | `docker compose up` works, the schema migrates on an empty DB, and `uv run pytest` is green (full gate: `docs/components/01_infrastructure.md` §14) | ✅ done 2026-11-25 (phone pushes confirmed by Dom) |
| 2 | Collectors (point-in-time) | Alpaca bars and assets, FRED + ALFRED vintages, yfinance (fallbacks), EDGAR companyfacts, Fed calendar + RSS, Congress.gov, crypto structure (CoinGecko, DefiLlama, OKX/Deribit/Kraken/Hyperliquid, SoSoValue), **news fetch + trafilatura + daily archive**, historical S&P 500 membership, APScheduler daily job | 10+ years backfilled, every row has `released_at`, and the daily job runs unattended for 5 days | 🟡 questioning done 2026-11-25 (spec: `docs/components/02_collectors.md`); build next |
| 3 | Feature layer + regimes | All per-asset, cross-sectional and macro features, normalization, GLI, feature registry, quadrant, HMM (filtered only), stress score | Every feature unit-tested against a hand calculation; no-lookahead tests pass | |
| 4 | Labels + validation harness | Triple-barrier labels, walk-forward with purge and embargo, metrics (log loss, Brier, rank IC, Sharpe/Sortino/MDD/turnover), cost model | The harness reproduces a known toy result, and the leak test fails when leakage is injected on purpose | |
| 5 | Rules engine + baseline | Deterministic engine (modes, tiers, sizing, vol target, constraints, exits, bands, breakers) driven by the momentum-rank baseline; backtester | **EXP-001 baseline** IC and Sharpe recorded. Every rule has a unit test. | |
| 6 | Executor (Alpaca paper) | Pre-open check, q70 limit rule, urgency by reason code, 15:30 sweep, broker-side stops, fill and slippage accounting, reconciliation | The full daily paper cycle runs end to end for 2 weeks | |
| 7 | Numeric models + meta learner | MLP, CNN #1, out-of-fold stacking, isotonic calibration, meta learner, ablation | Each model is kept only if its out-of-sample IC beats the baseline (EXP log) | |
| 8 | News pipeline + homemade LLM | Exact + SimHash dedup, quota, SVM + weekly n-gram/chi² vocab, LLM pretrain → task heads, Δ adjustment, injection defenses, KV cache, the KV-cache experiment | SVM and LLM validation metrics logged; Δ clip and risk-event tests pass | |
| 9 | Text models | Transformer (feature + keyword tokens), CNN #2 | 6+ months of news exist, and the ablation shows a gain | |
| 10 | Price transformer | Timing head (historical intraday bars), limit-quantile head after ~300 fills | It beats the q70 rule on measured slippage | |
| 11 | Go-live gate | 3+ months of paper meeting the Validation criteria | Criteria met and Dom signs off | |
| 12 | **Windows migration** | Write [`docs/setup/WINDOWS_SETUP.md`](../setup/WINDOWS_SETUP.md) so Claude Code on Windows can set everything up precisely and fast; add `scripts/verify_setup.py` | A fresh Windows clone, set up by following the doc alone, passes tests, sees the GPU (`sm_120`) and completes a dry-run cycle | |

## Phase 12 plan: Windows migration
The `WINDOWS_SETUP.md` instructions must be written for Claude Code running on the Windows PC (RTX 5070, 12 GB). They should cover:

1. **Install, with pinned versions:**
   - Git
   - uv
   - Python 3.13 (via uv)
   - Docker Desktop with the WSL2 backend
   - the NVIDIA driver (new enough for CUDA 13)
2. **Clone the branch:** `git clone -b vibe/quant_trader https://github.com/dominictedrow/journey quant_trader`.
3. **Set up the environment:**
   - set `UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\quant_trader`
   - run `uv sync`
   - install torch from the `cu130` index (configured in `pyproject.toml` with platform markers)
   - verify that `torch.cuda.get_arch_list()` contains `sm_120`
4. **Set up secrets:** copy `.env` across by hand. It is never in git.
5. **Move the database:**
   - run `docker compose up -d`
   - `pg_restore` the dump made on the Mac with `scripts/db_backup.py` (D-017)
6. **Move data and models:** copy the `data/` and `models/` folders (gitignored) by external drive or `rsync`.
7. **Set up the scheduler:** register a Task Scheduler entry at logon for the APScheduler service, timezone America/New_York.
8. **Verify:** run `uv run python scripts/verify_setup.py`, which checks the DB connection, API keys, GPU, feed freshness and a dry-run cycle.
9. **Known gotchas:**
   - long paths (`git config core.longpaths true`)
   - CRLF line endings (`.gitattributes`)
   - `triton-windows` is only needed if `torch.compile` is used
   - Windows Defender exclusions for data folders

Rules that apply from Phase 1 onward to keep this phase short:
- pathlib only
- Python scripts, not shell scripts
- no macOS-only dependencies
- `.gitattributes`
- device-agnostic torch
- Docker for the database
