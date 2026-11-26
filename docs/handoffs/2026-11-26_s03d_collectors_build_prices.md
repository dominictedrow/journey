# Handoff: S03d · 2026-11-26 · Phase 2 build, step 4 (prices, assets, membership, aliases)

**Phase:** 2 (Collectors), step (b) build
**Session goal:** Finish step 4 (Alpaca bars + assets, S&P 500 membership, EDGAR ticker map → aliases), backfill it live, commit.
**Outcome:** 🟡 partial (build step 4 of 5 done)

## What was done
- **Code (written in the prior, unhanded-off part of this session; verified and committed here as `eca541a`):**
  `collect/alpaca.py` (data client), `assets.py` (`alpaca_assets`, `crypto_assets`), `bars.py` (`bars_equity`, `bars_crypto`), `membership.py` (fja05680 history + weekly Wikipedia snapshot, security tracing across renames), `aliases.py` (EDGAR ticker map + manual aliases, ticker eras), `backfill.py` (CLI: `--source`, `--all`, `--report`). Migration 007 (`assets.security_key`, `listed_until`). Tagger expands aliases over ticker eras.
- **Registered:** `stock_daily` += `alpaca_assets`, `bars_equity`; `crypto_daily` = `crypto_assets`, `bars_crypto`; `weekly` = `membership`, `aliases`.
- **Live backfill:** membership 248,039 rows (fja05680 490 snapshots 2015-12-31 → 2026-08-18 + 1 Wikipedia); equity bars 1,733,819 (728 assets, 2016-01-04 → 2026-11-25); ETF 37,755 (14); crypto 40,273 (33, from 2021-01-01); 2,450 aliases, 93 CIK updates. 0 null `released_at`. Rerun inserts 0 bars.
- Renames traced (FB→META = `SPX:FB:2015-12-31`, ANTM→ELV). 731 equity securities, 135 delisted.
- launchd service reinstalled: now `caffeinate -is … qt.orchestrator.scheduler` (paper), running on step 4 code.

## Files touched
| File | Change |
|---|---|
| `src/qt/collect/{alpaca,assets,bars,membership,aliases,backfill}.py` | New |
| `src/qt/db/migrations/007_assets_identity.sql` | New |
| `tests/test_{aliases,bars,membership}.py` | New |
| `src/qt/collect/news/tag.py` | Era-expanded aliases |
| `src/qt/common/config.py`, `configs/collect.yaml`, `configs/aliases.yaml` | universe/alpaca/crypto/membership/edgar config; manual aliases |
| `src/qt/orchestrator/jobs.py`, `tests/test_scheduler.py` | Register step 4 collectors |
| `scripts/install_scheduler.py` | caffeinate wrapper |
| `pyproject.toml`, `uv.lock` | `lxml` |
| `DECISIONS.md` | D-030 |

## Tests
- Command: `uv run pytest -q` and `uv run ruff check .`
- Result: 149 passed; ruff clean.

## Decisions made (also in DECISIONS.md)
- D-030: securities not tickers; raw bars only (supersedes 6.1 `alpaca_adj`); no in-progress bars; era-expanded aliases; caffeinate.

## Experiments
None.

## State of the system
- DB: migrated to 007; prices, membership, aliases loaded. Report: `reports/backfill_20261126.md` (gitignored); cross-check `reports/membership_crosscheck.md` (7 vs 7 diffs, all post-2026-08-18 index changes).
- Scheduler running? Yes, launchd `com.qt.scheduler` (paper) under caffeinate.
- Git: `eca541a`. Nothing pushed (steps 1–4 all unpushed).

## Blockers / surprises
- **3 former members have no Alpaca bars:** SNDK (old SanDisk, out 2016-05), TE (TECO, 2016-06), STI (SunTrust, 2019-12). Small survivorship gap; yfinance (step 5) may fill them.
- **Membership writes a new Wikipedia snapshot every run**, even if unchanged (503 rows per run). Fine weekly; manual reruns add near-duplicate snapshots. Consider skipping identical snapshots in step 5.
- Corporate actions (needed for Phase 3 adjustment, D-030) are added to step 5.

## Next exact task
1. Build step 5: yfinance, Alpaca corporate actions, EDGAR facts, Congress, earnings, crypto structure (CoinGecko/DefiLlama/derivs); register them; full `backfill --all --report`.
2. Then the Phase 2 exit gate (5-day unattended run).

## Questions to ask Dom next session
- Confirm the Fed Board roles in `configs/collect.yaml` (Warsh Chair?).
- OK with D-029 release-time changes (18:00 ET default; HY/VIX next session 12:00 ET)?
- OK to push steps 1–4 to `vibe/quant_trader`?
