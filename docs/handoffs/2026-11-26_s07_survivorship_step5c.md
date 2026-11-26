# Handoff: S07 · 2026-11-26 · Phase 3 build step 5c (survivorship) + features rebuild

**Phase:** 3 (Feature layer + regime models), step (b) build
**Session goal:** Step 5c, §9 survivorship: CIKs, sectors and bars for former S&P members.
**Outcome:** ✅ 5c done and committed (`5008828`). The full `features` history is being rebuilt (see State).

## Dom's answers this session
- CIK source: **spec as written** (name match against cik-lookup-data; Wikipedia's CIK never accepted).
- Sectors: **historical Wikipedia GICS, SIC map as the fallback** (changes Q3).
- Nov 2–25 rows: **delete and rebuild**. Then, for the running backfill: **stop now, wipe all of `features`, rebuild after 5c**.
- Push d8c37b4 / 70b05bf: yes. They were already on origin (`153e9ad` = origin), so nothing needed pushing.

## What was done
- **`normalize.asset_ts_z` fix:** ready times went through float64 in `rolling().max()`, which rounds epoch ns and turns the never-ready sentinel into int64-min (the RuntimeWarning in the backfill log). Now an exact int64 `rolling_max_int`. Output is unchanged while ready times are monotone (they are), so no version bump. Regression test fails on the old code.
- **`collect/survivorship.py` + `scripts/map_delisted_ciks.py` (D-043):**
  - Names and sectors come from old revisions of the Wikipedia constituents table (2,169 revisions since 2015-12, cached under `data/cache/wikipedia/`). The current page lost its "Selected changes" table, so it's pinned at rev 1355685534 as a second name source.
  - CIK: normalized name vs `cik-lookup-data.txt` (40 MB, cached under `data/cache/sec/`), exact or fuzzy ≥ 92. A candidate must also have periodic filings across the membership span, exactly one may pass, and Wikipedia's CIK vetoes (never accepts) a match when it names another covering filer.
  - Bars: yfinance then Stooq, exact ticker plus the date rule.
  - Outputs: `configs/cik_overrides.yaml` (120), `configs/sector_overrides.yaml` (226), `reports/survivorship_20261126.md`, `reports/cik_review_20261126.md`.
- Bugs caught on live data, before commit:
  - A fuzzy match took SVB Financial → CVB Financial, which passed the filing check. Fixed by the Wikipedia veto plus `group` as a stripped suffix; SIVB is now exact (719739).
  - BBT's `assets.name` is "Beacon Financial", the symbol's current holder. A stored name is now the last-resort name only; BBT is now exact (92230).
  - A rerun only rewrote the entries it re-targeted. Targets now include every generated override.
- Migration 010: `sector_source` gains `wikipedia_gics_hist`, and members added since 008 are labeled `wikipedia_gics`. Membership now labels the sectors it writes.
- The aliases collector applies the overrides weekly. `edgar_facts` now fetches former members too: 298,608 new fact rows, 112 former members with facts.
- **Wipe:** stopped the S05 backfill, marked its `runs` row failed, then `TRUNCATE features` (84.9M rows) in one transaction with both triggers disabled and re-enabled. Full rebuild 2016-01-01 → 2026-11-26 started (`logs/features_rebuild_20261126.log`).
- Scheduler restarted on `5008828` (PID 22950).

## Results
| | |
|---|---|
| CIKs | 120 / 137 mapped (88%, target 80%), all exact; 118 agree with Wikipedia's CIK. ENDP and EVHC are post-reorganization filers where Wikipedia lists a predecessor whose filings don't cover the span. 17 to review. |
| PARA | CIK 1826011 (today's symbol holder) → 813828 (Paramount Global) |
| Sectors | 225 historical GICS, 1 SIC map, 2 unresolved (AABA, WYND, which also have no bars) |
| Bars | 0 / 6. yfinance serves today's holders of SNDK/TE/STI/PARA and nothing for AABA/WYND. Yahoo's TNL (same entity as WYN) is spin-off-adjusted before 2018-06, so it isn't used. Stooq serves a JS proof-of-work bot check (not circumvented). All documented gaps. |

## Files touched
| File | Change |
|---|---|
| `src/qt/collect/survivorship.py`, `scripts/map_delisted_ciks.py` | New |
| `src/qt/db/migrations/010_survivorship.sql` | New |
| `configs/{cik_overrides,sector_overrides,sic_to_gics}.yaml` | New |
| `configs/features.yaml` | `survivorship` section; `yfinance_surv`/`stooq_surv` in source priority |
| `src/qt/common/config.py` | Survivorship, SIC map, override classes |
| `src/qt/collect/{aliases,edgar,membership}.py` | Apply overrides; former members' facts; sector label |
| `src/qt/features/normalize.py` | Exact int64 window max |
| `tests/test_survivorship.py` (29), `tests/test_normalize.py` (+2) | New tests |
| `DECISIONS.md` (D-043), `docs/components/03_features.md` (§9.4) | Docs |

## Tests
- `uv run pytest -q`: **739 passed**. `ruff check` clean.

## State of the system
- DB: migrations up to 010. `features` was truncated; the rebuild is running in the background.
- `features_daily` first unattended run 2026-11-26 23:00 UTC: **ok** (17 min, holiday, crypto only). Its rows were wiped with the rest and are rebuilt by the backfill.
- Scheduler: `com.qt.scheduler` (paper), PID 22950 on `5008828`.
- Git: `5008828` committed, **not pushed**.

## Blockers / surprises
- **`DECISIONS.md` has two `D-041` rows** (S06 model plan + S05 inflation v2). The file is append-only, so I left it. Refer to them as D-041 (model plan) and D-041 (infl v2), or tell me to add a renumbering note.
- **macOS `._*` files** litter `.git/objects` (exFAT AppleDouble), so git prints "garbage found" / "non-monotonic index" warnings. Harmless. I removed the three in `objects/pack`; `dot_clean -m .git` would clear the rest.
- The migration runner waits for `assets` locks: a long build or the survivorship script holding a read transaction blocks it. Apply migrations between jobs.
- `features_daily` took 17 min for a holiday (crypto-only) run on the 8 GB Mac, against a 60 min timeout. Worth watching on a full session day.

## Next exact task
1. When the rebuild finishes: check rows, size (~1.8 GB expected), NaN rates per group, compression. Spot-check that delisted members now have fundamentals ranks.
2. Dom: go through `reports/cik_review_20261126.md` (17 names). To confirm one, add it as `match: manual` to `configs/cik_overrides.yaml`.
3. Exit gate: `features_daily` unattended for 3 sessions (Fri 11-27, Mon 11-30, Tue 12-1); Phase 2 gate (Congress `backfill.done`, 5 unattended days from Fri 11-27). Then close Phase 3.

## Questions to ask Dom next session
- Push `5008828` (+ the docs commit)?
- The 17 CIK review names: confirm any by hand, or accept them as the known gap?
