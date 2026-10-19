# HANDOFF P2-S04 — 2026-10-18

## Done this session
- P2.2 done: `casmi/features/clean.py` (`clean_peaks`, `iso_features`, `keep_row`; constants at top), `tests/test_clean.py` (3 tests; suite 13/13 green), `scripts/clean_stats.py` (full streamed check, log `logs/clean_stats.log`).
- Order per spectrum: iso features [I_M, I_M+1, I_M+2, has×3] from raw peaks (tol max(0.01 Da, 10 ppm), intensities vs raw base peak) -> drop m/z<20, >prec+0.5, zero -> renormalize to base -> floor 0.001 -> top-128 -> sort by m/z -> sqrt. Output mz/inten float32.
- `keep_row` (train only): precursor 50–2000, |ppm| ≤20 (NaN kept), ≥3 peaks after cleaning.

## Results / numbers
- Train 2,539,608 -> 2,314,108 kept (91.12%). Drops: ppm 119,478 · min_peaks 105,945 · prec_range 77.
- Peaks kept p5/p50/p95: train 5/30/128 (10.0% at cap), test 13/51/128 (15.4% at cap). Test: 0 dropped.
- Iso present M/M+1/M+2: train 47.9/21.2/11.5%, test 56.0/49.5/30.3% (matches audit).
- Speed 19–26 µs/spectrum single-core numpy (target ≤10 ms) — no numba needed here.

## Decisions & approvals
- none new. `[2M…]` adducts NOT dropped (341,105 kept rows, 13.4%): dropping is a data-source change (gate 3); decide at P2.5.

## Traps discovered
- `ListArray.offsets/.values.to_numpy()` per batch is the fast path; avoid `to_pylist()` on peak lists.
- ppm NaN must be kept explicitly (`abs(nan) > 20` is False, so the comparison form keeps it).

## Background jobs
- none

## Next session starts with
- 🛑 P2.3 validation-design gate (pending_gates). Plan text already holds the proposal; get Dom's yes, then freeze splits keyed on `structures.parquet` ik14.

## Open questions for Dom
- P2.3 approve as written? Drop `[2M…]` adducts from training (audit proposal)? Final-pick count (assumed 2)? Delete stale Windows copy?
