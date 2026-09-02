# Watcher revisit — implementation plan

Companion to `watcher_svm_repair_plan.md`. That plan ran to completion:
Phases 0–5 are done (0–2 committed, 3–5 in the working tree), and its verdict
was **"ship nothing, agent stays on the deterministic rules."** The A/B/C
diagnosis in that document was fixed or consciously declined. What remains are
the blockers it parked at the end plus the scattered deferred items. This plan
re-opens the watcher investigation to clear them.

**Injection scope decision (Dom): A-lite (market-only).** Phase D (2.5
slack-free add interventions) is therefore dropped; `animal_add_one` /
`crop_add_two` keep needing pre-existing slack. Critical path is
A → B+C+E → F → G → H → I, roughly one week of work to the Phase G
regeneration.

---

## What this buys, and what it does not

It fixes the **data funnel** so `product:*:avoid` and premium
`product:*:increase` can produce signed rows at all. Today they cannot: the
`:avoid` deviation removes only 1–2 tiles out of ~19 (too small to move a
bounded-horizon money label), and the checkpoint lands ~day 2–6, before the
glutter's premium supply matures and craters the price, so the label window
never sees the crash (`watcher_svm_repair_plan.md` §"Phase 5.2 wiring + a
blocking finding" and §"If the watcher is revisited").

It does **not** address the core Phase 5 finding that the one output which
trained (`land_expansion`) overfit to family-specific patterns
(leave-one-family-out pooled balanced accuracy 0.42). Making `product:*:avoid`
trainable does not guarantee a transferable opponent-adaptive signal. **Phase H
is a real decision gate and may again conclude "ship nothing."** Treat this as
"make the SVM investigable," not "ship the SVM."

## Closed on inspection — no work required

| Deferred item | Status |
|---|---|
| Phase 1.4 — `--min-per-class 1` "SMOKE TEST ONLY" warning | Present, `scripts/train_watcher.py:763-765` |
| Phase 1.4 — training doc names the `--min-per-class` flag | Present, `train.md:144-145` (repo-root `train.md`, not `MD files/`) |
| A3 / 3.4 — stale "impossible gate" section in `tune.md` | Already rewritten; "Superseded corpus" banner in place |
| 3.5 — dead fields (`competitive_expansion_pressure`, `invalid_noop_purchase_indicators`, double `overflow`) | Removed in Phase 3 |

---

## Phase A — market-only divest primitive (~2 days)

Blocker (b) needs an `:avoid` deviation that removes *all* of X's exposure, not
2 units. Clearing a planted tile is a farmer tile action (`DIG`) that needs a
unit to path across the board over several turns — that fights the focal's plan
and is no longer a bounded deviation. A-lite avoids that entirely:

- New deviation kinds `crop_divest_all` (crops) and `animal_defer_all`
  (animals). At every turn from the checkpoint to `checkpoint + label horizon`,
  the intervention strips **every** matching future order out of
  `action["market"]` — `BUY_SEED X` for crops, `BUY_ANIMAL <species>` for
  animals — not just the first two units on one turn. Incremental exposure to X
  goes to zero over the window; already-planted tiles are left alone.
- Opportunistic `DIG` injection into `action["farmer"]` / `action["hands"]`
  **only when a focal unit is already standing on an X plant/animal tile** at
  that turn (read the unit position from the obs; no routing, no pathfinding).
  This is the only farmer-action write A-lite makes, and it is a no-op whenever
  the focal's units are elsewhere.
- `_intervention_eligibility` gains a reason for "focal never emits an X
  purchase and never stands on an X tile in the window" so a genuinely dead
  `(output, family, opponent)` combination is rejected once rather than
  producing an unverifiable pair.

**Files:** `scripts/counterfactual_dataset.py` — `RecordingAgent._intervene`
(around :101), `_intervention_eligibility` (:201), `DEVIATIONS` (:51),
`REMOVAL_KINDS` (:574).

## Phase B — wire the full-divest deviation into `product:X:avoid`

- `deviation_for_output` (:558-571): `product:X:avoid` maps to
  `crop_divest_all` / `animal_defer_all` instead of `crop_remove_two` /
  `animal_defer_one`.
- `_VERIFY_METRICS` / `_execution_verified` (:264 / :282): verification for the
  new kinds is "focal X unit count (`_crop_units` / `_animal_units`) stays at
  least K below baseline at every step across the window," replacing the
  single "dropped by ≥ 1 once" check.
- `label_product` (:411) already blends near-term money delta with an X-scoped
  value-flow delta and only signs a row when they agree — no label-math change.
  The larger deviation is what finally pushes the money delta past the neutral
  band.
- `modeled_output_id` (:606) updated for the new kinds.

## Phase C — floor the checkpoint at the opponent's volume day

The label window must contain the price crash. Lands together with Phase B —
for crops the focal often emits its X buys on day ~2 while the glut crash
starts day ~10, so there is no valid pre-emission checkpoint; only Phase B's
divest-over-a-window deviation (no day-0 emission dependency) makes a later
checkpoint usable.

- `_install_sales_probe` (:469): generalize to record **both** players' realised
  sell volume by product (currently focal-only), or add a second install keyed
  to `1 - idx`.
- New `_opponent_volume_days(sales_log, product, threshold)` alongside
  `_order_emission_days` (:500): first day the opponent's cumulative realised
  sales of X clear a units-per-day bar.
- `checkpoint_for_deviation` (:577): new `min_day` parameter; the eligible /
  usable day set is filtered to `day >= min_day`. Applies to `product:*:avoid`
  (all kinds) and premium `product:*:increase` (`crop_add_two`,
  `crop_redirect_two`, `animal_add_one` where the product is in
  `PREMIUM_PRODUCTS`).
- `generate_pair` (:627): the baseline probe replay already runs for
  `REMOVAL_KINDS`; extend the trigger to premium `:increase`, and have the one
  probe yield both the emission days and the opponent-volume days so no extra
  replay is added.
- New `watcher_config` constants: `OPPONENT_VOLUME_UNITS_PER_DAY` and a note on
  which outputs the floor applies to.

## Phase D — dropped (A-lite)

2.5 (add interventions without pre-existing slack) needs farmer-action
injection to `BUILD_COOP` / `BUILD_PASTURE` or plant on an idle tile, which
A-lite does not build. `animal_add_one` / `crop_add_two` stay gated on a
pre-existing empty structure / affordable spare cash. Accepted limitation.

## Phase E — feature-matrix normalization (independent; shares the v5 bump)

`shared_features.extract_features` (:62): normalize count features by their
natural caps rather than leaning entirely on `StandardScaler` —
better-conditioned at n ≈ 150.

- tile counts ÷ 64 (4 quadrants × 16), `*_quadrants` ÷ 4, crop-age buckets ÷
  tile cap, `*_hands` ÷ a fixed sane maximum, `own_shed_*` / `own_carried_*` ÷
  100 (shed capacity), `own_seed_*` ÷ a fixed cap.
- `price_ratio_*` and `market_displacement_*` are already normalized — leave
  them.
- `SCHEMA_VERSION` → `counterfactual-v5` (so `SCHEMA_HASH` changes). This is the
  single bump that also covers the Phase B/C deviation-contract changes.
- `train_watcher.py` already reports `constant_feature_columns` /
  `effective_feature_count`; confirm the numbers still make sense at v5.

*Flag only, out of scope:* Phase 5.2 wants to inspect opponent-supply-**trend**
coefficients, but no trend feature exists (only static `opponent_crop_*` /
`opponent_animal_*` counts), and adding one needs history at the checkpoint obs
that `extract_features` does not receive. Raise separately if wanted.

## Phase F — precompute the probe in `generate_until_quota` (perf; low risk)

The focal policies (`strategy_follower`, `ring`, `leader_clone`) are
opponent-blind per `MD files/CLAUDE.md`, so emission / volume days are roughly
stable per `(family, opponent)`.

- `generate_until_quota.py` runs one probe replay per `(family, opponent)` at
  startup, writes a sidecar JSON (emission days + opponent-volume days).
- `counterfactual_dataset.py` loads that cache and skips its own probe when a
  matching entry is present. Extend the existing `--checkpoint-days`
  passthrough (`generate_until_quota.py:186`) with a `--probe-cache` path.
- Fix the stale `signed_quota_unreachable` default `min_signed=40`
  (`generate_until_quota.py:81`) → `wc.MIN_SIGNED`.

## Phase G — schema bump, regenerate, re-run the Phase 4 stats gate

- One bump only: `counterfactual-v4` → `counterfactual-v5`, covering Phases B,
  C and E together.
- Full regeneration as `data/counterfactual_dataset.v6.jsonl` via
  `generate_until_quota.py` — opponents = the three glutters + `starter`,
  `MIN_SIGNED = 75`. Launch with `run_in_background` per
  `tune.md` §"Generation-run reliability".
- `train_watcher.py` on the v6 corpus. Export an output only if **all** hold:
  CV mean ≥ 0.62, bootstrap test CI lower ≥ 0.58, LOFO pooled ≥ 0.55, confirm
  split within the test CI width of the test point estimate. If nothing passes,
  the SVM does not ship — an acceptable Phase 4 outcome.

## Phase H — re-run the adaptivity decision (Phases 5.2 + 5.3)

- **5.2:** retrain with `product:*:avoid` now trainable. Inspect the fitted
  coefficients on `price_ratio_*`, `market_displacement_*` and `opponent_*`. If
  they are near zero, the SVM is not learning the adaptive signal regardless of
  accuracy — same failure mode the plan already names.
- **5.3:** `scripts/phase5_rules_tuning.py` is already built. Re-run the OFAT
  sweep (`OPPONENT_SIGNAL_WEIGHT`, `TOWN_EXPOSURE_MULTIPLIER`,
  `OPPONENT_EXPANSION_THRESHOLD`) against the v6 glut regime; screen n=10,
  confirm top candidates n=30/opponent including `starter` as a
  non-glut regression guard.
- **Decision gate.** Promote whichever of legacy rules / tuned rules / SVM wins
  paired final-money CI. May again be "ship nothing new."

## Phase I — docs

- Append a dated "Watcher revisited" entry to `watcher_svm_repair_plan.md` §6.
- Resolve the four stale `MD files/train.md` references in
  `watcher_svm_repair_plan.md` and `tune.md` §"Dead (output, strategy-family)
  combinations" — the file is repo-root `train.md`.
- Point `tune.md`'s training / readiness commands at
  `data/counterfactual_dataset.v6.jsonl`.

---

## Tests (mirroring `watcher_svm_repair_plan.md` §4)

- `test_counterfactual_dataset.py`:
  - opponent-volume-day probe returns the expected day for a synthetic sales
    log;
  - checkpoint is floored to that day for `:avoid` and premium `:increase`;
  - `crop_divest_all` strips every future `BUY_SEED X` across the window, not
    just two units;
  - opportunistic injected `DIG` fires only when a focal unit is on an X tile
    and never corrupts the pre-checkpoint action hash;
  - end-to-end: an **accepted** `product:MELON:avoid` row with a **signed**
    (`-1` / `+1`) label comes out under a glut opponent — impossible on the v4
    corpus.
- `test_shared_features.py`: v5 `SCHEMA_HASH`; normalized feature values fall in
  the expected ranges; constant-column report.
- `test_generate_until_quota.py`: probe-cache precompute and passthrough;
  `signed_quota_unreachable` uses `wc.MIN_SIGNED`.
- `test_watcher_config.py`: new constants present; the three-tool readiness
  agreement still holds for a fixed corpus.

---

## Sequencing

```
A ── B ──┐
         C ──┐
E ───────────┼── G ── H ── I
F ───────────┘
```

- B, C and E must land under the same `counterfactual-v5` bump.
- A / B / C are tightly coupled inside `counterfactual_dataset.py`; E and F are
  the only independent pieces. Not worth a multi-agent fleet — one agent (or
  Dom) holding the full `counterfactual_dataset.py` context produces a more
  consistent result.
- A-lite path: ~1 week to the Phase G regeneration. Phase G's generation run
  and Phase H's paired gameplay are the long wall-clock items.

## What ships regardless of the Phase H verdict

The glut opponents (`strategies_v2/glut_*.py`), `MIN_SIGNED = 75`,
`generate_until_quota.py` `--opponents` / `--outputs` / `--probe-cache`
passthroughs, `phase5_rules_tuning.py`, the v6 corpus, and the feature
normalization — all useful for any future watcher work even if the SVM stays
dormant.

---

## Progress log

### 2026-09-02 — Phases A–I complete. Verdict: ship nothing new.

All phases executed. A/B/C (divest deviations + opponent-volume checkpoint
floor), E (feature normalization, `counterfactual-v5`), F (`--probe-cache`,
`signed_quota_unreachable` → `wc.MIN_SIGNED`) landed in the working tree.
Phase G regenerated `data/counterfactual_dataset.v6.jsonl` (13,317 pairs /
5,282 accepted) and re-ran the trainer — **0 accepted**. The A-lite funnel
made `product:*:avoid` produce signed rows for the first time
(`melon:avoid` 186 `-1` vs ~3 on v4) but the rarer signed class stays far
below quota — divesting a floored product rarely hurts, an economic wall.

Phase H:
- **5.2** — `scripts/phase_h_coefficient_audit.py`: adaptive-feature
  (`price_ratio_*` / `market_displacement_*` / `opponent_*`) L1 coefficient
  share is 35–54% on the three fitted outputs — not degenerate-zero, but
  incoherent (scattered sign, unrelated products) and LOFO ≈ 0.50 on all
  three. Family-specific overfit, not a transferable adaptive signal.
- **5.3** — `phase5_rules_tuning.py` re-run: no core-3 constant tuning beats
  the shipped baseline vs the glutters; best screen candidate
  (`TOWN_EXPOSURE_MULTIPLIER` 1.5) fails confirm.
- **Decision gate** — SVM has no exportable model; tuned rules lose the
  paired CI; **legacy rules stay, constants unchanged.**

Phase I — this log, the "Watcher revisited" entry in
`watcher_svm_repair_plan.md` §6, `repo-root train.md` disambiguation, and
`tune.md` pointed at `data/counterfactual_dataset.v6.jsonl` / `MIN_SIGNED` 75.

Full detail: `watcher_svm_repair_plan.md` §6 "2026-09-02 — Watcher revisited".
