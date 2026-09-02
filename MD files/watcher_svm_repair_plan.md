# Watcher SVM — repair & adaptivity plan

Built from a full code-review pass over the watcher SVM stack (data generation,
counterfactual labeling, features, trainer, acceptance gates, export, harness /
`main.py` integration). Every claim below was re-verified against the code.

The goal is not "make the SVM export a file." It is: **the submitted agent
adapts its production to the shared market and a real opponent, and that
adaptation is worth money by season end.** Some of the fixes below make the
current design trainable; others change the design because the current framing
cannot produce an opponent-adaptive signal even if it trained perfectly.

---

## 1. Diagnosis — three independent failure classes plus one strategic question

### A. The submission is physically disconnected from the model

- **A1 — `main.py` never imports the model.** `scripts/main.py:154-155` sets
  `_watcher_features = None` / `_watcher_model = None` as plain literals. There
  is no `import shared_features` / `import watcher_model` anywhere in `main.py`.
  The entire `svm_ready` branch (`main.py:759-905`) is dead code —
  `_watcher_model is not None` is always false. `strategy_follower.py:154-159`
  still has the `try: import …` hook, so the dev script would pick a model up,
  but the actual Kaggle submission cannot. The sell-gate wiring added in commit
  `6bd3ac9` (`_svm_sell_frac`, `import watcher_model`) has been fully reverted
  out of `main.py`; the sell gate now uses only the static
  `SELL_MIN_PRICE_FRAC`. **Net: a perfectly trained, exported, flag-flipped
  model changes nothing in the submitted agent.** Everything else in this plan
  is moot for scoring until this is fixed.

- **A2 — `:exit` / `:retain` outputs have no runtime consumer.**
  `watcher_signals` (`strategy_follower.py:890-905`, and the dead copy in
  `main.py`) only ever queries `product:{p}:increase`, `product:{p}:avoid`,
  `land_expansion`, `labor_expansion`. But `generate_until_quota.py:15-23`,
  `watcher_readiness.py:13-21`, and `counterfactual_dataset.deviation_for_output`
  all generate and audit `product:{MELON,STRAWBERRY,MILK,WOOL}:{retain,exit}`.
  Effort spent bringing those to quota produces models nothing reads.

- **A3 — doc drift.** `MD files/tune.md` §"Important acceptance-gate issue"
  describes the pre-`6bd3ac9` gate (balanced accuracy vs. raw majority
  prevalence, "impossible above 90% skew"). The current `gate()`
  (`train_watcher.py:344-367`) already compares against a fixed 0.5. The doc's
  headline critique is stale and will send the next person down the wrong path.

### B. The acceptance gates cannot be passed by any realistic corpus

- **B1 — leave-one-family-out is structurally unpassable.**
  `gate_leave_one_family_out` (`train_watcher.py:370-375`) requires **every**
  family entry to be `status == "evaluated"` *and* pass `gate()` (bacc ≥ 0.60).
  But `leave_one_family_out` (`:308-338`) marks a family `"skipped"` whenever
  either the held-out pool or the fit pool has `< min_per_class` of either sign
  (`:319-320`). The corpora are heavily family-imbalanced. Worked example from
  the current `land_expansion` (the single best output):
  `strategy_follower` -1:58/0:40/+1:47, `ring.agent` -1:40/0:35/+1:27,
  `leader_clone.agent` -1:0/0:18/+1:1. Holding out `leader_clone` → held pool
  has 0 negative signed rows → `"skipped"` → the whole gate returns false.
  land_expansion clears every other check and still cannot be accepted. This is
  a primary reason "0 accepted" persists even when the nominal data quota is
  met.

- **B2 — eligibility and LOFO use inconsistent "usable family" definitions.**
  `run()` (`train_watcher.py:490-491`) counts a family usable if it has ≥1 of
  each sign across **all** its rows, *including the test split*. LOFO only ever
  sees train+val (`:308-310`). So an output can clear `min_families=2` on
  test-only signed rows in the second family, then fail LOFO because that family
  has no signed rows in the train+val pool.

- **B3 — the margin clause in `gate()` is vacuous.** `bacc ≥ 0.60` and
  `(bacc − 0.5) ≥ 0.10` are algebraically identical (`0.5 + 0.10 == 0.60`). The
  second clause can never diverge from the first. `majority_baseline_unweighted`
  is still computed and threaded through every report (`:301-302`, `:393`,
  `:438`) as if load-bearing; it is not a gate input. Dead machinery that
  invites misreading.

- **B4 — `min_per_class` default of 1 is a silent footgun.**
  `run(..., min_per_class=1)` and CLI default `--min-per-class 1`
  (`train_watcher.py:453,601`). `check_split_adequacy` and
  `leave_one_family_out` both take this value. Forget `--min-per-class 5` and
  training proceeds gating on balanced accuracy computed from a single sample
  per class (bacc ∈ {0, 0.5, 1.0}). Repo-root `train.md` step 6 does not emphasize the
  flag; `tune.md`'s command includes it.

- **B5 — no cross-validation, no confidence interval, test split is both the
  tuning-feedback endpoint and the headline number.** `gate()` fires on test
  bacc ≥ 0.60 from one partition holding ~6-15 signed cases per class. `C` is
  chosen as best-of-7 on an equally tiny validation set (`select_model:251-264`).
  P(bacc ≥ 0.60 | true skill = 0.5) at ~10/class is roughly 0.2-0.25 per output;
  across ~20 outputs a spurious pass is likely and a real pass is
  indistinguishable from one. The only output that ever trained (old combined
  `competitive_expansion`, repo-root `train.md:31-36`) scored val 0.420 / test 0.474 —
  below chance.

- **B6 — threshold drift across the toolchain.** `min_accepted`:
  `train_watcher` 150 / `generate_until_quota` 160 / `watcher_readiness` 150.
  `min_signed`: 40 / 50 / 40. `min_per_class`: `train_watcher` 1 (default) vs
  `watcher_readiness` hard-coded 5. `watcher_readiness` can report an output
  `ready_for_training` under rules the trainer's own defaults don't enforce,
  and vice-versa.

### C. The data funnel cannot produce signal

Measured on `data/counterfactual_dataset.v3.jsonl` (summary JSON): 19,800 pairs
→ 2,498 accepted (12.6%) → class counts -1:314 / 0:1687 / +1:497, i.e. ~811
signed rows spread across ~30 output ids (~27 each; quota wants ≥80 with ≥40
per sign). Four compounding mechanisms:

- **C1 — interventions are ineligible or never applied against a scripted
  focal.** `_intervention_eligibility` (`counterfactual_dataset.py:195-224`):
  `crop_remove_two` / `animal_defer_one` / `hand_remove_one` / `land_defer`
  require a *matching future order already present in the same turn's action*
  (`:212-216`); `animal_add_one` requires a pre-existing **empty** COOP/PASTURE
  (`:203`) — a script that builds-and-fills a structure in one turn never has
  one; `crop_add_two` / `land_advance` / `hand_add_one` require spare cash a
  cash-tight script rarely has. `tune.md` §"Dead (output, strategy-family)
  combinations" documents whole outputs at 2000/2000 attempts, 0 accepted.
  Rejection buckets: `unapplied_intervention` 9,347, `intervention_ineligible`
  4,970.

- **C2 — execution verification is over-strict and mistimed.**
  `_execution_verified` (`counterfactual_dataset.py:227-255`) demands a specific
  state counter move **exactly one env step** after the order. A `BUY_SEED`
  that is immediately planted (seeds count returns to baseline), a `HIRE` with
  onboarding travel, land that unlocks with lag, or a script that would have
  made the same purchase anyway all defeat it. Largest single rejection bucket:
  `execution_not_verified` 14,265. It is rejecting *mechanically valid*
  interventions, not just no-ops.

- **C3 — the safety filter censors the positive class.** `safety_regression`
  (`counterfactual_dataset.py:296-305`) rejects a pair if **any** lower-is-better
  metric increases, including `ending_saleable_inventory` and
  `bankruptcy_indicators` (`min(money_curve) <= 0`). Every "increase / advance /
  hire" intervention that actually works tends to leave a few extra unsold units
  at season end and/or deepen the early cash trough — so the exact interventions
  that would earn a `+1` label are thrown out as unsafe. `safety_regression`
  rejections: 6,362, disproportionately would-be `+1`. Visible in the resulting
  skew: `labor_expansion` -1:3 / 0:124 / +1:3; `land_expansion` +1 mostly from
  one family.

- **C4 — ±$250 neutral band vs. a one-shot nudge over a 720-step season.**
  `label_delta` (`counterfactual_dataset.py:308-315`) labels on final-money
  delta. The intervention is a single bounded deviation (2 seeds / 1 animal /
  1 hand) that then reverts to script. Its effect on *season-end* money after
  30 shared-market days is < $250 for anything but premium products. `tune.md`
  confirms: `product:{CARROT,TOMATO,STRAWBERRY,WHEAT}:avoid` have 84-110
  accepted rows and **100% are label 0**. The label is far downstream of the
  decision and dominated by weed/shop RNG plus opponent variance.

### D. Strategic question — is the counterfactual SVM the right instrument at all?

Per `MD files/CLAUDE.md` the watcher exists because the market is **shared** and
a competent opponent converging on melon / livestock can crater the price you
realize. For the SVM to capture that, its training data must contain (a)
opponents that actually sell at volume and (b) a label that responds to the
resulting price collapse. Neither holds today:

- The three training opponents (`starter`, `strategies_v2.ring`,
  `strategies_v2.leader_clone`) are the weak / scripted set `CLAUDE.md` itself
  flags as never stress-testing real sell volume. The glut regime is essentially
  absent from the training distribution.
- The label (season-end money after one one-shot deviation) has tiny
  signal-to-noise — hence the 0.474 test balanced accuracy on the one output
  that trained.
- Deployment applies the score **persistently every turn** as a multiplicative
  `watcher_factor` (`strategy_follower.py:611-614, 646-649`, capped 0.5-1.35)
  and as hand / quadrant deltas — a different quantity from the one-shot
  marginal effect the label measures.

**The deterministic rules path already computes the opponent-supply forecast and
glut risk it needs** (`watcher_signals` in `strategy_follower.py`, the
six-day public-farm supply forecast, price/base-price ratio, inventory
displacement). The SVM is only supposed to *replace the parts supported by
counterfactual evidence* (`SVM.md` §"Assumptions"). Phase 5 below makes
"SVM vs. tuned rules" an explicit, evidence-gated decision rather than an
assumption.

---

## 2. Repair plan

Phases are ordered by dependency. Phase 0 and Phase 1 are cheap and unblock
everything. Phase 2-3 are the real work. Phase 4-5 decide whether the SVM ships
at all. Phase 6 is promotion.

> **Status (see the Progress log at the end for detail):**
> Phase 0 ✅ landed · Phase 1 ✅ landed · Phase 2 ✅ landed (2.5 + one 2.4
> optimisation deferred) · Phase 3 ✅ landed (schema → `counterfactual-v4`) ·
> Phase 4 ✅ run — **SVM does not ship this iteration, agent stays on rules**
> (0/6 quota-met outputs clear the split/family gates; see log) ·
> Phase 5 ✅ run — **ship nothing new, agent stays on rules.** 5.1 landed
> (3 glutter opponents); 5.2 the SVM does not learn a transferable adaptive
> signal (best output LOFO 0.42); 5.3 no tuning of the core watcher constants
> beats the shipped baseline vs glutters. **Phase 6 skipped** (SVM not
> picked). See log for the two label-contract blockers to fix first if the
> watcher is revisited.

### Phase 0 — Reconnect the submission (0.5 day) — ✅ landed

**Purpose:** make the SVM path in `main.py` real, so any later model actually
affects the score, and lock it with a test so it cannot silently revert again.

0.1 Add to `main.py`, next to the current literals at `:154-155`, the same
guarded import `strategy_follower.py:154-159` uses:

```python
try:
    import shared_features as _watcher_features
    import watcher_model as _watcher_model
except (ImportError, AttributeError):
    _watcher_features = None
    _watcher_model = None
```

0.2 Re-verify the `svm_ready` gate (`main.py:759-765`) and the scoring block
(`:859-906`) are byte-for-byte equivalent to the `strategy_follower.py` copy.
They have drifted once; make them identical or extract the shared logic.

0.3 Decide the sell-gate question explicitly: either restore the `6bd3ac9`
`_svm_sell_frac` wiring in `main.py` or delete its remnants and document that
the sell gate is intentionally static. Do not leave half of it.

0.4 **New test** `scripts/test_main_watcher_integration.py`: `import main`,
monkeypatch `main._watcher_features` / `main._watcher_model` with a stub module
exposing a valid `SCHEMA_VERSION` / `SCHEMA_HASH` / `FEATURE_ORDER` / `MODELS`
and a `score()` returning fixed values, call `main.watcher_signals(obs,
"linear_svm")`, and assert the stub's scores actually move
`product_attractiveness` / `land_expansion_pressure`. Also assert that with
`_watcher_model = None` the function returns the pure rules result and never
raises. This is the test that would have caught A1.

0.5 Add a submission-build assertion (`build_submission.py` /
`test_build_submission.py`): the packaged `main.py` contains the import, and the
packaged artifact imports neither `numpy` nor `sklearn` at module load. AST-walk
like `test_train_watcher.py:332-364` already does for the model.

**Exit:** with a hand-written toy `watcher_model.py` in place, a 100-turn smoke
of `main.agent` shows the toy scores changing actions; removing the file falls
back to rules cleanly; all watcher tests pass.

### Phase 1 — Make the gates reachable and honest (1 day) — ✅ landed

**Purpose:** a genuinely skillful model should be able to pass; a lucky one
should not.

1.1 **Fix LOFO (B1).** Replace `gate_leave_one_family_out` with a *pooled
held-family* evaluation: refit on all-but-one family, collect predictions for
every held-out family, then compute **one** balanced accuracy over the pooled
held-out signed rows and gate that against the threshold with a bootstrap CI
lower bound (see 1.5). Families that individually lack both signs in the pool
contribute their rows to the pooled metric but do not each need to independently
clear 0.60. Keep a per-family breakdown in the report as a diagnostic only.

1.2 **Unify the "usable family" definition (B2).** One predicate, used by both
`run()` eligibility and LOFO: a family is usable for an output iff it has
≥ `min_per_class` of **both** signs in **train ∪ val** (never test). Eligibility
requires ≥ `min_families` usable families under this definition. Test-only
signed rows never count toward usability.

1.3 **Collapse the vacuous margin clause (B3).** Delete
`ACCEPT_MIN_MARGIN_OVER_MAJORITY` and the second clause of `gate()`. Keep a
single documented constant `ACCEPT_MIN_BALANCED_ACC` and justify its value in a
comment (see 1.5 — it should become "CI lower bound ≥ X", not "point estimate
≥ 0.60"). Drop `majority_baseline_unweighted` from gate-adjacent report fields
or clearly relabel it `diagnostic_only`.

1.4 **Raise `min_per_class` default to 5 (B4)** in both the `run()` signature
and the CLI default. Make `--min-per-class 1` emit a loud "SMOKE TEST ONLY"
warning. Update repo-root `train.md` step 6 to state the flag explicitly.

1.5 **Add resampling-based evaluation (B5).** Two changes:
   - **k-fold CV over seed groups** for `C` selection: instead of one val split,
     do grouped k-fold (group = seed group) on train ∪ val, pick `C` by mean
     CV balanced accuracy, tie-break to smaller `C`. The test split stays
     untouched.
   - **Bootstrap CI on the test metric**: resample the test signed rows B=2000
     times, report the 5th percentile of balanced accuracy, and gate on
     **CI lower bound ≥ 0.58** (tunable) rather than point estimate ≥ 0.60.
     This is what makes a 10-per-class "pass" honest.

1.6 **Single source of truth for thresholds (B6).** Put `MIN_ACCEPTED`,
`MIN_SIGNED`, `MIN_PER_CLASS`, `MIN_FAMILIES` in one module (e.g. add to
`shared_features.py` or a new `watcher_config.py`) and import them into
`train_watcher.py`, `generate_until_quota.py`, `watcher_readiness.py`. Delete
the local copies. Add a test asserting all three tools report the same
readiness verdict for a fixed corpus.

1.7 **Add a true confirmation split.** Currently train/val/test where test is
also the tuning-feedback signal. Split seed groups 60/15/15/10 →
train / val / test / **confirm**. `test` is used once for the gate; `confirm`
is opened only after a candidate is otherwise finalized and its number goes in
the report untouched by any decision. If `confirm` and `test` disagree by more
than the CI width, do not export.

**Exit:** `test_train_watcher.py` gains a case with realistically imbalanced
synthetic families where the *old* LOFO gate fails and the *new* pooled gate
passes for a genuinely separable target and fails for a noise target.

### Phase 2 — Open the data funnel (3-5 days, the core work) — ✅ landed

**Purpose:** get from ~27 signed rows/output to ≥ 150 accepted / ≥ 40 per sign
for a meaningful set of *coherent* outputs, without manufacturing fake signal.

2.1 **Retime and loosen execution verification (C2).** In `_execution_verified`:
   - Compare state over a **window** (checkpoint → checkpoint + K env steps,
     K≈3-5), not a single step, and accept if the intended delta holds at *any*
     step in the window or if the *cumulative* purchase count (seeds bought,
     not seeds on hand) differs. Track "seeds purchased to date" and "hands ever
     hired" as monotonic counters in the recorder rather than reading
     instantaneous inventory that planting immediately drains.
   - For `crop_add_two`, verify against tiles planted-or-seeds-bought, not
     seeds-on-hand.
   - Record the *reason* an intervention failed verification with the metric
     trajectory in `intervention_audit`, so dead combinations are visible
     without re-deriving them.

2.2 **Make the safety filter a soft flag, not a hard reject, for the
outcome-relevant metrics (C3).** Split `safety_regression` into:
   - **Hard rejects** (genuine causal-validity failures): `escaped_animals`,
     `missed_watering_deaths`, `episode_status_failures`,
     `market_order_cap_violations`. An intervention that causes an animal to
     starve is not a clean example of "more exposure was good/bad."
   - **Soft flags** recorded on the row but not rejecting it:
     `ending_saleable_inventory`, `shed_overflow_indicators`,
     `bankruptcy_indicators` (rename to `cash_trough_touched`),
     `feed_risk_tile_turns`, `water_risk_tile_turns`. A profitable-but-messy
     pair is a legitimate `+1` — the model *should* see that region. The
     follower already clamps by cash / deadline capacity at runtime
     (`strategy_follower.py:614,672-673`), so it will not blindly act on a
     score that implies an unsafe purchase.
   - Keep `min(money_curve) <= 0` as a **hard** reject only if the episode
     actually ended in failure; a temporary dip to a low-but-positive balance
     is normal and should be a soft flag.

2.3 **Shorten the label horizon (C4).** This is the single highest-leverage
label change. Replace "season-end money delta" with a **product-scoped,
bounded-horizon revenue outcome**:
   - For `product:X:{increase,avoid}`: label on the change in **realized
     revenue from product X over the H days following the checkpoint**
     (H≈6-8, matching the deterministic forecast horizon), plus the change in
     **held/decayed inventory value** of X over that window. This directly
     encodes "the shared market rewarded / punished more exposure to X *now*"
     and strips out most of the whole-season RNG.
   - For `land_expansion` / `labor_expansion`: label on money delta over
     H≈10-12 days (capacity investments pay back slower) *minus* the up-front
     cost already sunk, so the label is "did the marginal capacity earn its
     keep in the near term."
   - Make the neutral band a **fraction of the horizon's baseline revenue for
     that product** (e.g. ±8%), not a flat $250 — a flat band is why cheap
     crops are all-neutral and premium crops are not.
   - Keep season-end money delta as a *secondary recorded field* for auditing,
     not the label.

2.4 **Move removal / defer checkpoints off day 0 (C1).**
`checkpoint_for_deviation` (`counterfactual_dataset.py:~371`) pins removal
checkpoints to day 0. Schedule them at the days the focal policy *actually*
emits the matching purchase (drive from `extract_replay_states.py` /
`--checkpoint-days` and from a dry-run pass that records when each focal family
buys what). If a family never emits the decision in any window, mark
`(output, family)` dead once — `generate_until_quota.py:57-66` already has the
dead-family machinery; feed it this data up front instead of discovering it
over 2000 wasted attempts.

2.5 **Add "add" interventions that don't need pre-existing slack.** For
`animal_add_one`, allow the intervention to *also* emit the enabling structure
purchase (BUY + place COOP/PASTURE then BUY_ANIMAL) as one bounded deviation,
so a build-and-fill focal policy can be perturbed. For `crop_add_two`, allow
redirecting an *idle* tile the focal would have left empty, not only filling a
future planned slot. Keep each a single bounded deviation.

2.6 **Fix `crop_redirect_two` unit loss (`counterfactual_dataset.py:123-128`)**
— on a partial redirect, create an order for the un-redirected remainder so
units don't vanish. Then wire `crop_redirect_two` into
`deviation_for_output` so `product:X:increase` rows come from both add and
redirect paths.

2.7 **Harden the recorder identity (S12).** `_run` relies on the in-process
`RecordingAgent` instance being mutated (`:318-328`). Add an assertion that
`wrapped.checkpoint_obs is not None` after the run and raise a clear error (not
a silent per-pair reject) if kaggle copied the callable. Consider recording
checkpoint obs / audit to a per-worker file keyed by seed instead of relying on
closure identity.

**Exit:** a 2,000-pair regeneration run against the current 3 opponents yields
≥ 150 accepted with ≥ 40 per sign for at least `land_expansion`,
`labor_expansion`, `product:MELON:{increase,avoid}`,
`product:MILK:{increase,avoid}`, and ≥ 2 usable families each under the Phase 1
definition.

### Phase 3 — Target coherence and cleanup (1-2 days)

3.1 **Delete the dead outputs (A2).** Remove `product:*:{exit,retain}` from
`deviation_for_output`, `generate_until_quota.py`, `watcher_readiness.py`, and
the premium_exit / premium_retain intervention family in
`counterfactual_dataset.py` — unless Phase 5 decides to add an `:exit` consumer
to `watcher_signals`. Do not generate data nothing reads.

3.2 **Keep `land_expansion` / `labor_expansion` split** (already done, correct
call) but verify each is one causal decision after the Phase 2 label change —
if `labor_expansion` still can't produce ≥ 40 `+1` rows across families, it
stays on rules.

3.3 **Audit the feature matrix for the new sample sizes** (`shared_features.py`):
   - Drop or merge the perfectly-collinear time features: with all checkpoints
     at hour 0, `turn = 24·day`, `remaining_fraction` is linear in `day`, and
     `phase_{opening,midgame,endgame}` are step functions of `day`. Keep `day`
     (or `remaining_fraction`) and the phase one-hot; drop `turn` and `hour`.
   - Normalize counts by their natural caps as `SVM.md` §Features specifies,
     rather than leaning entirely on `StandardScaler` — better-conditioned at
     n≈150.
   - Flag constant columns (unused crops / animals / shops for the focal
     families present) in the training report so the effective dimensionality
     is visible against the row count.
   - Do **not** change `SCHEMA_VERSION` / `SCHEMA_HASH` casually — it is
     checked end to end (`shared_features.py:54`, dataset rows, exported module,
     `watcher_signals:766-768`). Any feature change bumps the version and
     invalidates the existing corpus; batch all feature changes into one bump.

3.4 **Update `tune.md` (A3)** — remove the stale "impossible gate" section,
replace with the actual current gate and the Phase 1 CI-based gate.

3.5 **Remove dead fields** — `competitive_expansion_pressure`
(`strategy_follower.py:916` / `main.py`) is returned but never read;
`invalid_noop_purchase_indicators` is always 0; `overflow` computed twice
(`counterfactual_dataset.py:286-287`); `sample_weight` is hard-coded 1.0 so
`--reweight-mode` / `to_weights` / `renormalize_weights` are no-ops — either
make the generator emit real weights (e.g. inverse propensity for how often
that state appears) or delete the reweighting path.

### Phase 4 — Statistical validity checkpoint (0.5 day, gate)

Run the Phase 1 trainer on the Phase 2/3 corpus. For each candidate output:
grouped k-fold CV balanced accuracy, bootstrap test CI, LOFO pooled metric,
and the untouched `confirm` split. **Export an output only if all of:**
CV mean ≥ 0.62, test CI lower bound ≥ 0.58, LOFO pooled ≥ 0.55,
`confirm` within CI of `test`. If nothing passes, the SVM does not ship this
iteration and the agent stays on rules — that is an acceptable outcome, not a
failure.

### Phase 5 — Adaptivity: does the SVM beat tuned rules? (2-3 days, decision gate)

This is where "adaptable and as profitable as possible" is actually decided.

5.1 **Build glutter opponents.** Add 2-3 focal *opponents* (not focal policies)
under `strategies_v2/` that deliberately flood premium markets: a melon-mono
high-volume seller, a cow+sheep livestock-flooder, and a mixed premium
dumper. These create the shared-market price-collapse regime `CLAUDE.md` says
the watcher exists for. Regenerate a slice of the Phase 2 corpus with these as
opponents (the focal policy stays `strategy_follower` / `ring` / `leader_clone`).

5.2 **Re-train and re-evaluate** with the glut regime in-distribution.
Specifically check: does `product:MELON:avoid` / `product:WOOL:avoid` /
`product:MILK:avoid` now learn "opponent is scaling this product AND
price/base-price is falling → avoid"? Inspect the fitted coefficients on the
opponent-supply and price-ratio features — if those coefficients are near zero,
the SVM is not learning the adaptive signal regardless of accuracy.

5.3 **Head-to-head vs. tuned rules.** Take the deterministic `watcher_signals`
rules path and tune its constants (`OPPONENT_SIGNAL_WEIGHT`,
`TOWN_EXPOSURE_MULTIPLIER`, the glut-risk blend) against the glutter opponents
using the harness. Then run paired gameplay: **legacy rules vs. tuned rules vs.
SVM-backed**, identical seeds, alternating order, ≥ 30 episodes per opponent
including the glutters. Promote whichever wins on paired final-money CI — and
if tuned rules match or beat the SVM, **ship tuned rules** and keep the SVM
code dormant. The SVM is not the goal; adaptive profit is.

### Phase 6 — Export and promotion (1 day, only if Phase 5 picks the SVM)

6.1 Export `scripts/watcher_model.py` for the passing outputs only. The
exporter and runtime loader are already correct (score parity to 6 dp, `[-1,1]`
clip, `None` on unknown output / length mismatch / non-finite, schema-hash
mismatch disables the whole SVM) — verified in review, no changes needed except
handling the `margin_scale ≈ 1e-6` degeneracy: if the 95th-percentile training
margin is below a floor (e.g. 1e-3), refuse to export that output rather than
shipping a bang-bang ±1 score.

6.2 Run `python -m unittest discover -s scripts` + `py_compile` on
`shared_features.py`, `watcher_model.py`, `strategy_follower.py`, `main.py`.

6.3 Flip `WATCHER_BACKEND = "linear_svm"` in **both** `strategy_follower.py` and
`main.py`. Rebuild the submission via `build_submission.py`. Assert the packaged
artifact imports no numpy / sklearn and `agent(obs)` is the last top-level
callable.

6.4 Paired gameplay promotion per `SVM.md` §"A/B Testing": paired-money 95% CI
above zero overall, no opponent with a confirmed regression > 1% of baseline,
no new animal escapes or order-cap violations, safety metrics ≤ 5% worse than
the rules predecessor. Store the round under `ab_rounds/`.

---

## 3. Sequencing and effort

| Phase | Blocks | Effort | Ships value if it's the last one done? |
|---|---|---|---|
| 0 — reconnect submission | everything | 0.5 d | No (but nothing works without it) |
| 1 — reachable gates | 4, 5 | 1 d | No |
| 2 — data funnel | 3, 4 | 3-5 d | No |
| 3 — coherence / cleanup | 4 | 1-2 d | Marginally (dead-code removal) |
| 4 — stats checkpoint | 5 | 0.5 d | It's a gate |
| 5 — adaptivity decision | 6 | 2-3 d | **Yes — tuned rules can ship here** |
| 6 — SVM promotion | — | 1 d | Yes, if Phase 5 picks it |

**Fastest path to shipped adaptive profit:** Phase 0 → Phase 5.3's rules-tuning
sub-task → paired gameplay → ship tuned rules. That is ~2 days and does not
depend on the SVM training at all. Do that first; treat Phases 1-4 + 6 as the
"can the SVM beat the tuned rules" investigation running behind it.

## 4. New tests required

- `test_main_watcher_integration.py` — `import main`, stub model, assert scores
  move actions and `None` model falls back cleanly (Phase 0).
- `test_build_submission.py` additions — packaged `main.py` has the import; no
  numpy / sklearn at load (Phase 0).
- `test_train_watcher.py` additions — imbalanced-family synthetic case: old
  LOFO fails, new pooled LOFO passes for separable target, fails for noise;
  `--min-per-class 1` warns; bootstrap CI lower bound gates as expected
  (Phase 1).
- `test_watcher_config.py` — all three tools agree on readiness for a fixed
  corpus (Phase 1).
- `test_counterfactual_dataset.py` additions — windowed execution verification
  accepts an immediately-planted seed; soft-flag metrics don't reject a
  profitable-messy pair; `crop_redirect_two` conserves units; recorder-identity
  assertion raises loudly (Phase 2).
- One end-to-end test that runs a tiny real `kaggle_environments` episode
  through `generate_pair` and asserts an accepted row comes out — currently
  nothing exercises the actual simulation path (Phase 2).

## 5. What "done" means

The watcher is fully repaired when:

1. `main.py` loads and uses `watcher_model.py` when present, falls back to rules
   when absent, and a test proves both — **and this is true in the packaged
   submission.**
2. At least one *coherent, consumed* output has ≥ 150 execution-verified accepted
   rows, ≥ 40 per sign, ≥ 2 usable families in train ∪ val, with a
   shorter-horizon product-scoped label.
3. The gates are reachable by a skillful model and reject a lucky one: grouped
   CV, bootstrap test CI lower bound, pooled LOFO, untouched confirm split.
4. Either the SVM or the tuned deterministic rules wins paired gameplay against
   a set of opponents **that includes market-glutting sellers**, on final-money
   CI, with no safety regression.
5. Whatever ships is the one that won that comparison — not "the SVM because we
   built it."

---

## 6. Progress log

### 2026-08-30 — Phases 0-2

**Phase 0 — ✅ landed (verified present this pass, not re-audited line by line).**
The submission is reconnected at build time rather than in `scripts/main.py`:
`build_submission.py:_replace_watcher_import` swaps the `_watcher_features =
None` / `_watcher_model = None` literals for the embedded feature + model
namespaces (or leaves them `None` when no model is supplied).
`scripts/strategy_follower.py` keeps the guarded `import shared_features` /
`import watcher_model` for dev runs. Tests: `test_main_watcher_integration.py`
(stub model moves `product_attractiveness` / `land_expansion_pressure`, `None`
model falls back to rules) and `test_build_submission.py` (packaged artifact
imports no numpy/sklearn) — 8 tests, passing. 0.3 decided: the sell gate is
intentionally static; the `6bd3ac9` `_svm_sell_frac` experiment is reverted and
documented in both `main.py` and `strategy_follower.py`.

**Phase 1 — ✅ landed (verified present this pass).**
`scripts/watcher_config.py` is the single source of truth for
`MIN_ACCEPTED` / `MIN_SIGNED` / `MIN_PER_CLASS` / `MIN_FAMILIES` and the gate
constants. `train_watcher.py` has: pooled leave-one-family-out
(`leave_one_family_out`, one balanced accuracy over the pooled held-out signed
rows, gated at `ACCEPT_MIN_LOFO_POOLED`); one "usable family" predicate
(`usable_families`, ≥ `MIN_PER_CLASS` of both signs in train ∪ val only);
grouped k-fold CV for `C` selection (`select_c_grouped_kfold`); a percentile
bootstrap CI on the test balanced accuracy gated on the 5th-percentile
(`bootstrap_balanced_accuracy_ci`, `gate`); and a `confirm` split
(`gate_confirm`, `REQUIRE_CONFIRM_SPLIT`). `MIN_PER_CLASS` default is 5. Not
re-verified: the `--min-per-class 1` "SMOKE TEST ONLY" warning text and whether
repo-root `train.md` step 6 names the flag.

**Phase 2 — ✅ landed.** One mixed generation run went from ~12% accepted to
~75%. Commit `5c6608a`.

- **2.1 execution verification** — `_execution_verified(kind, item, benv, ienv,
  idx, applied_step, window_steps)` now scans a window
  (`wc.EXEC_VERIFY_WINDOW_STEPS`, 4 days) and accepts if the intended delta
  holds at *any* step. Crop checks use `_crop_units` = seeds on hand + planted
  tiles ("seeds acquired to date"), so an immediately-planted seed still
  counts. The failure evidence (metric, window, end values) is recorded in
  `intervention_audit.execution_evidence` and `summarize_rows` aggregates it
  under `execution_verify_failure_metrics`.
- **2.2 safety** — `safety_regression` → `safety_split(baseline, intervention)`
  returning `(hard_fail, soft_flags)`. Hard: `escaped_animals`,
  `missed_watering_deaths`, `episode_status_failures`. Soft (recorded on the
  row as `soft_safety_flags`, never rejecting): `feed_risk_tile_turns`,
  `water_risk_tile_turns`, `shed_overflow_indicators`,
  `ending_saleable_inventory`, `invalid_noop_purchase_indicators`,
  `cash_trough_touched` (renamed from `bankruptcy_indicators`), and
  **`market_order_cap_violations`**. *Deviation from the plan:* the plan lists
  `market_order_cap_violations` as hard. The smoke showed it rejecting most
  otherwise-usable `ring`-family rows for queue over-fill the engine silently
  truncates (`queues.append(q[:max_orders])`), and the `ring` focal already
  violates it in the baseline. Moved to soft. Revisit if a later phase shows a
  case where the intervention's own appended order truncates a *meaningful*
  focal order.
- **2.3 label** — season-end money delta → bounded-horizon **blend**
  (`compute_label` → `label_product` / `label_capacity`):
  - Primary signal: near-term total-money delta,
    `Δmoney_i − Δmoney_b` over the horizon.
  - `product:X:{increase,avoid}` gets a signed label only when that money
    delta clears the band **and** agrees in sign with an X-scoped value-flow
    delta: `Σ(units of X sold × price) + (work-in-progress + shed X value)`,
    intervention vs baseline. Sold volume and price come from an instrumented
    market phase (`_install_sales_probe` wraps `kaggriculture._process_market`
    — during that phase only SELL reduces a player's shed). This keeps signed
    rows causally about exposure to X, at the cost of yield (Dom's call).
  - Per-product horizon (`wc.PRODUCT_HORIZON_DAYS`: 7d wheat/carrot/egg …
    13d melon/tomato/milk/wool, 16d strawberry) so a slow crop's outcome is
    actually inside the window. `_held_product_value` counts `yield_units`
    still on the plant/animal so a planted-but-unharvested crop shows value.
  - Band = `max(8% × baseline window X revenue, $120)` for products,
    `max(8% × |Δmoney_b|, $350)` for capacity.
  - Capacity outputs (`land_expansion` / `labor_expansion`) use money delta
    only, no X-flow gate.
  - `season_end_delta` + `season_end_label` kept as recorded audit fields.
- **2.4 checkpoints off day 0** — for removal/defer deviations `generate_pair`
  runs one probe replay, `_order_emission_days` finds the days the focal
  actually emits the cancelled order, and `checkpoint_for_deviation` picks from
  the earlier half of that set. *Not done:* precomputing these days in
  `generate_until_quota` and passing them in, so the probe replay is currently
  one extra episode per removal/defer pair.
- **2.5 add interventions without pre-existing slack — NOT done.** Needs
  farmer-action injection (BUILD_COOP / BUILD_PASTURE / PLANT are unit actions,
  not market orders) and a design decision from Dom on how a bounded deviation
  overrides `action["farmer"]` / `action["hands"]` without fighting the focal's
  own plan. `animal_add_one` / `crop_add_two` against build-and-fill focals
  still fail here.
- **2.6 crop_redirect_two** — partial redirect now splits the order (keeps the
  un-redirected units as their own `BUY_SEED`) instead of shrinking and losing
  them. Wired into `product:X:increase` with `item=None` (redirect the first
  non-target `BUY_SEED` the focal emits), so it does not need the focal to buy
  one specific source crop. `_intervention_eligibility` gained
  `no_redirectable_purchase`.
- **2.7 recorder identity** — `_run` raises `RuntimeError` if
  `wrapped.checkpoint_obs is None` after a run whose checkpoint was in range.
  The per-worker-file alternative was not needed.

**Schema:** `shared_features.SCHEMA_VERSION` → `counterfactual-v3` (and so
`SCHEMA_HASH`). The row *contract* changed (label semantics, safety fields), so
the existing `data/counterfactual_dataset.v3.jsonl` rows are now schema-skipped
by `train_watcher.py` and **the corpus must be regenerated**. Phase 3.3 will
bump again for the feature-matrix audit.

**Tests:** `test_counterfactual_dataset.py` +11 (windowed verification, soft vs
hard safety, redirect unit conservation, blend label sign-agreement, capacity
label, recorder-identity RuntimeError, end-to-end `generate_pair`). 89 pass.

**Smoke (not the Phase 2 exit criterion — that is a 2000-pair run):** a 60-pair
mixed run accepted 45/60 (75% vs 12.6% on the old v3 corpus); rejection buckets
flattened (execution_not_verified 8, hard_safety 7, ineligible 7, unapplied 7).
Signed-row yield against `strategy_follower` is thin (~11%) because that focal
re-plans around the nudge; weaker families do better
(`ring` / `land_expansion`: 3 signed of 9 accepted). The blend's low signed
yield is expected and leans on Phase 5's glutter opponents.

**Next:** run the 2000-pair regeneration (`generate_until_quota.py`) and check
it against the Phase 2 exit bar (≥ 150 accepted / ≥ 40 per sign / ≥ 2 usable
families for `land_expansion`, `labor_expansion`, `product:MELON:*`,
`product:MILK:*`). Then Phase 3.

### 2026-08-31 — Phase 3

**Phase 3 — ✅ landed. Schema `counterfactual-v3` → `counterfactual-v4`.**

- **3.1 dead outputs — already gone in code, docs cleaned.** The Phase 2
  rewrite already dropped `product:*:{exit,retain}` and the
  `premium_exit` / `premium_retain` intervention family from
  `counterfactual_dataset.py`, `generate_until_quota.py` and
  `watcher_readiness.py` — nothing in `watcher_signals` ever read an
  `:exit` / `:retain` score. This pass only purged the stale `:exit` / `:retain`
  rows and per-output counts from `MD files/tune.md`.
- **3.2 land/labor split — verified.** `land_expansion` and `labor_expansion`
  are distinct `output_id`s driven by distinct deviations (`land_*` vs
  `hand_*`), each one causal decision. Whether `labor_expansion` clears ≥ 40
  `+1` rows is a Phase 4 data outcome.
- **3.3 feature-matrix audit.** `shared_features.py`: dropped `turn`, `hour`
  **and** `day` — every counterfactual checkpoint is at hour 0, so all three
  were mutually collinear and collinear with `remaining_fraction`. Kept
  `remaining_fraction` + the non-linear `phase_{opening,midgame,endgame}`
  one-hot. `FEATURE_ORDER` 129 → 126. `train_watcher.py` now reports
  `constant_feature_columns` / `effective_feature_count` per output so the
  effective dimensionality is visible against the row count. *Not done:*
  per-natural-cap count normalisation — `StandardScaler` already fronts the
  model and the model design is Dom's; deferred as a conditioning tweak, not a
  contract change.
- **3.4 tune.md** — the stale "impossible gate" section was already rewritten in
  Phase 1; this pass added a "Superseded corpus" banner (v3 label + v4 feature
  changes, exit/retain removal), dropped the retired per-output snapshot, and
  pointed the training / readiness commands at
  `data/counterfactual_dataset.v4.jsonl`.
- **3.5 dead fields** — removed `competitive_expansion_pressure` (returned by
  `watcher_signals` in both `strategy_follower.py` and `main.py`, never read)
  and `invalid_noop_purchase_indicators` (always 0; was in `_metrics` and
  `SOFT_SAFETY_METRICS`). `overflow` is computed once now (the "twice" was
  pre-Phase-2). `sample_weight` / `--reweight-mode` kept — `train_watcher.py`'s
  docstring assigns that field to Dom's preprocessing pipeline; left as a live
  hook rather than deleted.

**Schema:** `shared_features.SCHEMA_VERSION` → `counterfactual-v4`,
`SCHEMA_HASH` → `2ce35fef…`. The `counterfactual-v3` corpus never existed on
disk (the file `data/counterfactual_dataset.v3.jsonl` is still
`counterfactual-v2`), so nothing is invalidated that was not already stale. The
corpus must be generated fresh as `data/counterfactual_dataset.v4.jsonl`.

**`generate_until_quota.py` — projection-based dead-output skip added.** The
first regeneration batches confirmed the `tune.md` prior: low/mid-value crops
(`WHEAT/CARROT/TOMATO/STRAWBERRY/EGG`, both directions) produce accepted rows
that are ~100 % neutral — the one-shot bounded nudge never moves near-term
money past the label band — so they can never reach 40/sign and would each burn
the full 2000-attempt cap. New `signed_quota_unreachable()` abandons an output
once ≥ 400 attempts show its rarer signed class projected (rate × remaining
budget) to fall short of the quota. Reported as `signed-quota-unreachable` in
the final summary. Tests: `test_generate_until_quota.py` +3.

**Tests:** `test_shared_features.py` (126 features / v4 hash),
`test_counterfactual_dataset.py` (v4 schema string), `test_train_watcher.py`
(separable fixture now uses a ±6σ single-dim signal, robust to the 129 → 126
RNG-stream shift), `test_watcher_config.py` +1 (readiness vs. trainer must
agree when a family is fit-pool sign-imbalanced), `test_generate_until_quota.py`
+3. 94 pass. `build_submission.py` still packages (rules mode); artifact
compiles.

### 2026-08-31 — Phase 4 (statistical validity checkpoint)

**Result: no output ships. The agent stays on the deterministic rules path.**
Per the plan this is an acceptable Phase 4 outcome, not a failure.

**Corpus.** `data/counterfactual_dataset.v4.jsonl` — 11,442 pairs, 5,035
execution-verified accepted, 3 opponents (`starter`, `strategies_v2.ring`,
`strategies_v2.leader_clone`), focal `strategy_follower` / `ring` /
`leader_clone`. Six outputs reached the Phase 2 exit bar (≥ 150 accepted /
≥ 40 per sign): `land_expansion`, `labor_expansion`, `product:MELON:avoid`,
`product:MILK:increase`, `product:MILK:avoid`, `product:WOOL:avoid`. The other
twelve are `signed-quota-unreachable` (cheap/mid crops; premium `:increase`,
which produces plenty of `+1` but essentially no `-1` against non-glutting
opponents) or `all-families-dead` (`product:EGG:avoid`, 0 accepted).

**Why the six quota-met outputs still fail the Phase 1 gates** (none even
reached model-fitting — all blocked at split/family adequacy):

| output | blocker | detail |
|---|---|---|
| `land_expansion` | `test_insufficient_class_counts` | test `-1 = 4` (need ≥ 5); confirm `+1 = 4` |
| `labor_expansion` | `confirm_insufficient_class_counts` | confirm `-1 = 2` |
| `product:MILK:increase` | `confirm_insufficient_class_counts` | confirm `-1 = 2` |
| `product:WOOL:avoid` | `confirm_insufficient_class_counts` | confirm `-1 = 3` |
| `product:MELON:avoid` | `insufficient_usable_strategy_families` | only `strategy_follower` has ≥ 5 of both signs in train ∪ val; `ring` has `-1 = 1` |
| `product:MILK:avoid` | `insufficient_usable_strategy_families` | only `strategy_follower` usable; `ring`/`leader_clone` fit-pool sign-imbalanced |

Two distinct causes:

1. **The Phase 2 per-sign quota (40) is too low for the split geometry.** With
   the 60/15/15/10 train/val/test/**confirm** split, ~40-45 rare-sign rows put
   only ~4 in the 10 % confirm split — under the 5-per-class floor. `MIN_SIGNED`
   needs to rise to ~75-80 (≈ 8 in confirm with RNG headroom) for
   `land_expansion` / `labor_expansion` / `MILK:increase` / `WOOL:avoid`, which
   have healthy family diversity and only miss on `confirm`/`test`. This is a
   `watcher_config` correction, not a design change — it re-invalidates nothing
   (features/label contract unchanged) but needs another generation pass.
2. **Premium `product:*:avoid` cannot reach two usable families with the
   scripted opponents.** `ring` / `leader_clone` almost never over-expose melon
   or milk, so removing that exposure almost never helps near-term money → they
   produce `+1` (the rarer sign for `:avoid`) at a rate near zero *in the fit
   pool*. No amount of same-opponent generation fixes this. **This is exactly
   the regime Phase 5's glutter opponents create.**

**Tool consistency fix (Phase 1.6 residual, found here).** `watcher_readiness.py`
reported `product:MELON:avoid` as `ready_for_training` while `train_watcher.py`
rejected it — readiness counted a family "usable" on *any* `-1` and *any* `+1`
across *all* splits, the trainer requires ≥ `MIN_PER_CLASS` of both in
train ∪ val only. Readiness now uses the trainer's predicate; both report
**0 eligible outputs** for the v4 corpus. Regression test added.

**Artifacts:** `artifacts/watcher_training_report.v4.{json,md}`,
`artifacts/watcher_readiness_v4_report.json`.

**Next (Phase 5, and a Phase 2 quota bump):**
- Raise `watcher_config.MIN_SIGNED` to ~75 and regenerate `land_expansion`,
  `labor_expansion`, `product:MILK:increase`, `product:WOOL:avoid` — the four
  that only miss on small-split class counts — then re-run Phase 4 for them.
  (They may still fail the CV / test-CI / LOFO gates; the old combined output
  scored 0.474 test balanced accuracy. That is what Phase 4 is for.)
- Build the Phase 5.1 glutter opponents to open up `product:*:avoid` family
  diversity and to put the price-collapse regime in-distribution at all.
- Phase 5.3's rules-tuning sub-task is still the fastest path to shipped
  adaptive profit and does not depend on any of the above.

### 2026-08-31 — Phase 5.1 (glutter opponents)

**Phase 5.1 — ✅ landed.** Three market-flooding opponents added under
`strategies_v2/`, each a scripted focal *opponent* (not a focal policy):

| file | build | flood target | validated behaviour (5 seeds, vs `strategy_follower`, 720 steps) |
|---|---|---|---|
| `glut_melon.py` | 32 melon tiles NW+NE, 6 hands, `SELL_MIN_PRICE_FRAC = 0` | MELON | MELON end price **4–7 in 5/5**; opponent solvent (~$14k) |
| `glut_livestock.py` | 16 COW / 10 SHEEP NW+NE, 7 hands, dump milk+wool | WOOL (MILK partial) | WOOL floored to **5 in 3/5**, MILK dented (76–241) in those; solvent ($56–77k) |
| `glut_mixed.py` | 16 COW / 10 SHEEP + 26 melon, 3 quadrants, dump all three | MILK (MELON/WOOL partial) | MILK **3–42 in 4/5**; occasionally near-insolvent late by design (price stays floored after) |

Baseline for reference: `strategy_follower` vs `starter` holds MELON ≈ 232,
MILK ≈ 310, WOOL ≈ 240. Across the set, every glut-fragile premium product
(`MD files/CLAUDE.md` "Why the watcher matters": melon, wool, milk) is
reliably crashed by at least one glutter. All use the shared
`strategies_v2/engine.py` helpers; each supports direct `python glut_*.py`
self-check. `resolve_opponent` in `counterfactual_dataset.py` already loads
`strategies_v2.<name>.agent` with no change. 94 script tests still pass.

**Design note (deviation from a literal reading of 5.1).** The plan names "a
melon-mono high-volume seller, a cow+sheep livestock-flooder, and a mixed
premium dumper." An all-quadrant / 8-hand build of each was tried first and
stalled or bankrupted before its first yield on the $3000 opening bankroll
(no income until melon day 10 / cow day 8 while paying ~$50/day hires). The
shipped builds are held near ring/solid's proven two-to-three-quadrant,
6–7-hand envelope; the flood comes from the **zero sell-gate** (dump every
unit every turn regardless of price), not from raw capacity. Each glutter
now floods one product reliably rather than all three — which is why there
are three of them.

### 2026-08-31 — Phase 5.2 wiring + a blocking finding

**Decisions taken (Dom):** full sequence (5.2 then 5.3); the glutters
**replace** ring/leader_clone as the *opponent* set (focal families
unchanged); `MIN_SIGNED` 40 → **75**.

**Wiring landed:**
- `watcher_config.MIN_SIGNED = 75` (comment explains the confirm-split
  geometry). `test_train_watcher.py` end-to-end fixture scaled up
  (fam_a/fam_b train counts) to keep ≥ 75 per sign. 94 tests pass.
- `generate_until_quota.py`: new `DEFAULT_OPPONENTS =
  (starter, glut_melon, glut_livestock, glut_mixed)` and a `--opponents`
  passthrough to `counterfactual_dataset.py` (previously the quota driver
  always used the hard-coded scripted-opponent default).

**Blocking finding — the glut regime does not, on its own, make
`product:*:avoid` trainable.** A 24-pair smoke of `product:MELON:avoid`
(focal `strategy_follower`, opponents = the 3 glutters) with MELON crashed to
$4–7: **18 accepted → −1:3, 0:15, +1:0.** Same "no rarer sign" wall as
Phase 4. Cause is not the opponent — it is the **perturbation size**:
`crop_remove_two` drops 1–2 melon tiles out of ~19, and over the 13-day melon
label window that moves near-term money by `$0` on most pairs and by
`−$1300` on the few that register (the freed tile still loses to strategy
follower's own melon even at a floored price). To get `+1` rows for `:avoid`
the deviation or the label has to change — options, all Phase-2-contract
(schema bump), Dom's call:
  - (a) bigger `:avoid` deviation — remove *all* remaining melon exposure at
    the checkpoint, not 2 tiles;
  - (b) credit the freed capital's redeployment in the label instead of just
    the removed product's flow;
  - (c) pin the label window fully inside the price crash (checkpoint ≥ the
    day the opponent's premium supply starts maturing, ~day 8–10).

**`land_expansion` / `labor_expansion` / `product:MILK:increase` /
`product:WOOL:avoid`** — the four that in Phase 4 only missed on confirm/test
class counts — are not blocked by this; they just need the `MIN_SIGNED = 75`
generation run. That run is safe to start now.

**Decisions taken (Dom, round 2):** skip `product:*:avoid` structural fix
this iteration — generate only the outputs that work; 5.3 grid = "core 3"
one-factor-at-a-time (`OPPONENT_SIGNAL_WEIGHT` 2–8, `TOWN_EXPOSURE_MULTIPLIER`
1.5–3.0, `OPPONENT_EXPANSION_THRESHOLD` −0.2–0.3), screen n=10 then confirm
n=30/opponent.

**Generation run — launched (background).**
`generate_until_quota.py --output data/counterfactual_dataset.v5.jsonl
--seed-base 500000 --outputs land_expansion labor_expansion
product:MILK:increase product:WOOL:avoid product:MELON:increase
product:WOOL:increase product:STRAWBERRY:increase product:EGG:increase`
(opponents = the 3 glutters + starter, `MIN_SIGNED = 75`). Corpus is a fresh
file; schema string is still `counterfactual-v4` (no feature/label change —
the `:avoid` contract change was declined). Premium `:increase` is included
because a glut regime should finally give it `-1` rows ("added melon
exposure while melon is floored → hurt").

**5.3 harness — built.** `scripts/phase5_rules_tuning.py`: OFAT sweep from
the shipped baseline, each variant generated as its own
`scripts/ab_tests/*.py` and run as a subprocess doing
`compare_paired(tuned, strategy_follower.agent, opp, n)` vs the 3 glutters +
starter (starter = non-glut regression guard). Note: the parent-`importlib`
+ `compare_paired` pattern in `run_phase_experiments.py` is **broken under
the 'spawn' start method** on this machine (dynamically-loaded module can't
be unpickled in the worker) — the subprocess-per-variant pattern from
`test_gen.py` is the one that works. Smoke-validated; full screen pending
(hold until generation finishes to avoid CPU oversubscription).

### 2026-08-31 — Phase 5.2 result (retrain with the glut regime)

**Corpus:** `data/counterfactual_dataset.v5.jsonl` — 6,400 pairs, 3,570
accepted, opponents = `starter` + the 3 glutters, focals unchanged.
Quota (`MIN_ACCEPTED` 150 / `MIN_SIGNED` **75**): **met for 3** —
`land_expansion`, `labor_expansion`, `product:MILK:increase`. Unreachable
for `product:{WOOL:avoid, MELON:increase, WOOL:increase, STRAWBERRY:increase,
EGG:increase}` — the glut regime did **not** give premium `:increase` its
missing `-1` rows: the checkpoint lands ~day 2–6, before the glutter's
premium supply matures and craters the price, so adding 2 tiles still nets
positive over the label window (same root cause as the `:avoid` blocker).

**Trainer (`artifacts/watcher_training_report.v5.{json,md}`): 0 accepted.**
Unlike Phase 4, all 3 quota-met outputs reached model fitting:

| output | CV mean | test bacc (CI low) | confirm | LOFO pooled | blocker |
|---|---|---|---|---|---|
| `land_expansion` | **0.73** | **0.81 (0.67)** | **0.71** | **0.42** | leave-one-family-out |
| `product:MILK:increase` | 0.66 | 0.54 (0.37) | 0.67 | 0.49 | test CI |
| `labor_expansion` | 0.60 | 0.47 | 0.55 | 0.50 | CV |

**Phase 5.2's question — "is the SVM learning the opponent-adaptive signal?"
— answered: no.** `land_expansion` clears CV, test *and* confirm but its
leave-one-family-out pooled balanced accuracy is **0.42** (per-family: ring
0.43, `strategy_follower` **0.34** — worse than chance). Trained on two
families it cannot predict the third; it is fitting family-specific patterns,
not a transferable "opponent is scaling capacity → expand / don't" rule.
`milk:increase`'s CV signal (0.66) collapses on the untouched test split
(0.54). This matches the plan's own stated failure mode ("if those
coefficients are near zero, the SVM is not learning the adaptive signal
regardless of accuracy").

**Decision (Phase 4/5 gate): the SVM does not ship. Proceed to Phase 5.3 —
tuned deterministic rules head-to-head. The agent stays on rules; whether it
ships *tuned* rules is 5.3's call.**

### 2026-08-31 — Phase 5.3 result (tune the deterministic rules)

**Harness:** `scripts/phase5_rules_tuning.py` — OFAT sweep from the shipped
baseline (`OPPONENT_SIGNAL_WEIGHT` 4.0, `TOWN_EXPOSURE_MULTIPLIER` 2.0,
`OPPONENT_EXPANSION_THRESHOLD` 0.0), each variant paired vs the shipped
baseline against the 3 glutters + `starter` (regression guard). Screen n=10,
confirm top-3 at n=30. `artifacts/phase5_rules_tuning_report.json`.

**Result: no tuning of the core-3 constants beats the shipped baseline.**

| knob change | screen glut_score | note |
|---|---|---|
| `OPPONENT_SIGNAL_WEIGHT` → 2 / 6 / 8 | −1361 / −1775 / −1502 | all worse |
| `TOWN_EXPOSURE_MULTIPLIER` → 1.5 | +442 (best) | **fails confirm:** n=30 glut_score −120, all glutter CIs straddle 0, and **regresses vs `starter` −587 (CI [−1085, −88])** |
| `TOWN_EXPOSURE_MULTIPLIER` → 2.5 / 3.0 | −680 / −744 | worse |
| `OPPONENT_EXPANSION_THRESHOLD` → −0.2 / 0.15 | **0.0** | **zero effect** — confirmed at n=30; the threshold never changes an action in these matchups |
| `OPPONENT_EXPANSION_THRESHOLD` → 0.3 | −376 | worse |

Note `glut_melon` shows ≈ 0 diff for almost every variant: the watcher
constants barely bite on the melon-glut matchup at all (the follower's melon
build is largely fixed and the watcher only redirects *future vacancies*,
which the melon path holds rather than replants). The watcher has thin
runtime leverage on exactly the case it exists for.

### Phase 5 verdict

**Ship nothing new. The agent stays on the current deterministic rules
(`WATCHER_BACKEND = "rules"`, shipped constants unchanged).**
- 5.2: the SVM does not learn a transferable opponent-adaptive signal (best
  output `land_expansion` has leave-one-family-out 0.42).
- 5.3: no tuning of the core watcher constants beats the shipped baseline
  against market-glutting opponents; the best screen candidate regresses vs
  a weak opponent.

**Phase 6 is skipped** (it runs "only if Phase 5 picks the SVM").

**What was added and kept** (useful regardless of the negative verdict):
`strategies_v2/glut_{melon,livestock,mixed}.py` (glut regime now available as
an opponent for any future work), `watcher_config.MIN_SIGNED = 75`,
`generate_until_quota.py` `--opponents` / `--outputs` passthrough,
`scripts/phase5_rules_tuning.py`, `data/counterfactual_dataset.v5.jsonl` +
`artifacts/watcher_training_report.v5.*` + `artifacts/phase5_rules_tuning_report.json`.

**If the watcher is revisited**, the two blockers to fix first (both Phase-2
label-contract changes, schema bump) are: (a) the counterfactual checkpoint
lands before the opponent's premium supply matures, so the label window
never sees the price crash — pin removal/`:avoid` and premium-`:increase`
checkpoints to ≥ the day the opponent starts selling at volume; (b) the
one-shot ±2-tile deviation is too small to move a bounded-horizon money
label — a `:avoid` deviation should remove *all* of X's exposure at the
checkpoint.

### 2026-09-02 — Watcher revisited (`repair.md` Phases A–I)

`repair.md` re-opened the investigation to clear the two blockers above. Both
were built (injection scope: **A-lite / market-only**, Dom's call — Phase D
dropped):

- **Phases A/B/C — `counterfactual_dataset.py`.** New `crop_divest_all` /
  `animal_defer_all` deviations strip *every* future `BUY_SEED X` /
  `BUY_ANIMAL <species>` order across the label window (not two units on one
  turn), with an opportunistic `DIG` only when a focal unit already stands on
  an X tile. `product:X:avoid` now maps to the full divest; verification is a
  sustained ≥ `DIVEST_VERIFY_MIN_GAP` unit gap held to the window end. A
  both-players sales probe + `_opponent_volume_days` floors the checkpoint for
  every `:avoid` kind and premium `:increase` to the first day the opponent's
  realised X sales clear `OPPONENT_VOLUME_UNITS_PER_DAY` (8), so a glut that
  matures ~day 10 lands inside the horizon.
- **Phase E — `shared_features.py`.** Count features divided by natural caps
  (`TILE_CAP` 64, `HANDS_CAP` 12, `SHED_CAP`/`SEED_CAP`) before
  `StandardScaler`. `SCHEMA_VERSION` → `counterfactual-v5` (one bump covering
  B/C + E). Feature *list* unchanged (126).
- **Phase F — `generate_until_quota.py`.** `--probe-cache`: one probe replay
  per `(family, opponent)` at startup, sidecar JSON, `counterfactual_dataset`
  skips its own probe on a hit. `signed_quota_unreachable` default fixed to
  `wc.MIN_SIGNED`.
- **Phase G — regenerate + Phase 4 gate.** `data/counterfactual_dataset.v6.jsonl`
  — 13,317 pairs, 5,282 accepted; opponents = 3 glutters + `starter`,
  `MIN_SIGNED = 75`. `artifacts/watcher_training_report.v6.*`.

**Phase G result — the A-lite funnel makes `:avoid` *investigable*, not
*trainable*.** `product:MELON:avoid` now yields 186 `-1` rows (v4: ~3) and
`product:MILK:avoid` 97 `-1` / 49 `+1` — the divest deviation and the volume
floor work. But the rarer signed class stays far short of 75 (`melon:avoid`
`+1` = 13): under a glut, divesting the floored product almost never *hurts*,
so the `+1` wall is economic, not a data-volume problem. Quota is met for the
same three outputs as v5 — `land_expansion`, `labor_expansion`,
`product:MILK:increase` — and the trainer accepts **0**:

| output | CV mean | test bacc (CI low) | confirm | LOFO pooled | blocker |
|---|---|---|---|---|---|
| `land_expansion` | 0.71 | 0.76 (0.65) | 0.78 | **0.50** | leave-one-family-out |
| `product:MILK:increase` | 0.63 | 0.69 (**0.56**) | 0.70 | 0.50 | test CI lower < 0.58 |
| `labor_expansion` | **0.61** | 0.68 | 0.44 | 0.50 | CV mean < 0.62 |

**Phase H / 5.2 — coefficient audit** (`scripts/phase_h_coefficient_audit.py`
→ `artifacts/phase_h_coefficient_audit.*`). For each fitted output, the L1
mass of the standardized coefficient vector on the *adaptive* features
(`price_ratio_*` + `market_displacement_*` + `opponent_*`) is **35–54%** —
**not** the near-zero degeneracy the plan named as the clean "not learning it"
signature. But the market coefficients are scattered in sign across unrelated
products (`labor_expansion` leans on `market_displacement_MELON` and
`price_ratio_TOMATO`; `milk:increase` on `market_displacement_WHEAT`), i.e. the
model spends weight on the opponent/market channel but does not encode a
coherent "X is glutted → act on X" rule. Combined with LOFO ≈ 0.50 (chance)
on every fitted output, the read is unchanged from v5: **family-specific
overfit, not a transferable opponent-adaptive signal.**

**Phase H / 5.3 — rules OFAT re-run** on the v6 glut regime
(`scripts/phase5_rules_tuning.py`, screen n=10 / confirm n=30,
`artifacts/phase5_rules_tuning_report.json`):

| knob change | screen glut_score | outcome |
|---|---|---|
| `OPPONENT_SIGNAL_WEIGHT` → 2 / 6 / 8 | −1602 / −1925 / −1980 | all badly worse |
| `TOWN_EXPOSURE_MULTIPLIER` → 1.5 | **+185** (best) | **fails confirm:** n=30 glut_score −107; every glutter CI straddles 0; `starter` −191 (CI [−769, +387]) |
| `TOWN_EXPOSURE_MULTIPLIER` → 2.5 / 3.0 | −299 / −247 | worse |
| `OPPONENT_EXPANSION_THRESHOLD` → −0.2 / 0.15 | **0.0** | zero effect (never changes an action), confirmed |
| `OPPONENT_EXPANSION_THRESHOLD` → 0.3 | −15 | regresses `starter` −638 |

No tuning of the core-3 constants beats the shipped baseline against the
glutters.

### Phase H verdict (unchanged from Phase 5)

**Ship nothing new. The agent stays on the current deterministic rules
(`WATCHER_BACKEND = "rules"`, shipped constants unchanged).** The A-lite
data-funnel work cleared blocker (b) enough to *inspect* the SVM on
`product:*:avoid`, but:

- **SVM:** 0 accepted on v6; no exportable model. The three outputs that
  reach fitting overfit to focal-family patterns (LOFO pooled ≈ 0.50) — the
  5.2 coefficient audit shows adaptive-feature weight is present but
  incoherent, not the transferable rule the design needs.
- **Tuned rules:** no core-3 constant tuning wins a paired final-money CI vs
  the shipped baseline; the one positive screen candidate collapses on
  confirm.
- **Legacy rules:** promoted by default (both challengers lose or fail to
  separate).

`repair.md`'s own framing held: "make the SVM investigable, not ship the
SVM," and Phase H "is a real decision gate and may again conclude ship
nothing." It did.

**Kept regardless of the verdict** (`repair.md` "What ships regardless"):
the A-lite divest deviations + opponent-volume checkpoint floor in
`counterfactual_dataset.py`, `shared_features` normalization
(`counterfactual-v5`), `generate_until_quota.py --probe-cache`,
`data/counterfactual_dataset.v6.jsonl` +
`artifacts/watcher_training_report.v6.*`,
`scripts/phase_h_coefficient_audit.py`, and the refreshed
`artifacts/phase5_rules_tuning_report.json` /
`artifacts/watcher_readiness_report.json`.
