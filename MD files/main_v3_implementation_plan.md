# Main v3 implementation and watcher integration plan

## Purpose

Build a new, evidence-driven `main.py` from the last validated strategy rather
than continuing from the unvalidated 45-melon, three-quadrant submission
artifact. Improve the deterministic strategy in isolated, measurable stages,
then add accepted `watcher_model.py` outputs without allowing an unavailable or
weak model to replace proven rule behavior.

The goal is not to copy leaderboard output numbers blindly. The goal is to
reproduce the mechanisms that appear valuable in replay evidence—gradual
hiring, compact specialist routes, staged crop transitions, and better sale
timing—while proving that each mechanism improves this strategy's own economy.

## Non-negotiable invariants

1. `scripts/strategy_follower.py` is the editable source of truth during
   strategy development.
2. `scripts/main.py` is a generated, self-contained submission artifact. Do
   not develop experimental behavior directly in it.
3. `agent(obs)` must be the last top-level callable in the submitted file.
4. Every returned action must retain this shape:

   ```python
   {"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}
   ```

5. The submission may not import NumPy, scikit-learn, or other development-only
   packages.
6. No watcher output is enabled merely because a model file exists. Each
   output must pass its complete offline acceptance gate and paired gameplay
   promotion test.
7. Missing, malformed, incompatible, rejected, or non-finite model outputs
   must fall back to deterministic rules for only the affected signal.
8. Experimental changes are introduced one mechanism at a time. Do not bundle
   hiring, layout, crop, land, and selling changes into one candidate.
9. All performance comparisons use identical seeds and alternate player order.
10. Generated datasets, reports, replays, and development environments remain
    outside the Kaggle submission.

## Current problem to correct first

The current `scripts/main.py` is not behaviorally equivalent to the validated
editable strategy. It contains an experimental configuration with 45 melon
tiles, three melon hands, three land quadrants, and a day-19 melon exit. That
configuration was not promoted through paired testing. The editable strategy
uses the previously validated scale of 19 melon tiles, two melon hands, and two
quadrants, plus the later planner and watcher-rule layers.

This mismatch makes the submitted result unsuitable as evidence about the
validated strategy or the SVM design. Before optimizing anything, establish a
reproducible source-to-submission build and prove action parity.

## Evidence and promotion standards

### Smoke tests

Smoke tests prove only that an agent loads, returns legal-looking actions, and
finishes a short or full episode. They do not establish performance.

For every candidate:

- compile all touched Python files;
- run at least four short episodes;
- run at least two complete-season episodes;
- check action shape on every turn;
- count exceptions, invalid statuses, silent market-cap truncations, escaped
  animals, watering deaths, bankruptcies, and unsold ending inventory.

### Paired screen

For an isolated strategy change:

- use at least 20 episodes per side;
- use identical seeds for control and candidate;
- alternate which player slot contains the candidate;
- test at minimum `starter`, `strategies_v2.ring.agent`, and
  `strategies_v2.leader_clone.agent`;
- add reconstructed replay-family opponents when they are available.

Report:

- control and candidate mean money;
- mean paired money difference;
- standard deviation and range of paired differences;
- approximate 95% confidence interval;
- win/tie/loss counts;
- safety and operational failures;
- per-opponent results, not only a pooled mean.

### Confirmation and promotion

A candidate that survives screening receives at least 30 episodes per
opponent, preferably 60 for noisy or small effects. Promote only when:

- the pooled paired difference is positive;
- the result is not driven by one weak opponent;
- no representative competitive opponent has a material regression;
- safety and completion rates are no worse than control;
- the mechanism produces the intended behavioral change in traces;
- the final submission artifact reproduces the candidate's actions.

Keep a JSON result and a concise Markdown conclusion for every round. A failed
candidate stays available as evidence but is not retained in the source of
truth.

## Phase 0 — preserve and identify artifacts

### Objective

Make it impossible to confuse the submitted file, editable source, historical
literal replay agent, or an experimental generated variant.

### Work

1. Preserve the exact poorly performing Kaggle file as a dated read-only
   reference under an archival location. If `copy.py` is the file downloaded
   from Kaggle, add it to the workspace before analysis.
2. Record its Kaggle submission ID, public/private score, upload time, and any
   known episode IDs.
3. Compute and record SHA-256 hashes for:

   - the submitted `copy.py`;
   - the current `scripts/main.py`;
   - `scripts/strategy_follower.py`;
   - any bundled `watcher_model.py`.

4. Confirm which file Kaggle actually executed. Do not infer this from local
   filenames.
5. Save a constants manifest for each artifact, including animal mix, hand
   counts, crop tiles, crop exit days, land target, watcher flags, and selling
   thresholds.

### Exit gate

The exact submitted bytes and configuration are known. Any claim about the
poor result refers to that artifact rather than a nearby local version.

## Phase 1 — establish a clean v3 baseline

### Objective

Return to the last validated deterministic scale and make it the explicit v3
control.

### Baseline configuration

Start with the editable strategy's evidence-backed settings:

- `ANIMAL_MIX = {"COW": 11, "SHEEP": 6}`;
- `LIVESTOCK_HANDS = 6`;
- `MELON_HANDS = 2`;
- `MAX_MELON_TILES = 19`;
- `LAND_TARGET_QUADRANTS = 2`;
- `SELL_MIN_PRICE_FRAC = 0.4`;
- deadline planner enabled;
- adaptive targets enabled;
- town watcher rules enabled;
- opponent watcher rules enabled;
- `WATCHER_BACKEND = "rules"`.

Do not enable the SVM in this phase.

### Work

1. Remove submission behavior that exists only in the unvalidated expanded
   artifact.
2. Retain correctness fixes that have independent justification, such as:

   - owned-cell routing and locked-spawn recovery;
   - market-order-cap enforcement;
   - wheat-feed purchasing before discretionary investment;
   - end-of-season action and sale safety;
   - deterministic fallback behavior.

3. Run a trace comparison against the historical validated version to identify
   every remaining behavioral difference.
4. Classify each difference as:

   - intentional validated behavior;
   - correctness-only fix;
   - unvalidated experiment;
   - artifact-generation difference.

5. Remove or gate every unvalidated difference.

### Exit gate

The v3 baseline finishes reliably and reproduces the last promoted strategy
except for documented correctness fixes. Its paired results become the control
for all subsequent phases.

## Phase 2 — create a deterministic source-to-submission build

### Objective

Prevent another mismatch between `strategy_follower.py` and `main.py`.

### Build requirements

Create or extend the established build process so it:

1. Reads the editable source and explicitly selected generated watcher module.
2. Produces one self-contained `scripts/main.py`.
3. Removes demo/test entry points and development-only imports.
4. Preserves constants and feature order exactly.
5. Embeds the pure-Python watcher model only when requested.
6. Leaves `agent(obs)` as the last top-level callable.
7. Refuses to build if:

   - the watcher schema or feature order differs;
   - a scientific runtime dependency remains;
   - another callable appears below `agent`;
   - the action contract test fails;
   - the selected watcher report does not identify accepted outputs.

### Parity test

Build a deterministic parity test that runs the editable source and generated
artifact on copies of the same observations. Compare their returned actions
exactly for:

- synthetic edge-case observations;
- observations from seeded local games;
- a representative sample of extracted leaderboard replay states;
- early, middle, and late-season states;
- locked land, low money, full market queues, feed emergencies, crop maturity,
  liquidation, and missing watcher-model cases.

The test should report the first differing observation, constant, and action.

### Exit gate

At least several thousand representative observations produce byte-equivalent
actions between editable source and built artifact. The artifact passes a full
season and the last-callable check.

## Phase 3 — instrument behavior before changing strategy

### Objective

Measure why money is gained or lost rather than optimizing only final totals.

### Per-episode metrics

Add evaluation-only instrumentation for:

- final money and minimum intraseason cash;
- income and quantity sold by product;
- achieved sale price distribution by product;
- seed, animal, land, feed, and daily hire spending;
- requested versus executed market orders;
- market-order-cap truncations;
- hands requested, hired, and active by day;
- hand move/work/pass shares;
- tiles serviced per hand and by role;
- missed watering, escaped animals, and unfed/unwatered tile-turns;
- land purchase request and execution day;
- unlocked but unused tile-days;
- crop planting, harvest, decay, and exit days;
- unsold ending shed and carried inventory;
- opponent production exposure and shared-market price displacement.

Instrumentation belongs in evaluation tooling, not the Kaggle submission.

### Replay baselines

Use `data/replay_states.jsonl` and replay action data to establish competitive
ranges for:

- daily hand-count curves;
- land purchase timing;
- crop mix by day;
- specialist movement/work ratios;
- sale batch sizes and prices;
- midgame melon exit and strawberry entry;
- late wheat conversion and liquidation.

These are diagnostic targets, not automatic promotion criteria.

### Exit gate

Every subsequent A/B result includes enough operational metrics to explain
whether a money change came from production, prices, labor, land, failures, or
ending inventory.

## Phase 4 — gradual hiring ramp

### Hypothesis

Paying for a near-full crew from day zero consumes cash before production can
repay it. Competitive agents begin near four or five hands and ramp toward
larger crews. A cash- and workload-aware ramp may improve early animal, seed,
and land investment while still providing enough labor later.

### Design

Replace a fixed daily hire count with a target function using only current
observation data:

- day or production phase;
- number of live animals and active crops;
- estimated required work actions;
- travel/capacity margin;
- current cash after feed reserve;
- marginal Fibonacci cost of the next hand;
- remaining season and expected payback.

Candidate mechanisms should be independently testable:

1. Calendar ramp matching replay-family median hand counts.
2. Workload ramp based on service demand.
3. Cash-aware workload ramp that declines when the next hire crowds out feed,
   seeds, or profitable land.

Do not immediately target 12–14 hands. Sweep conservative ceilings first,
including the known safe region around eight or nine daily hires.

### Safety rules

- Never spend the feed reserve on marginal labor.
- Never request more hands than can be paid for after earlier queued orders.
- Respect the ten-order market cap.
- If fewer hands execute than expected, recompute assignments from actual
  `me["hands"]` rather than planned count.

### Measurements

Track daily hire cost, missed service work, asset count, land timing, minimum
cash, production, and final money.

### Exit gate

Promote only a ramp that improves paired money against competitive opponents
without increasing animal/crop losses or creating a cash-collapse tail.

## Phase 5 — fixed livestock specialists and compact routing

### Hypothesis

The strongest replay family keeps several hands in compact livestock zones,
where 58–66% of turns perform work instead of movement. The current broad
assignment may spend too much labor walking.

### Design

1. Partition livestock tiles into contiguous spatial zones.
2. Assign stable hand slots to stable zones for the duration of a day.
3. Keep the farmer or selected hands responsible for shed-facing logistics.
4. Assign remaining hands to crop zones rather than global nearest-needs.
5. Reassign only when:

   - a specialist is missing because fewer hands were hired;
   - its zone has no actionable work;
   - an emergency elsewhere would otherwise cause irreversible loss;
   - the day/phase explicitly changes the production layout.

6. Preserve urgency ordering within each zone:

   - prevent escape or plant death;
   - harvest capped output;
   - feed/water/care;
   - build/place/plant;
   - routine collection;
   - movement or pass.

### Candidate sequence

Test increasingly broad versions:

1. One fixed livestock specialist.
2. Two specialists divided by pasture region.
3. Cow and sheep specialization.
4. Specialist plus dedicated shed runner.

Do not simultaneously change animal counts or land layout.

### Measurements

- work, move, and pass share by hand slot;
- serviced livestock tiles per hand;
- missed feed/care and escaped animals;
- product yield and final money;
- idle specialist turns.

### Exit gate

Promote the smallest specialization scheme that improves labor efficiency and
paired money without reducing coverage under low-hire or emergency states.

## Phase 6 — staged crop transition

### Hypothesis

Competitive replay families converge on an early melon/wheat opening, a
midgame move away from melon toward strawberry, and a late conversion toward
wheat. This may manage shared-market glut better than permanent melon exposure.

### Required separation

Do not combine crop transitions with more land, more hands, or a larger herd.
First test them on the promoted baseline footprint and crew.

### Stage A: determine melon exit behavior

Test one variable at a time:

- last melon seed-purchase day;
- last melon planting day;
- whether existing melon is allowed to mature;
- whether empty former melon tiles remain empty or transition immediately.

Candidate exits should cover a reasonable range around observed market and
harvest timing rather than relying on a single day from replay imitation.

### Stage B: strawberry transition

After selecting a melon exit, test bounded strawberry replacement counts.
Check that strawberry planted late enough to miss its production window is
never purchased. Include travel and tending load in the deadline planner.

### Stage C: late wheat transition

Test replacing expiring crop slots with wheat only while wheat can mature and
be harvested before liquidation. Separate feed-reserve wheat from sellable
wheat in diagnostics.

### Adaptive rules to test

Only after fixed transition schedules are understood, test observation-driven
conditions using:

- live product price relative to base;
- market displacement from equilibrium;
- own and opponent public production exposure;
- unlocked town-shop demand;
- remaining productive days;
- available labor capacity.

### Exit gate

Promote only a transition that beats permanent melon exposure across multiple
opponents and does not strand immature crops or increase unsold inventory.

## Phase 7 — product-specific sale scheduling

### Hypothesis

Frequent small sales can achieve better prices than large dumps, especially
for wheat, while premium goods may benefit from holding through temporary
price depressions. One static threshold does not represent both behaviors.

### Design

Create deterministic product-specific sale policies using:

- product price ratio;
- current shed quantity;
- projected near-term production;
- inventory-cap risk;
- remaining days;
- known town demand;
- opponent public exposure;
- minimum liquidation deadline.

Candidate mechanisms:

1. Small, frequent wheat and fertilizer sales.
2. Maximum batch size per routine sale.
3. Product-specific minimum price thresholds.
4. Overflow override that sells before shed capacity destroys output.
5. Time-decaying hold thresholds as the season ends.
6. Mandatory final liquidation independent of normal thresholds.

### Market-order safety

- Reserve order slots for feed, hires, and critical purchases.
- Ensure sale batching does not starve higher-priority orders.
- Record intended and executed sales separately.
- Test both player orders because simultaneous market processing affects price.

### Exit gate

Promote only if revenue improvement comes from better realized prices or lower
losses, not from accidentally suppressing required purchases or production.

## Phase 8 — reconsider land expansion only after utilization improves

### Hypothesis

Three quadrants failed when added to an economy that could not productively
service them. Gradual labor, specialist routing, and staged crops may change
that constraint, but land must still justify its purchase and operating cost.

### Design

Keep two quadrants as control. Test a third quadrant only when:

- existing unlocked land exceeds a measured utilization threshold;
- the planner has labor capacity for the additional tiles;
- feed and operating reserves remain funded;
- the intended crop has enough season remaining to repay seed, labor, travel,
  and the $2,000 land cost;
- a concrete layout exists for the new quadrant.

Test third-land behavior separately from increasing the herd or maximum crew.

### Exit gate

Require a positive competitive paired result and evidence that newly unlocked
tiles are used profitably. Reject expansion that merely raises gross output
while lowering final cash through land, labor, or glut costs.

## Phase 9 — replay-derived challenger strategies

### Objective

Use replay observations to produce strong evaluation opponents and rule
challengers without treating replay actions as causal labels.

### Work

1. Cluster leaderboard names by behavioral fingerprints rather than treating
   all names as independent strategies.
2. Extract per-family templates for:

   - daily hires;
   - tile layout by phase;
   - animal mix;
   - land schedule;
   - crop transition schedule;
   - role assignment;
   - sell cadence and batch size;
   - liquidation.

3. Implement at least one representative local opponent per genuinely distinct
   family.
4. Validate each reconstruction against held-out replay states using action and
   state-trajectory similarity.
5. Add these opponents to all promotion-quality paired tests.

### Exit gate

Local evaluation covers several genuine competitive behaviors rather than
many aliases of one weak or duplicated policy.

## Phase 10 — prepare causal watcher data for the promoted v3 strategy

### Objective

Train watcher decisions on states produced by the final deterministic economy,
not obsolete strategy versions.

### Data sources

The SVM's causal rows come from same-seed paired local simulations. Replay rows
are used only to choose representative states, timing distributions, and focal
families.

For each pair:

1. Select a focal strategy family and opponent.
2. Run a baseline season.
3. Run an intervention season with identical seed, opponent, and player order.
4. Confirm checkpoint state and all earlier actions match.
5. Confirm the intervention is legal, affordable, actionable, and executes.
6. Record intended-state execution evidence.
7. Compare final money and safety outcomes.

### Labels

```text
+1  intervention improves final money by more than $250 and causes no safety failure
 0  final-money difference is between -$250 and +$250 inclusive
-1  intervention loses more than $250 or introduces a defined safety failure
```

### Outputs

Keep coherent decisions separate:

- `land_expansion`;
- `labor_expansion`;
- `product:<PRODUCT>:increase`;
- `product:<PRODUCT>:avoid`.

Do not restore a combined expansion target unless evidence shows land and labor
share a learnable boundary.

### Replay-guided targeting

Use the observational replay corpus to:

- weight checkpoint days toward when competitive agents make the decision;
- avoid states far outside competitive feature ranges;
- prioritize common and economically meaningful outputs;
- design local focal families reflecting genuine replay clusters;
- report generated-versus-replay feature coverage.

Do not copy replay final rewards into causal labels.

### Quotas

For every eligible output require:

- at least 150 accepted execution-verified rows;
- at least 40 positive labels;
- at least 40 negative labels;
- both signed classes in train, validation, and test;
- at least five examples of each signed class per split;
- multiple usable focal strategy families;
- no seed group crossing splits or families.

Stop after 2,000 attempted pairs for an output if it cannot satisfy these
requirements. Leave that output on deterministic rules.

## Phase 11 — train and accept watcher models offline

### Training protocol

For each quota-complete output:

1. Fit `StandardScaler` on training rows only. Neutral rows may participate in
   scaler fitting.
2. Exclude neutral rows from binary fitting and evaluation.
3. Renormalize signed training weights to mean `1.0` when using the documented
   weighting mode.
4. Fit candidate `LinearSVC` models over the fixed `C` grid.
5. Select `C` by validation balanced accuracy, breaking ties with smaller `C`.
6. Calculate score margin scale only from training decisions.
7. Run leave-one-focal-family-out diagnostics using only train and validation.
8. Evaluate the untouched test split exactly once.

### Offline gate

An output is exportable only when test performance satisfies both:

- balanced accuracy at least `0.60`;
- balanced accuracy at least `0.10` above the majority baseline.

Rejected outputs remain absent from the export. Never adjust features,
thresholds, weights, or hyperparameters after inspecting test results; a new
design requires a new untouched test allocation.

### Pure-Python export

Export only accepted records to `scripts/watcher_model.py`. Verify:

- exact parity with scikit-learn decision scores;
- finite results clipped to `[-1, 1]`;
- schema version, hash, and 129-feature order match runtime extraction;
- no NumPy or scikit-learn imports;
- missing output IDs return `None`;
- malformed or non-finite features return `None` safely.

## Phase 12 — integrate watcher input alongside v3 rules

### Execution status — complete (2026-08-22)

Implemented in `scripts/strategy_follower.py` and regenerated into
`scripts/main.py`. Runtime activation now requires an explicit `linear_svm`
backend, successful modules, matching schema version/hash/129-feature order,
and a nonempty model export. Product outputs fall back independently; land and
labor signals remain separate; labor influence is bounded to one hand; land
influence cannot bypass configuration, affordability, reserves, ownership, or
planner capacity. Evaluation output records rule/model/final scores, selected
target adjustments, changed signals, and fallback reasons. Focused watcher and
build tests pass.

### Integration principle

The watcher is a sparse per-signal override, not an all-or-nothing replacement
for the deterministic strategy.

### Runtime loading

The editable source should attempt to load shared feature extraction and the
pure-Python model. The model is usable only if:

- backend is explicitly `"linear_svm"`;
- module import succeeds;
- schema version and hash match;
- feature order matches exactly;
- at least one requested output is present.

Otherwise the backend reports and behaves as `"rules"`.

### Product signal composition

For each product:

```text
model attractiveness = score(product increase) - score(product avoid)
```

If neither score exists, retain rule attractiveness. If only one exists, use
that accepted component while treating the missing component as zero. Validate
each returned value independently before use.

### Expansion composition

Keep land and labor model scores separate internally. Apply them to their own
decisions where possible:

- `land_expansion` adjusts whether/when land is purchased;
- `labor_expansion` adjusts the marginal daily hand target.

Do not average them into one action unless the downstream code still exposes
only one expansion control. If temporary averaging is required, document it
as a compatibility bridge and test it separately.

### Bounded influence

Initially constrain model influence:

- it may adjust one crop vacancy at a time;
- it may adjust the hand target by at most one;
- it may advance or defer one land decision, not override affordability;
- it may modify sale thresholds only within safe product-specific bounds;
- it may never violate feed reserves, deadline feasibility, land ownership,
  market-order caps, or liquidation rules.

Add broader influence only through later paired promotion rounds.

### Fallback granularity

Fallback is per output. For example, an accepted wheat-increase model may be
used while melon, labor, and land continue using rules. One malformed output
must not disable other valid model outputs, and one valid output must not mark
missing outputs as model-controlled.

### Explainability trace

In evaluation builds, record:

- rule score;
- model score;
- final bounded score;
- selected action or target change;
- whether the model actually changed the action;
- fallback reason when applicable.

Do not include verbose tracing in the Kaggle artifact.

## Phase 13 — watcher gameplay promotion

### Execution status — gate completed; SVM not promoted (2026-08-22)

The phase-10 readiness report contains zero eligible outputs, so there is no
accepted watcher report or `scripts/watcher_model.py` candidate to promote.
The required first comparison checked 5,000 replay observations with
`WATCHER_BACKEND="rules"` versus a `linear_svm` request and found zero action
differences. Per the required-comparisons rule below, the watcher is
operationally inactive and episode screening/confirmation was skipped.
`WATCHER_BACKEND` remains `"rules"`; this is a closed rejection gate, not a
claim that watcher gameplay performance passed.

### Control and candidate

- Control: fully promoted deterministic v3 with `WATCHER_BACKEND = "rules"`.
- Candidate: byte-identical strategy and constants with
  `WATCHER_BACKEND = "linear_svm"` plus accepted exported records.

### Required comparisons

1. Run action-difference traces first. If the model changes no actions, skip
   expensive performance testing and report the watcher as operationally
   inactive.
2. Screen with at least 20 episodes per side against every representative
   opponent.
3. Confirm with at least 30 episodes per opponent; expand to 60 when effects
   are noisy or small.
4. Report results separately for each accepted output if possible by enabling
   one record at a time before testing their combination.

### Promotion gates

Promote the SVM backend only when:

- offline gates passed for every exported output;
- pure-Python score parity passed;
- the model changes intended decisions in realistic states;
- paired money improves overall;
- no material competitive-opponent regression appears;
- safety and completion do not regress;
- runtime remains within Kaggle time limits;
- final built artifact matches editable behavior.

If a combination regresses, retain individually beneficial outputs and remove
the harmful interaction. Accepted offline accuracy does not guarantee gameplay
promotion.

## Phase 14 — final submission construction and verification

### Execution status — local rules artifact complete (2026-08-22)

The accepted-output set is empty, so the final local artifact correctly embeds
no watcher model. `scripts/main.py` was reproducibly generated from the
editable source, and `artifacts/final_submission_manifest.json` freezes source
and output hashes, deterministic constants, schema metadata, feature order,
readiness/dataset hashes, and the empty accepted-output list. The artifact hash
is `0e21bff58231afc7449868f923cf468e73386a809ada2f80cc4d55bc35a65d6e`.

Python 3.11 compilation, last-callable validation, forbidden-import checks,
action shape, ten-order market cap, and exact editable/artifact parity on 5,000
replay observations passed. Fifteen focused tests passed. Full-season,
both-player-position, deterministic-money, and paired-performance checks could
not run because the custom Kaggriculture runtime is unavailable; they remain
explicitly unverified. Upload/download hash comparison and Kaggle submission
metadata also remain pending because no upload was requested. Detailed evidence
is in `artifacts/phases_12_14_implementation_report.md`.

### Build

1. Freeze the promoted deterministic constants.
2. Freeze the accepted watcher report and dataset hashes.
3. Generate the pure-Python watcher module.
4. Build a self-contained `scripts/main.py`.
5. Produce a manifest containing source hashes, constants, accepted output IDs,
   schema hash, report hash, and build timestamp.

### Static verification

- Python 3.11 compilation succeeds.
- `agent` is the last top-level callable.
- no NumPy or scikit-learn imports exist.
- no local filesystem or development-path dependency exists.
- no demo, unittest, multiprocessing runner, or training code is included.
- model coefficients and feature order are embedded completely.

### Behavioral verification

- editable source and artifact action parity passes;
- rules fallback works when model data is removed or corrupted;
- action shape is correct on every tested turn;
- market orders never exceed the cap;
- full episodes complete in both player positions;
- deterministic repeated seeds return identical actions and money;
- performance matches the promoted candidate within exact paired expectations.

### Submission discipline

Upload only the hash-verified artifact. Immediately download the submitted copy
and compare its hash to the local file. Record the Kaggle submission ID and
score alongside the manifest.

## Phase 15 — close remaining validation and submission gates

### Objective

Finish the work that could not be completed in phases 4–14, establish whether
the deterministic strategy or any watcher output deserves promotion, and make
the final Kaggle submission fully reproducible. This phase is complete only
when every item below is either passed or recorded as a final evidence-based
rejection. A prepared experiment, unavailable dependency, or local smoke test
does not count as completion.

### 1. Restore the gameplay environment

Obtain the exact custom Kaggriculture runtime used by the competition and pin
its source/version or hash in the final manifest. Install the development-only
evaluation dependencies in an environment outside the submission artifact.
Before running promotion experiments:

- run the harness and editable-strategy smoke checks;
- confirm four short and two full episodes finish with valid action shapes;
- exercise both player positions;
- verify deterministic repeated seeds produce identical actions and money;
- verify requested and executed market orders, including the ten-order cap;
- verify instrumentation captures cash, revenue, spending, labor, land, crop,
  inventory, safety, and opponent-exposure metrics without changing actions.

If the competition runtime cannot be obtained, record its provenance and the
specific missing component. The performance and submission gates below then
remain open; they may not be waived based on replay-state parity alone.

### 2. Resolve the unpromoted deterministic experiments

Run the existing isolated round configurations for phases 4–8 in order:

1. gradual hiring: calendar, workload, and cash-aware workload ramps;
2. zero, one, and two compact livestock specialists;
3. melon exit date in isolation;
4. bounded strawberry replacement;
5. late wheat replacement;
6. product-specific sale batches and thresholds;
7. conditional third land, but only if the promoted layout needs more than two
   quadrants and can repay the purchase.

For each mechanism, screen with at least 20 paired episodes per side against
`starter`, `strategies_v2.ring.agent`, and
`strategies_v2.leader_clone.agent`, alternating player order and reusing
identical seeds. Confirm survivors with at least 30 episodes per opponent, or
60 for noisy/small effects. Record per-opponent means, paired differences,
standard deviation, range, approximate 95% confidence interval, win/tie/loss
counts, operational metrics, and safety failures. Promote only mechanisms that
pass the plan's existing confirmation rules; retain rejected configurations
and results as evidence.

### 3. Validate replay-family opponents

Improve or reject the phase-9 reconstructed challengers. Require held-out
action validation plus state-trajectory and gameplay validation before adding
any challenger to the representative opponent registry. Do not describe the
current low-similarity nearest-state agents as leaderboard equivalents. If no
reconstruction passes, explicitly retain the standard opponent set and record
replay families as observational diagnostics only.

### 4. Regenerate causal watcher data

Using the fully promoted deterministic v3 families, regenerate
`counterfactual-v2` rows for each intended product, land, and labor output.
Discard the obsolete v1 `competitive_expansion` data. For every output:

- reach at least 150 accepted execution-verified rows;
- include at least 40 positive and 40 negative labels;
- include at least five examples of each signed class in train, validation,
  and untouched test splits;
- include at least two usable focal families, each with both signed classes;
- keep seed groups isolated across splits and focal families;
- preserve exact schema version, schema hash, and feature order;
- stop and record the output as rejected if the 2,000-attempt cap is exhausted.

Freeze the resulting dataset files, hashes, family manifest, rejection counts,
and readiness report before training.

### 5. Train, gate, and export watcher outputs

Run the phase-11 training pipeline without changing features, thresholds, or
hyperparameters after inspecting untouched-test results. Export only outputs
that pass readiness, leave-one-family-out diagnostics, test balanced accuracy
of at least 0.60, and a margin of at least 0.10 over the majority baseline.
Verify the pure-Python export has exact score parity with scikit-learn, clips
finite results to `[-1, 1]`, returns `None` safely for missing/malformed input,
and imports no scientific package. Freeze the accepted report, report hash,
dataset hashes, schema metadata, model hash, and accepted output IDs.

If no output passes, record watcher training as a completed rejection and keep
`WATCHER_BACKEND="rules"`. An empty accepted set is a valid final outcome, but
only after fresh v2 data generation reaches either the quota or attempt-cap
decision for every intended output.

### 6. Complete watcher gameplay promotion

For each accepted output, run action-difference traces first. Reject an output
as operationally inactive if it changes no intended decision in realistic
states. Test active outputs individually before combinations, using the same
20-episode screens and 30/60-episode confirmations required above. Promote
only outputs that improve pooled paired money without a material competitive
opponent, safety, completion, or runtime regression. Remove harmful
interactions while retaining individually beneficial records. Rebuild and
re-run source/artifact parity after every accepted-set change.

If phase 5 accepts no outputs, close this step as not applicable and retain the
rules backend. Do not characterize an inactive or rejected watcher as gameplay
promoted.

### 7. Run final behavioral verification

Freeze all promoted constants and the watcher accepted set, regenerate
`scripts/main.py`, and run the complete phase-14 verification suite:

- several thousand source/artifact parity observations, including synthetic
  edge cases, seeded local states, and replay states;
- missing, corrupt, partial, malformed, and non-finite watcher fallback cases;
- action shape and market-cap checks on every tested turn;
- multiple full seasons in both player positions;
- repeated-seed determinism for actions and final money;
- paired comparison showing the artifact exactly matches the promoted source;
- runtime measurement against Kaggle limits;
- Python 3.11 compilation, last-callable, dependency, and embedded-model
  completeness checks.

Update the final implementation report and manifest with commands, sample
sizes, opponents, results, failures, hashes, and explicit pass/reject status.
The manifest's verification fields must contain machine-checkable results, not
`pending` placeholders.

### 8. Submit and verify provenance

This step requires explicit authorization to mutate Kaggle state. After that
authorization, upload only the final hash-verified `scripts/main.py`, download
the submitted copy immediately, and require byte-for-byte SHA-256 equality.
Record the submission ID, upload timestamp, local and downloaded hashes,
public/private scores when available, and relevant episode IDs in the final
manifest. Preserve the downloaded bytes in the dated submission archive.

If upload authorization is not provided, mark the local artifact ready but the
overall plan incomplete at the external-submission gate.

### Exit gate

Phase 15 and the complete v3 plan are finished only when:

- every phase-4–8 mechanism has a recorded promotion or rejection decision;
- replay challengers are either validated or explicitly excluded;
- every intended watcher output reaches a v2 quota/attempt-cap decision and an
  offline acceptance/rejection decision;
- every accepted watcher output reaches a gameplay promotion/rejection
  decision, or the empty accepted set closes watcher gameplay as not
  applicable;
- the final source and artifact pass all static, parity, full-season,
  determinism, safety, performance, and runtime checks;
- the manifest contains final hashes and no unresolved verification fields;
- after explicit submission authorization, the downloaded Kaggle artifact
  matches the local hash and its submission provenance is recorded.

## Recommended experiment order

Execute phases in this order unless evidence supplies a strong reason to stop:

1. Artifact provenance and exact `copy.py` comparison.
2. Clean v3 baseline restoration.
3. Deterministic source-to-submission build and parity testing.
4. Evaluation instrumentation.
5. Gradual hiring ramp.
6. Fixed livestock specialists.
7. Melon exit in isolation.
8. Strawberry transition in isolation.
9. Late wheat transition.
10. Product-specific sale scheduling.
11. Reconsider third land only after labor and crop economics improve.
12. Add replay-family opponent reconstructions.
13. Regenerate causal watcher data from the promoted v3 families.
14. Train, gate, and export individual watcher outputs.
15. Integrate bounded per-output watcher influence.
16. Run watcher gameplay promotion tests.
17. Build, hash, upload, download, and verify the final submission.

## Suggested initial A/B matrix

The first implementation rounds should remain deliberately small in scope:

| Round | Control | Candidate | Primary measurement |
|---|---|---|---|
| v3-00 | current validated editable strategy | clean artifact-equivalent baseline | parity and operational correctness |
| v3-01 | fixed hires | conservative calendar hire ramp | early cash and paired money |
| v3-02 | v3-01 winner | workload-aware ramp | service failures and paired money |
| v3-03 | promoted ramp | one livestock specialist | move/work ratio and money |
| v3-04 | v3-03 winner | two compact specialists | specialist efficiency and money |
| v3-05 | promoted roles | one melon exit date | realized melon price and money |
| v3-06 | best exit | bounded strawberry replacement | crop revenue and maturity losses |
| v3-07 | best crop plan | late wheat replacement | closing revenue and inventory |
| v3-08 | promoted crop plan | small frequent wheat sales | realized wheat price and money |
| v3-09 | promoted sales | product-specific hold thresholds | premium revenue and overflow |
| v3-10 | complete two-land v3 | conditional third land | land payback and money |
| v3-11 | deterministic v3 rules | one accepted SVM output | action changes and paired money |
| v3-12 | best single model | accepted model combination | interaction and regression safety |

## Definition of success

Main v3 is complete when:

1. The submitted artifact is reproducibly generated from the editable source.
2. Its behavior matches the promoted deterministic strategy exactly.
3. Gradual hiring, specialist routing, crop transitions, selling, and land are
   each supported by isolated paired evidence or explicitly left at control.
4. Evaluation includes representative reconstructed competitive families.
5. Any enabled watcher output has execution-verified multi-family data,
   untouched-test acceptance, pure-Python parity, and gameplay promotion.
6. Missing or rejected watcher outputs retain deterministic rule behavior.
7. The final uploaded and downloaded file hashes match.
8. Submission performance can be traced to a specific version, manifest, and
   set of promoted experiments rather than an unidentified `main.py` copy.
