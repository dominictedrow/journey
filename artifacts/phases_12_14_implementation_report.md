# Phases 12–14 implementation report

## Phase 12 — bounded watcher integration

The editable strategy now validates watcher schema version, schema hash,
129-feature order, and a nonempty model export before using `linear_svm`.
Product increase/avoid outputs compose independently, invalid or missing
records fall back per output, and land and labor remain separate signals.

Watcher influence is bounded at the decision point: crop redirection remains
limited to one vacancy, labor can change the rule target by at most one hand,
and land can only permit or defer the next configured purchase while retaining
affordability, reserve, capacity, and ownership checks. Evaluation signals now
include rule, model, and final scores plus changed-output and fallback details.

## Phase 13 — gameplay promotion gate

`artifacts/watcher_readiness_report.json` contains no eligible outputs. There
is no accepted watcher report and no `scripts/watcher_model.py`, so the SVM
cannot enter gameplay promotion. The required action-difference precheck
compared rules with a `linear_svm` request over 5,000 checked-in replay
observations and found zero differences. The watcher is therefore classified
as operationally inactive and expensive paired gameplay testing was skipped,
as required by the plan. `WATCHER_BACKEND` remains `"rules"`.

## Phase 14 — final rules artifact

`scripts/main.py` was regenerated from `scripts/strategy_follower.py` without
a watcher model. `artifacts/final_submission_manifest.json` freezes source and
artifact hashes, promoted constants, the 129-feature schema and order, watcher
readiness and dataset hashes, and the empty accepted-output list.

Static checks passed under Python 3.11: the source, builder, tests, and artifact
compile; `agent` is the last top-level callable; the artifact has no imports,
scientific dependencies, demo, test runner, training code, or local path
dependency. Exact source/artifact parity and the action contract passed on
5,000 replay observations, including the ten-order market cap assertion.

Focused watcher/build tests passed (9 tests), as did the phase 3–5 suite (6
tests). Three broader suites could not import because this workspace lacks
`kaggle_environments` and NumPy. Consequently full-season, both-player-position,
deterministic-money, and paired-performance checks remain unavailable and are
not claimed. No Kaggle upload or download was performed because the user did
not request an external submission.
