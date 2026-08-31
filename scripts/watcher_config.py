"""Single source of truth for watcher data quotas and acceptance thresholds.

Imported by train_watcher.py, generate_until_quota.py and watcher_readiness.py
so the generator, the readiness audit and the trainer can never disagree on
what "ready" means. Every threshold the toolchain shares lives here and nowhere
else.
"""

from __future__ import annotations

# ---- Corpus quotas (a modeled output needs all of these to be trainable) ----

MIN_ACCEPTED = 150          # execution-verified accepted rows for the output
MIN_SIGNED = 40             # accepted rows per signed class (label -1 and +1)
MIN_PER_CLASS = 5           # per-split signed rows of each class; also the
                            # leave-one-family-out usability floor
MIN_FAMILIES = 2            # distinct usable focal families in train u val

# A --min-per-class below this is a smoke-test-only configuration: balanced
# accuracy computed from one or two samples per class is meaningless and the
# trainer emits a loud warning when it sees it.
SMOKE_TEST_MIN_PER_CLASS = MIN_PER_CLASS

# ---- Phase 1 statistical gate (a skillful model passes, a lucky one does not) ----

# Grouped k-fold CV (group = seed group) picks C by mean balanced accuracy on
# train u val; the test split stays untouched until the single gate evaluation.
CV_FOLDS = 5
ACCEPT_MIN_CV_MEAN = 0.62          # Phase 4 export floor for the CV mean

# The headline test metric is a bootstrap CI on balanced accuracy, not a point
# estimate: resample the test signed rows and gate on the lower percentile so a
# 10-per-class "pass" is honest.
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CI_LOWER_PERCENTILE = 5
BOOTSTRAP_CI_UPPER_PERCENTILE = 95
BOOTSTRAP_SEED = 0
ACCEPT_MIN_BALANCED_ACC_CI_LOWER = 0.58   # gate on the 5th-percentile test bacc

# Pooled leave-one-family-out: refit per held family, pool every held-out
# prediction, gate one balanced accuracy over the pool. Individual families
# that lack both signs contribute rows without each needing to clear the bar.
ACCEPT_MIN_LOFO_POOLED = 0.55
LOFO_MIN_POOLED_ROWS = 20                 # too few pooled rows -> not evaluable

# The confirm split is opened once, after a candidate is otherwise finalized,
# and never feeds back into any decision. If confirm and test balanced accuracy
# disagree by more than the test CI width, the candidate is not exported.
REQUIRE_CONFIRM_SPLIT = True

SPLITS = ("train", "val", "test", "confirm")
