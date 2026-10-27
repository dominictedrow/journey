# HANDOFF P7 S21 (2026-10-26)

## What happened
- Pushed 015da35 + 36a8f5c to origin (Dom OK).
- D-045: Dom chose strategic option (b): rebase on the public 0.467 notebook
  (alexchilton/casmi26-lb-0-467-one-knob-rrf-offset-3-to-1) and add our validated pieces as extra signals.
- Licence check DONE, verdict: **cleared for prize eligibility** (rules read 2026-10-26 from the competition Rules tab):
  - Winner licence = MIT for our code. Rules 2.5.a: "input data or pretrained models with an incompatible license"
    need no open-source grant, so NC data/models (CC-BY-NC-SA v4g assets, competition-data-derived models) are fine.
  - Foundational 3.6.c: open-source *code* we use must be OSI + commercial-OK. Code inside the datasets is:
    ms-pred/ICEBERG/GLACIER MIT, DreaMS MIT, RDKit BSD-3, Lightning Apache 2.0, einops MIT, v4g notebook Apache 2.0
    (per v4g SOURCE-LICENSES.md). Competitor code shared publicly on Kaggle (sanpier, dmitriigluzdov scripts) is
    deemed OSI commercial-OK licensed by Foundational 3.6.b.
  - External data: all 7 are public Kaggle datasets at no cost -> 2.6.a satisfied.
  - Winner obligations would need us to list each third-party model/dataset and how to obtain it.

## The 0.467 notebook (first look)
- Inputs (7): sanpier/casmi-spectrum-model-v1, sanpier/catalogue-experiment-saner, dmitriigluzdov/{casmi26-fold-safe-fpnet,
  casmi26-natural-product-knowledge-table, casmi26-natural-product-panel-benchmark, casmi26-pubchem-popularity-prior,
  casmi26-attributed-v4g-reproduction-assets}. GPU T4, Python 3.12 image (asserted).
- code.py is 51 lines / 509 KB: modules embedded as strings; runner.py hash-checks and executes a pinned baseline
  notebook (cells from a JSON file), then exports candidates; final knobs NEURAL_WEIGHT=0.25, RRF_OFFSET=1.0.
- Local copy: data/kaggle/public/casmi26-lb-0-467-one-knob-rrf-offset-3-to-1/ (read only, never execute locally).

## Next session starts with
0. DONE in S21: step 1 below, see reports/P7.5-rebase-map.md. Re-ask the pending gate (fork + reproduce).
1. Map the 0.467 pipeline: extract the embedded modules + the pinned baseline notebook into
   data/kaggle/public/.../extracted/ (read only), write reports/P7.5-rebase-map.md: stages, candidate lists, scores,
   fusion, where a per-candidate extra signal can be added (our contrastive cos + fragment expl).
2. Reproduce: push it as our own kernel (copy, Apache 2.0) and confirm ~0.467. 🛑 gate 7 (submission).
3. Then add our signals one at a time, each a gated submission.

## Gotchas
- gstack browse binary is not built on this machine; use the claude-in-chrome tools for Kaggle pages (rules page needs JS).
- `kaggle datasets metadata` shows licences under info.licenses.
