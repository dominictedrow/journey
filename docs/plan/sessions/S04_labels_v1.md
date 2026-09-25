# S04 — Labels v1: Claude API reader + public sets + blend + calibration + folds

**Track:** L · **Machine:** Windows (or the Mac; no GPU needed) · **GPU:** no (MAE keeps running)
**Depends on:** S01 (train.csv facts) · **Unblocks:** S05, S07

## Goal
`labels_v1.parquet`: StudyInstanceUID + 12 soft targets + 12 per-label weights; `folds_v1.csv`; a label QA report.

## Read first
`02_vit_detailed_plan.md` §B · `docs/kaggle/labels_from_notebooks.md` (what worked: exact prompts/conventions and
gold AUC per label set) · `docs/research/ssl_and_labels.md` §6–§9 · the decisions of 2026-09-24 (reports first, Claude API ≤ $50,
blend with vetted public sets, never validate with leaked sets).

## Steps
1. **Setup:** `ANTHROPIC_API_KEY` in `.env` (ask Dom to add it; never print or commit it). Use the `claude-api` skill for current
   model IDs/pricing and the SDK usage (structured JSON output, batching / Message Batches for cost). Choose the model after
   the pilot (quality vs cost).
2. **Schema + prompt** (`src/knee/labels/prompt.py`): per finding `state` (present/absent/uncertain/not_mentioned),
   `severity` 0–3, `laterality_note`, `evidence_quote`. Conventions (from what measurably helped [KL]):
   - Normal counts as absent.
   - Not-mentioned stays its own state.
   - OA per compartment.
   - Degenerative oedema is not a contusion.
   - Report the effusion size.
   - Keep the original language; output English JSON.
3. **Pilot:** 100 reports (include all 58 gold + 42 random, across languages) → the cost per report, the JSON validity rate, and
   gold agreement per finding. **Report the cost to Dom before the full run**; projected total must stay ≤ ~$50 for everything.
4. **Full run:** all 4,407 reports, resumable (cache the raw JSON responses in `data_cache/labels/raw/`, gitignored).
5. **Public sets** (`src/knee/labels/public_sets.py`): download candidates (pilkwang/rsna-knee-llm-labels,
   stevenleehans v2/v4, …). **Check each license** (note it in `docs/labels.md`); **leak check**: any set with gold AUC ≈ 1.0
   or rows copying the gold labels is excluded. Known leaked: yunusgmsoy v5, barun2104.
6. **Map + blend** (`blend.py`): state/severity → probability per reader; blend in probability space; the Synovitis-from-
   Effusion fill when not mentioned; weights `w = 2|p−0.5|` (or agreement × certainty, ablation).
7. **Gold calibration** (`calibrate.py`): per finding, a monotone map or threshold (e.g. effusion ≥ moderate) fitted
   **out-of-fold** on the 58 (leave-one-out or 5-fold). No per-finding blend weights tuned on gold.
8. **Folds** (`folds.py`): 5-fold GroupKFold, groups = normalized report hash (lowercase, whitespace-collapsed) ∪ the site
   proxy from S01; gold studies spread across folds; stratified on a few labels if possible. Save `folds_v1.csv`.
9. **QA report** (`docs/labels.md`): per-finding prevalence (ours vs public vs gold), gold AUC per reader and for the blend with
   bootstrap CIs, reader-agreement matrix, the top disagreements with quotes (for Dom to skim), and the total API cost.

## Done when
- [ ] `labels_v1.parquet` + `folds_v1.csv` exist (gitignored if they contain report text; the parquet holds only numbers →
      OK to commit? **Ask Dom**: the labels are derived from MIRA-licensed data; keeping them out of the public repo is the safe default).
- [ ] `docs/labels.md` written; the cost is logged; the decision log is updated with any new conventions Dom approves.

## Review guide for Dom
Skim 10 disagreement examples; confirm the effusion/OA conventions; learning notes: soft labels vs hard labels; why
grouped folds; why the gold CI is so wide.
