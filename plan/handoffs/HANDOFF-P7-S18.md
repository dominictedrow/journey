# HANDOFF P7-S18 — 2026-10-25

## Done this session
- Diagnosed V-np vs LB gap for #6 (V-np +0.13 / +0.15 vs LB +0.016).
- `scripts/eval_encoder.py`: `--pop LAM` (+ LAM x popularity) and `--src MASK` (keep src & MASK) applied BEFORE the
  top-K cut, mirroring casmi.kaggle.predict (P6_POP, P6_SRC); output tag `_pop<LAM>_src<MASK>`.
- `scripts/pop_bias.py <splits>`: truth's popularity percentile in its window (all / curated rows); writes
  `<valcand>/<split>_pop_pct.npy`. `scripts/np_probe.py` now also prints `low-pop curated` MRR (pct < 0.9, D-038).
- PLAN.md: P7.4a NP validation line (rerouted D-035, D-038).

## Results / numbers
- Kaggle-order eval (np_p6_ft, v_np, top 100, src 7): lam 0 0.640 / 0.02 0.768 / 0.05 0.790 / 0.1 0.784.
  So the top-200 rerank was NOT the cause of the gap.
- Truth popularity percentile among curated window rows (median / share where truth is most popular):
  V-np 0.99 / 46%; V-pool 0.30 / 1.5%; V-lib 0.14 / 6.6%. Popularity is near-oracle only on V-np.
- Low-pop V-np (42 mols), curated rows: lam 0 0.709, 0.01 0.710, 0.02 0.706, 0.05 0.648, 0.1 0.561, 0.2 0.419.
  Full V-np: 0.635, 0.719, 0.758, 0.765, 0.753, 0.704.
- Reading: the hidden test is a mix; lam 0.05 (#6) costs obscure molecules ~0.06. lam 0.01-0.02 is ~free on low-pop
  and keeps most of the high-pop gain. Low-pop n = 42, so ±0.05 noise.

## Decisions & approvals
- D-038 (gate 2): de-biased low-pop V-np metric, reported next to full V-np. Logged.

## Traps discovered
- Any val built from library-measured molecules is popularity-biased (measured standards are well known).
- Low-pop V-np base MRR (0.709) is higher than full V-np (0.635) at lam 0: popular molecules have more same-mass popular look-alikes.

## Background jobs
- none

## Next session starts with
- Optional gate 6 (ask Dom): Submission #7 = #6 with P6_POP 0.02 (expected LB about +0.00-0.01; mostly a check of
  the low-pop reading). Code change is one constant + bundle code version (gate 7).
- Otherwise P7.2/P7.3: fragment features on the src-7 top-100 (eval_encoder --top 100 ... --src 7 --pop 0.02), then
  LambdaRank with popularity as a feature, judged on low-pop V-np + full V-np; P7.3 still needs a separate training fold (gate 2).

## Open questions for Dom
- Spend a submission on lam 0.02 (#7), or go straight to the reranker?
