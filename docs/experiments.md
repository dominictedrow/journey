# Experiment log

One entry per run (training, labeling, or submission). Newest at the top. A result with no entry here doesn't exist.

## Template
```
### E<NNN> — <short name> — <YYYY-MM-DD> — session S<NN>
- Track: A (pretrained) | B (scratch ViT) | L (labels) | D (data)
- Git commit: <sha>   Config: <path or inline>
- Data: cache version <vN>, labels version <vN>, folds <file>
- Hardware / wall-clock: <GPU, h>   Peak VRAM: <GB>
- Metrics: gold-58 macro AUC <x> [bootstrap 95% CI a–b]; report-label CV macro AUC <y>; per-finding table (link)
- Kaggle: public LB <z> (submission id), notebook runtime <s>
- Verdict: keep / discard / needs follow-up — <one line why>
```

## Runs

### E001 — labels v0: blend of 4 public report-label sets — 2026-09-26 — session S04 part 1
- Track: L (labels). Machine: Mac (no GPU).
- Code: `python -m knee.labels.build_labels --version v0 --no-claude` (commit of S04 part 1). Readers: pilkwang,
  steven_v1, lixin, laymond (raw); all CC0-1.0; no leak (docs/labels.md §1). Blend: silence = 0.5 vote, agreement ×
  certainty weight, fills Synovitis ← Effusion and PF OA ← Medial OA.
- Output: `<cache_root>/labels/labels_v0.parquet`, `folds_v0.csv` (5 folds 882/882/881/881/881, gold 12/11/11/12/12,
  report-hash groups, no site proxy yet).
- Metrics: gold-58 macro AUC **0.889 [0.849, 0.924]** (1,000× bootstrap). Best single reader steven_v1 0.878; best
  public non-gold-tuned reference steven_v4 0.893. Leave-one-reader-out 0.888–0.890 (no reader is measurably
  needed). Fills on vs off: 0.889 vs 0.881. Silence "vote" vs "skip": 0.889 vs 0.883 (Fracture 0.817 vs 0.779,
  Lateral OA 0.821 vs 0.793). All inside the ±0.04 CI. Per-finding table: docs/labels.md §3.
- Verdict: keep as the fallback label set for S05 until v1 (with our Claude reader) exists. "vote" became the default
  because it's the convention each public set uses on its own; the gold numbers were seen before the choice (one
  global switch, not per finding).
