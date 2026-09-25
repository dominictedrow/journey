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
_(none yet)_
