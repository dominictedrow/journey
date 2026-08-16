| Family | Tuning | Best val acc | Delta (pp) | Epoch 15 val acc | Rose? |
|---|---|---:|---:|---:|:---:|
| control | production defaults + fixed seed | 59.65% | +0.00 | 59.65% | no |
| dropout | 0.2 | 60.45% | +0.80 | 60.45% | yes |
| dropout | 0.3 | 60.26% | +0.60 | 60.11% | yes |
| dropout | 0.4 | 59.89% | +0.23 | 59.01% | yes |
| hidden | 256 | 60.22% | +0.57 | 59.71% | yes |
| hidden | 512 | 59.15% | -0.50 | 58.68% | no |
| depth | 2 | 60.69% | +1.04 | 60.45% | yes |
| depth | 4 | 57.43% | -2.22 | 56.94% | no |
| lr | 1e-4 | 60.62% | +0.97 | 59.83% | yes |
| lr | 3e-4 | 60.85% | +1.20 | 59.06% | yes |
| lr | 1e-3 | 56.50% | -3.15 | 54.55% | no |
| optimizer | adamw | 58.68% | -0.98 | 58.68% | no |
| weight_decay | 1e-6 | 58.62% | -1.03 | 58.62% | no |
| weight_decay | 1e-5 | 59.82% | +0.17 | 59.82% | yes |
| weight_decay | 1e-3 | 53.48% | -6.17 | 51.97% | no |
| batch_size | 32 | 59.94% | +0.29 | 59.52% | yes |
| batch_size | 64 | 61.29% | +1.64 | 60.98% | yes |
| batch_size | 128 | 60.78% | +1.13 | 60.76% | yes |
| sequence_length | 40 | 58.84% | -0.81 | 58.57% | no |
| sequence_length | 80 | 58.72% | -0.93 | 58.38% | no |
| sequence_length | 100 | 59.13% | -0.52 | 59.13% | no |
| jitter | 0 | 59.33% | -0.32 | 58.97% | no |
| jitter | 0.01 | 60.05% | +0.40 | 59.47% | yes |
| jitter | 0.04 | 58.35% | -1.30 | 58.35% | no |
| temporal_keep | 0.5 | 59.74% | +0.09 | 59.50% | yes |
| temporal_keep | 0.85 | 59.68% | +0.03 | 58.72% | yes |
| temporal_keep | 1.0 | 59.28% | -0.37 | 58.79% | no |
