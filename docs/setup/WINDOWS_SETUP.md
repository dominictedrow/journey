# Windows Setup (stub: written in Phase 12)

> This file will become step-by-step instructions for **Claude Code on Windows** to set up the whole pipeline precisely and quickly.
> The outline is in `docs/plan/MASTER_PLAN.md` → "Phase 12 plan". The open questions are in `docs/interrogation/12_windows_migration.md`.
> Until Phase 12, add any Windows-relevant gotcha found during development to the list below.

## Gotchas collected during development
- DOMS_DRIVE is exFAT (on the Mac), so the venv lives outside it (`UV_PROJECT_ENVIRONMENT`). On Windows, use `%USERPROFILE%\.venvs\quant_trader`.
- PyTorch on an RTX 5070 (sm_120) needs CUDA 12.8 or newer wheels. Use the `cu130` index and check `torch.cuda.get_arch_list()`.
- `torch.compile` on Windows needs `triton-windows`, an unofficial fork. It is optional.
