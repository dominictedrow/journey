# Environment

Hand-written notes go outside the generated block. The block between the `env_check` markers is rewritten by
`python -m knee.audit.env_check` (run it via `scripts/run_s01.ps1`).

## Windows training box: **not measured yet** (S01 part 2)

Run `scripts/run_s01.ps1` on the Windows box; it fills the generated block below. Then answer by hand:
- **Cache drive:** the fastest internal drive with ≥ ~60 GB free → set `cache_root` in `configs/paths.yaml`. Ask Dom if none has room.
- **Data SSD drive letter:** → `data_root`.
- **Python env:** venv at `.venv` (per `scripts/run_s01.ps1`), Python 3.12 (matches the Kaggle image).
- **Driver:** torch 2.14.0+cu130 needs an R580+ driver. If `nvidia-smi` shows an older one, update it or use the cu128 fallback noted in `requirements.txt`.

## Mac dev box (S01 part 1, 2026-09-24): what was verified here
- Python 3.12.13 venv (uv) in the session scratchpad, torch 2.14.0 CPU, pydicom 3.0.2, pylibjpeg 2.1.0 (+libjpeg 2.4.0,
  openjpeg 2.6.0), python-gdcm 3.2.6.
- A real test-stub DICOM re-encoded to Implicit VR LE, JPEG Lossless SV1 (gdcm), JPEG 2000 lossless and RLE decodes
  **bit-exact** with every applicable plugin (pylibjpeg has no RLE plugin; pydicom's native RLE decoder handles it).
- DataLoader with 2 workers under `spawn` (same start method as Windows): ok.
- Kaggle CLI 2.2.4 works; `kaggle competitions list -s knee` shows `userHasEntered = True` (Dom has accepted the rules).
- `requirements.txt` resolves for Windows / Python 3.12 (`uv pip compile --python-platform windows`): torch 2.14.0+cu130.

<!-- env_check:begin -->
_Not generated yet: run `scripts/run_s01.ps1` on the Windows box._
<!-- env_check:end -->
