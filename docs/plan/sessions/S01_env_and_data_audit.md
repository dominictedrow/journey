# S01 — Windows environment + data audit

**Track:** D (data/infra, shared by both tracks) · **Machine:** Windows box · **GPU needed:** briefly (smoke test)
**Depends on:** nothing · **Unblocks:** S02 (preprocessing/cache), S03 (labels)

## Status after part 1 (2026-09-24, on the Mac)
Part 1 wrote and tested all of this session's code; part 2 runs it on the Windows box. Everything below still applies;
most steps are now "run the script, then read and act on its output".

| Brief step | Code | Status |
|---|---|---|
| A1–A6 environment | `python -m knee.audit.env_check` → `docs/env.md` generated block | code tested on the Mac (CPU); **run on Windows** |
| A4 requirements | `requirements.txt`, `pyproject.toml` | pinned; resolves for Windows/py3.12; **install on Windows** |
| B1–B3 counts, gold, series flags, reports | `python -m knee.audit.csv_audit` | **done** on the real CSVs → `docs/data_audit.md` §1–4 |
| B4 files, header read, truncation | `python -m knee.audit.header_scan` (resumable shards) | tested on the test stub; **run on train** |
| B5 tags | same scan (every file, not a sample) → `docs/dicom_tags.md` | test stub done (`docs/dicom_tags_test.md`) |
| B6 syntaxes + decode speed | `python -m knee.audit.decode_bench` | decoders verified bit-exact on the Mac; **throughput on Windows** |
| B7–B9 geometry, laterality, site proxy | `python -m knee.audit.series_summary` | tested on the test stub; **run on train** |
| Reports | `python -m knee.audit.write_reports` | regenerates `docs/data_audit*.md`, `docs/dicom_tags*.md` |

**Part 2 checklist:**
1. `git pull`; set up the venv (lines at the top of `scripts/run_s01.ps1`); copy `configs/paths.example.yaml` →
   `configs/paths.yaml` and fill in `data_root` (the SSD), `cache_root` and `audit_out` (internal drive).
2. `powershell -ExecutionPolicy Bypass -File scripts\run_s01.ps1`. The header scan is the long step (estimate 30–60 min;
   UNVERIFIED). It resumes if interrupted.
3. Read the regenerated docs and act on them:
   - CPU cores → CLAUDE.md open question #1.
   - Tag list → open question #3 and checklist items 2, 4, 5, 6, 12.
   - Corrupt/truncated files: the forum said 2.
   - Laterality: geometry-vs-tag agreement by |centre x| band. Pick a dead zone and write the rule into S02's brief.
   - Contrast class × CSV flag: confirm the flag means fat suppression.
   - Cold decode throughput vs the SSD's MB/s: is S02 disk-bound or CPU-bound?
   - Cache drive.
4. Update CLAUDE.md, HANDOFF, the master plan, and the S02 brief if reality differs. Commit the regenerated docs
   (aggregates only; the raw parquet stays in `audit_out`).

## Goal
A working, reproducible Python/CUDA environment on the Windows box, plus a written, measured audit
of the raw data that replaces guesses in CLAUDE.md with facts.

## Read first
`HANDOFF.md` → `docs/plan/00_MASTER_PLAN.md` → this file → `docs/kaggle/data_and_dicom_from_notebooks.md`
(what public notebooks already found about tags, transfer syntaxes, and stats: **verify, don't copy**).

## Steps

### A. Environment
1. Record the hardware in `docs/env.md`: exact CPU model plus **physical cores vs logical processors**
   (`wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors`, or Task Manager). This closes CLAUDE.md open question #1.
   Also record the RAM, the GPU driver version (`nvidia-smi`), the internal drives with free space and type (NVMe/SATA), and the data SSD's drive letter.
2. Pick the **cache drive**: the fastest internal drive with ≥ ~60 GB free (the public full-corpus uint8 cache was ~11–22 GB;
   ours may be larger if we keep more slices). Ask Dom if no internal drive has room.
3. Python env (venv or conda; record which in `docs/env.md`). **The RTX 5070 is Blackwell (sm_120)**: it needs a PyTorch
   build with CUDA ≥ 12.8 support (`UNVERIFIED`: confirm on pytorch.org "Get Started" for the current stable release).
   Verify with `torch.cuda.get_device_capability()` and a bf16 matmul.
4. Install and pin in `requirements.txt`: torch/torchvision, numpy, pandas, pyarrow, scikit-learn, pydicom, pylibjpeg,
   pylibjpeg-libjpeg, pylibjpeg-openjpeg, python-gdcm, opencv-python-headless, timm (Track A only; Track B must not import
   timm model classes), tqdm, matplotlib, kaggle.
5. Windows-specific checks (log results in `docs/env.md`):
   - DataLoader with `num_workers>0` uses *spawn* on Windows: the training entry points need `if __name__ == "__main__":`
     and picklable datasets. Run a 2-worker smoke test.
   - Does `F.scaled_dot_product_attention` use the flash or mem-efficient backend on this Windows build? Test with
     `torch.nn.attention.sdpa_kernel` forcing each backend (`UNVERIFIED` whether flash kernels ship in Windows wheels).
   - Is `torch.compile` usable (Triton on Windows)? `UNVERIFIED`. Record yes/no; nothing may depend on it.
6. Kaggle CLI works on Windows (`kaggle competitions list -s knee`).

### B. Data audit (write `docs/data_audit.md` and `docs/dicom_tags.md`)
Write the scripts in `src/knee/audit/` so they can be re-run. Nothing here loads full volumes into RAM.
1. **Counts** from `train.csv`, `train_series.csv`, `test*.csv`: studies, series/study, labeled vs unlabeled studies
   (confirm **4,407 / 58**), report languages (rough, e.g. a langdetect-style guess on 500 reports), report length.
2. **Prevalence** of each of the 12 labels in the 58 gold studies (count positives, since AUC noise depends on them).
3. **Series metadata:** crosstab of plane × Fluid_Sensitive × Fat_Suppression; the fraction of studies missing each plane.
4. **Files:** slices per series (distribution), total files vs the 819,640 on the Data page, any non-.dcm files, and a
   read test of every file header (`stop_before_pixels=True`) → list the failures. The forum reported 2 corrupt training
   files: find and list them in `docs/data_audit.md`.
5. **Tags:** from a stratified sample (≥ 1 series per plane/sequence combo per manufacturer, ~500 files), dump **all
   tags present** → `docs/dicom_tags.md` with, for each tag: present-rate, example values, and "useful for?". Explicitly
   answer: ImageOrientationPatient, ImagePositionPatient, PixelSpacing, SliceThickness, SpacingBetweenSlices,
   RescaleSlope/Intercept, PhotometricInterpretation, BitsStored, Manufacturer, MagneticFieldStrength, SeriesDescription,
   PatientSex, PatientAge, Laterality/BodyPartExamined, overlay groups (60xx). This closes CLAUDE.md open question #3
   and checklist items 2, 4, 5, 6, 12.
6. **Transfer syntaxes:** distribution over the sample plus **decode speed** per syntax (ms/slice, single process), and
   the throughput with N processes = physical cores. Compare pydicom+pylibjpeg vs gdcm for JPEG2000/JPEG-Lossless. This
   decides whether decoding is the training bottleneck and feeds the efficiency plan.
7. **Geometry:** distributions of pixel spacing, slice thickness, rows×cols, and the in-plane field of view (mm) per plane,
   plus whether IOP shows the rows/cols axes mapping consistently to anatomical axes (needed for the flip rules in 02 §C7).
8. **Laterality:** can left vs right knee be told from the tags (IOP/IPP sign conventions, Laterality tag)? Report the method
   and the L/R split, or `UNVERIFIED` if not determinable.
9. **Site proxy:** Manufacturer × MagneticFieldStrength × (other scanner tags) → a `site_proxy` id per study. Used for
   grouped CV in S02/S03.

## Done when
- [ ] `docs/env.md` complete; the smoke test passes (GPU bf16 matmul, 2-worker DataLoader, pydicom decodes all 4 syntaxes).
- [ ] `requirements.txt` pinned and committed.
- [ ] `docs/data_audit.md` and `docs/dicom_tags.md` written with numbers; CLAUDE.md open questions #1–#3 answered
      (edit CLAUDE.md's Data/Compute/Open-questions sections: replace the UNVERIFIED items with verified facts and cite the file).
- [ ] Any surprise that contradicts `docs/plan/*` is raised with Dom and logged in `docs/decisions.md`.

## Write back
HANDOFF §4/§5, the master-plan checklist, `docs/sessions_log/S01_<date>.md` with the Review guide for Dom (learning notes:
DICOM transfer syntaxes; why IOP/IPP ordering beats filename ordering; what pixel spacing means for the patch size).
