# COMPETITION.md — fact sheet (✅ = from official page, ⚠️ = from public write-ups, verify in P1)

- ✅ Task: predict 2D structure (SMILES) from MS/MS spectra of natural-product-like molecules.
- ✅ Metric: MRR@25. Only the first correct guess counts (1, ½, … 1/25). No hit scores 0.
- ✅ Match rule: RDKit tautomer canonicalization (pinned 2026.03.3), then InChIKey first block (InChIKey14). Stereo and tautomer form don't matter.
- ✅ Submission: per `molecule_id`, up to 25 SMILES, best first, `;`-joined in one field. Every id appears exactly once, with a header.
- ⚠️ Code competition: Kaggle notebook, CPU or GPU, ≤9 h, **no internet**, outputs `submission.csv`. Public external data and pretrained models allowed (upload as Kaggle datasets).
- ✅ Train: `data/train.parquet`, 2,539,608 spectra, 3.03 GB, 21 row groups (ZSTD), about 275.8k structures (⚠️). **11** `ingest_lib` values: enveda-180 1,153,785 · pluskal_ms2 527,581 · riken 347,171 · gnps 220,849 · massbank 101,727 · mona 92,416 · spectraverse 50,933 · msdial 40,765 · drug_plus 2,545 · enveda-np-examples 1,184 · masaryk 652. No molecule_id, but an `inchikey14` column exists, so group by it (recompute after tautomer canonicalization for the metric).
- ✅ Train columns: ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks, base_peak_intensity, collision_energy_ev (list), collision_energy_orig, collision_energy_orig_units. `instrument_type` is messy (null, "nan", ~30 free-text values).
- ✅ Instrument match: only `enveda-180` + `enveda-np-examples` are timsTOF like test (1,154,969 spectra, equal to the train `instrument_type == timsTOF` count; ~183k structures ⚠️). The rest is Orbitrap/QTOF.
- ⚠️ Test (hidden): about 400 molecules, about 1,500 spectra (1–16 each, median 3), all Bruker timsTOF, 157–1,159 Da. Test uses 7 adducts; train has 26+.
- ✅ Local `data/test.parquet`: 1,213 spectra, 400 `molecule_id`s (matches the 400 rows of `data/sample_submission.csv`, placeholder `CCO`×25), all timsTOF, adducts [M+H]+ 959 · [M-H]- 193 · [M+CH2O2-H]- 31 · [M+Na]+ 22 · [M+NH4]+ 4 · [M+K]+ 2 · [M+Cl]- 2. Columns: molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_orig, collision_energy_ev, collision_energy_orig_units (no SMILES, no precursor_error_ppm).
- ⚠️ The local `test.parquet` is a sample of train, so it is useless for validation.
- ⚠️ Precursor error is dirty: median about 1.5 ppm, p99 about 2,900 ppm. Filter it.
- ⚠️ Three kinds of test molecule: 1) in public spectral libraries, 2) known structure (PubChem/COCONUT) with no public spectra, 3) novel, needing generation.
- ⚠️ A ±5 ppm mass window holds the true structure about 99% of the time, with a median of about 12 train candidates.
- ⚠️ Leaderboard context (mid-Oct): public notebooks cap around 0.417; rank 1 is about 0.47.
- ❓ Daily submission limit: TBD (P0.5)
- ❓ Number of final submissions selectable: TBD (P0.5)
- ✅ Dates: opened Sep 14, 2026; entry/merge deadline Dec 7; final Dec 14, 2026.
