# Data audit (P1) — 2026-10-18

Sources: `scripts/audit.py` (streamed, 48 s, 4.8 GB peak RSS) → `logs/audit_{train,test}.log`; `scripts/test_in_train.py` → `logs/test_in_train.csv`; plots `scripts/audit_plots.py` → `reports/audit_*.png`.

## Files (P1.1)
| file | rows | row groups | codec | notes |
|---|---|---|---|---|
| train.parquet | 2,539,608 | 21 (1,184–131,072 rows) | ZSTD | 18 columns, lists are `list<double>` |
| test.parquet | 1,213 | 1 | SNAPPY | 12 columns; no structure, formula, ppm error or num_peaks |
| sample_submission.csv | 400 | — | — | `molecule_id,smiles`; `CCO`×25 joined by `;` |

## Train (P1.2)
- **Libraries (11):** enveda-180 45.4% · pluskal_ms2 20.8% · riken 13.7% · gnps 8.7% · massbank 4.0% · mona 3.6% · spectraverse 2.0% · msdial 1.6% · drug_plus 0.1% · enveda-np-examples 0.05% · masaryk 0.03%.
- **Instrument:** 83 raw values (+33,053 null). timsTOF 1,154,969 (= enveda-180 + enveda-np-examples exactly), Orbitrap 745,562, plus a long tail of QTOF/QFT/ion-trap spellings → needs a normalization map (P2).
- **Adducts:** 121 normalized values. [M+H]+ 52.2% · [M-H]- 18.3% · [2M+Na]+ 7.5% · [M+Na]+ 7.0% · [2M+H]+ 3.6% · [M+CH2O2-H]- 3.3% · [2M-H]- 2.2% · [M+NH4]+ 2.1% · [M+Cl]- 1.1% · [M+K]+ 0.75%.
- **Polarity:** positive 74.8%, negative 25.2%.
- **CE:** units eV 53.9% · NCE 31.1% · unknown 12.8% · V 2.2%. `collision_energy_ev` null for 13.3%; list length 1 (67%), 3 (15.5%), 4 (3.3%), up to 6 (stepped CE). First-value median 22.5 eV, p99 93.7.
- **Precursor m/z:** p1 137, p50 350, p99 976; garbage max 2.4e6 (filter).
- **Precursor error (ppm):** p50 1.5, p95 17.2, p99 2,884, max 4.7e8. |err| ≤5: 89.8% · ≤10: 92.9% · ≤20: 95.1% · ≤50: 95.8%. Null 0.17%.
- **Peaks/spectrum:** p50 42, p90 416, p99 1,347, max 73,318; 1-peak spectra exist (p1 = 1).
- **m/z range:** min p50 65.0 (p1 20.06), max p50 335; some spectra start near 0 m/z (junk).
- **Peaks above precursor+0.5:** 33.2% of spectra.
- **Intensities:** max-normalized (every spectrum's max = 1.0).
- **Structures:** 275,810 inchikey14 (raw column, *not* tautomer-canonical). Spectra/structure p50 6, p90 14, p99 79, max 1,468; 17,033 singletons.
- **Structures per library:** enveda-180 182,941 (180,653 unique to it) · pluskal 46,821 · gnps 45,750 · riken 15,892 · mona 11,681 · spectraverse 9,631 · massbank 9,180 · msdial 9,127 · drug_plus 2,539 · masaryk 416 · np-examples 250 (all shared).

## Local test
- All timsTOF, 7 adducts: [M+H]+ 959 · [M-H]- 193 · [M+CH2O2-H]- 31 · [M+Na]+ 22 · [M+NH4]+ 4 · [M+K]+ 2 · [M+Cl]- 2.
- CE all eV; list length 1 (915) or 3 (298); values 20–60 eV.
- Precursor m/z 245–460 (train: 65.9% of spectra fall in 240–470).
- Peaks/spectrum p50 230, p95 977, max 3,259 — **much noisier than train** (p50 42). Peak intensity p50 = 0.00017 of base peak.
- Top-k intensity share (median / p5): top64 0.987/0.890 · top128 0.996/0.946 · top256 1.000/0.979.
- Peaks kept at relative floor 0.001: median 52, p95 216; at 0.005: 25 / 92; at 0.01: 18 / 65.
- **Local test is a copy of train:** 1,213/1,213 spectra match enveda-180 train spectra exactly (precursor, n peaks, first 3 m/z); 400/400 molecules matched, 399 to a single inchikey14. Useless for validation; useful as a free end-to-end pipeline check with known answers.

## MS1 / isotope information
- No MS1 or isotope-envelope columns exist.
- But the **M+1 isotope peak (precursor + 1.00336) appears inside MS2** in 21.2% of train and 49.6% of test spectra (10 ppm / 0.01 Da tolerance); the precursor peak itself in 47.9% / 56.0%. Peaks above precursor+0.5 in 33.2% / 58.2%.
- So partial isotope evidence is available from MS2 for about half the test spectra (wide isolation window). See P1.4 gate.

## Proposals (P1.5)
- **Max peaks P = 128**, top by intensity after the floor: keeps 99.6% of intensity (median), 94.6% at p5 on test.
- **Noise floor:** drop peaks < 0.001 × base peak (test median 52 peaks left). Tune 0.0005–0.005 in a P2/P5 ablation.
- **Precursor-error cutoff:** keep |err| ≤ 20 ppm for training (95.1% of train). Use ≤ 10 ppm for building precursor-mass windows.
- **Peaks above precursor:** keep up to precursor + 3.5 Da in a separate token type (isotope evidence), drop everything above.
- **Junk filters:** precursor m/z outside 50–2,000; spectra with < 3 peaks after floor; m/z < 20.
- **Adducts to support:** the 7 test adducts first ([M+H]+, [M-H]-, [M+CH2O2-H]-, [M+Na]+, [M+NH4]+, [M+K]+, [M+Cl]-). Train on all single-M adducts with an adduct embedding; drop [2M…] and multiply charged (≈14% of train) unless an ablation says they help.
- **Instrument:** normalize to {timsTOF, Orbitrap/FT, QTOF, ion trap, QqQ, other}; test is all timsTOF.

## Plots
`audit_peaks.png`, `audit_precursor_mz.png`, `audit_ce.png` (train vs test), `audit_precursor_error.png` (train, with 5/20 ppm lines).
