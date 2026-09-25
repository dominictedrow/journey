# Labels: what the public notebooks and label datasets teach us

Compiled 2026-09-24 from public Kaggle notebooks and datasets for `rsna-knee-abnormality-detection`.
Every claim cites its source. Two kinds of number appear here:

- **[nb:<owner/slug>]** / **[ds:<owner/slug>]**: stated by that notebook or dataset. Pulled notebooks have **no saved cell outputs** (the CLI strips them), so a number only counts if it was written into markdown, a code comment or a constant.
- **[measured]**: computed by us on 2026-09-24 from the public label CSVs plus a copy of `train.csv` (see the note below). The script is `scratchpad/analyze.py`, outside the repo.

> **Data-handling note:** `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels/train_folds_with_pseudo.csv` is a full copy of the competition `train.csv` (4,407 rows with `Report`, 58 gold rows) plus folds. It was downloaded while surveying label datasets. It was used only in the session scratchpad and **not copied into the repo** (MIRA license). The 58 gold rows in it match `riadmohamed42/jev-knee-labels/gold74_eval_set.csv` (`source == original_gold58`) exactly. Once Dom has the real `train.csv`, re-run the numbers against it.

---

## 0. The quick version

1. **Everyone uses LLM-read report labels.** The best public single label sets score **about 0.87–0.89 macro AUC against the 58 gold studies**. Blends of label sets reach about **0.89–0.90**. Regex/lexicon labels score about **0.80–0.81** ([ds:stevenleehans/rsna-knee-llm-report-labels], [nb:wguesdon/rsna-knee-dinov2-at-meniscus-resolution]).
2. **The gold 58 were not drawn at random.** Every gold study has at least 1 positive finding (mean 4.14, min 1, max 9) [measured; also ds:stevenleehans, nb:karnakbaevarthur/rsna-knee-eda-to-2-5d]. Gold prevalence is far above the report-derived prevalence for Fracture (31% vs ~7%), Synovitis (47% vs ~13%), Lateral Meniscus (40% vs 15–25%) and ACL (41% vs 20–27%) [measured].
3. **Reports and gold disagree in a systematic way.** The annotator seems to have read the images and to use a stricter cut-off than the reporting radiologist for Effusion and Contusion. The annotator also marks Synovitis that most reports never mention [measured, nb:prvsiyan, nb:sadamtorres, ds:riadmohamed42/jev-knee-labels]. So report labels have a ceiling somewhere around 0.88–0.92 against gold.
4. **58 studies are a very noisy ruler.** The 95% CI on macro AUC is about ±0.04 wide [measured: steven_v2 0.887, CI 0.851–0.917], and a single finding's CI can span 0.3. Differences under about 0.02 macro cannot be measured [ds:stevenleehans]. Tuning on 58 has hurt people on the leaderboard (see §3).
5. **An image model trained on the 58 gold studies alone is weak.** It scored **0.66 macro AUC** (frozen RadImageNet ResNet-50 + head, 5-fold, [nb:dariushafshar/rsna-report-pseudo-labels-add-0-022-auc-4-5]). That is much weaker than any report-label source (0.83–0.90). This matters directly for Dom's teacher plan (§7).

---

## 1. How report→label extraction was done

### 1a. Rule/lexicon extractors

**pilkwang baseline lexicon** [nb:pilkwang/rsna-knee-baseline-v1] and its extended **v2** [nb:prvsiyan/rsna-knee-read-the-report-then-the-knee]:
- **No language routing.** Each cue list holds all 9 languages at once, and every clause is tested against the union. Routing first fails because, for example, `la` is common in both Spanish and French.
- **Pipeline order is normalize → unwrap → segment → scope.**
  - NFKD normalization plus fold-downs for `ı ß đ ø æ`.
  - Many Greek reports write mu as MICRO SIGN U+00B5, which NFKD maps to U+03BC. [nb:ulasdesouza/01-report-labels] counts 305 of 321 Greek reports doing this.
  - Hard-wrapped lines are rejoined when a line does not end in punctuation.
  - A `Header:` line is attached to the value on the next line (`Fractures :` / `Aucune.`).
  - ulasdesouza instead does not split on `:` at all.
- **Per-(finding, clause) polarity is one of assert, negate or hedge.**
  - Negation has a direction. Pre-posed negators (`no, sin, sans, ohne, geen, bez, χωρις, без`) govern the text after them within 90 characters, and a contrast word ("but", "pero", ...) ends their scope.
  - Post-posed negators (Turkish `yok, izlenmedi, saptanmadi`, Croatian `nema znakova`) govern the text before them.
  - The negation test is anchored on the pathology word, not the anatomy word ("Medial meniscus is **not torn**").
  - A negator followed by an abnormality noun (`sin alteraciones`, `ohne Auffälligkeiten`, `geen afwijkingen`, `bez osobitosti`, `χωρίς αλλοιώσεις`) is read as explicit normality, not as silence. This one fix cut mean silence over 4,407×12 cells from 49.97% to 48.09% (Greek −15 points on the ligament and meniscus targets). The gold agreement moved −0.0007 (CI −0.0035 to +0.0009), which is to say the 58 studies could not see the fix [nb:prvsiyan].
- **Graded, not binary.** Severity words (trace/mild → low, marked/large → high) change the score.
- **Numeric grades are read per structure.** Meniscus grade ≥3 counts as a tear (0.95) and grades 1–2 as degeneration (0.30). Ligament grade ≥2 counts as a partial tear (0.85).
- **Unmentioned is not negative.**
  - Score/confidence pairs: positive ≈ 0.5 + 0.45×severity (max 0.97), negative ≈ 0.04–0.20, silent = **0.28 with confidence 0.05** [nb:prvsiyan `_grade`].
  - Confidence becomes the sample weight: `W = 0.25 + 0.75·conf`, and gold rows get `W = 3.0` [nb:pilkwang].
- **OA inheritance.** Most reports write "tricompartmental chondrosis" or "gonarthrose" without naming a compartment. An unlocalized cartilage statement counts for all three OA targets at a discount unless a compartment was explicitly cleared. This was the one rule big enough to show up on gold agreement [nb:prvsiyan §2.2].
- **Synovitis backoff.** "88% of reports never write the word, and the annotator marks it on nearly half the studies" [nb:prvsiyan]. When the report is silent on synovitis, the score is set from Effusion plus inflammatory proxies (Hoffa's fat pad, distended bursa) at confidence 0.18.
- **Decoys.** `microfracture` is not a Fracture. A meniscal or parameniscal cyst is not a Baker's cyst. Degenerative marrow oedema is discounted for Contusion (×0.45) and trauma context is boosted (×1.35).
- **Two ways to check a lexicon** [nb:prvsiyan §2.2, nb:pilkwang]:
  - Gauge 1 is agreement with the 58 gold studies. It is too noisy to tune on.
  - Gauge 2 is the "silence rate" (did any rule fire?) on all 4,407 reports, split by language. It needs no labels, so it is the tool for finding vocabulary gaps.
  - Silence is either "never mentioned" (correct) or "mentioned but missed" (a bug). They are told apart by checking whether the anatomy word appears at all.

Bugs other people found that we should avoid [nb:ulasdesouza/01-report-labels]:
- Without `ı→i`, Turkish `sıvı` never matched, and Effusion fired on **4% of Turkish reports instead of ~40%**.
- Without `ß→ss`, **Lateral Meniscus was 0.00 across all 261 German reports** (`Außenmeniskus`).
- "medial patellar facet cartilage loss" leaks into Medial OA unless PF-only clauses are excluded.

Measured regex quality against gold: **0.8136** macro AUC [ds:stevenleehans]. Another rule extractor scored agreement 0.7960 against 0.8233 for Claude Opus 5, recall on English 0.861 vs 0.674 on other languages, paired delta +0.027 (CI +0.003 to +0.053, McNemar p = 0.020) [nb:wguesdon]. **The whole regex-vs-LLM gap is recall on non-English reports.**

### 1b. LLM extractors (public label datasets)

| Dataset | LLM / method | Output design | Stated gold-58 macro AUC | Our re-measure [measured] |
|---|---|---|---|---|
| `pilkwang/rsna-knee-llm-labels` `report_labels_v2.csv` | Hosted Anthropic model. `api_labeler.py` defaults to `claude-opus-5`, `effort="low"`. [nb:wguesdon] says "Claude Opus 5 at low effort". The yunus card calls it "GPT-4", which contradicts the code. | Structured JSON schema: 12 × (verdict YES/NO/UNK, severity 0–3), mapped to score + conf. Uncertain counts as UNK. The instructions sit in a cached system prompt, and runs used the Batch API. | 0.870 (per yunus card) | **0.866** (1 gold study missing from the CSV: 4,406 rows) |
| `stevenleehans/rsna-knee-llm-report-labels` v1 `llm_labels_full.csv` | LLM, model **UNVERIFIED** (not stated) | One probability per finding. **0.5 = "report does not address it"** (25.4% of cells). | 0.8780 | 0.878 |
| … v2 `llm_labels_v2.csv` | v1 + Synovitis cells that the report leaves undecided are filled from the Effusion field | same | 0.8873 | 0.887 (CI 0.851–0.917) |
| … v4 `llm_labels_v4_blend.csv` | v2 blended with lixin73, in **probability space** | same | 0.8927 | 0.893 |
| `lixin73/rsna-knee-llm-report-labels-sol56` | "GPT-5.6-Sol" per title (**UNVERIFIED**) | probabilities | 0.8347 | 0.835 |
| `laymond/…-qwen3-8b-weak-labels` `weak_labels_v2_filled.csv` | Qwen2.5-32B-Instruct 4-bit, local, about 7 h on an RTX 4090 at batch 4 | label / mentioned / negated / probability. The prompt asks for a "granular" probability, plus co-occurrence fills. | 0.8686 (raw 0.8521) | 0.869 |
| `riadmohamed42/jev-knee-labels` HYBRID | Per finding, the better of steven v4 / pilkwang ("hard swap"), then 0.95 × that + 0.05 × a 6-member OpenJev ensemble | probabilities | 0.9007 on 58, 0.9212 on 74 (58 gold + 16 the author labelled by hand from reports) | 0.902 on 58 (**tuned on these same 58 studies, so optimistic**) |
| `yunusgmsoy/rsna-knee-llm-labels-4-source-merged` v5 | Gemini Flash plus a merge of 4 sources | 0.00 = negative, 0.05 = unmentioned, 0.25 = mild, 0.55 = moderate, 0.9–1.0 = severe | 0.8798 (Gemini alone) | **1.000 on all 12**: the file has the gold labels pasted into the 58 rows. **Using it for validation leaks.** |
| `barun2104/…-llm-soft-labels` | not described | `pseudo_*` columns | – | 1.000 on gold (gold pasted in). Do not use. |
| [nb:sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee] | Qwen3-14B-AWQ with vLLM and a fixed JSON schema, choosing from a **closed vocabulary** (`complete_tear`, `degeneration_no_tear`, `absent_explicit`, grades). Python code then maps the choices to probabilities (e.g. "mild effusion is empirically positive ~45% of the time → 0.45"). | soft plus a confidence weight | 0.881 | not public |
| [nb:prvsiyan §9] quoting yashbishnoi98 `.903` system | Local Qwen3.6-35B, temperature 0, reasoning off, fixed 12-key JSON, prompt revisions measured on gold (mean per-class accuracy 0.833, F1 0.766). Fused per finding with pilkwang: where the two agree the label stays hard, and where they disagree it becomes a soft target weighted by each reader's per-finding accuracy. | soft | not stated | private |
| [nb:hosseini7798/rsna-knee-mri-labeling-typesafe-jev] | OpenJev single-report extraction | – | 0.8544 | – |

Prompt and design lessons that were measured:
- **Explicit UNK vs NO is essential.** "A structure described as normal or intact is NO, not UNK. A finding the report never names is UNK, not NO. Medial and lateral are distinct." [ds:pilkwang api_labeler.py SYSTEM prompt]
- **Ask for categories, not probabilities.** Have the LLM pick discrete states and map them to probabilities in code, "because asking a model for calibrated numbers directly is asking it to do something it is bad at" [nb:sadamtorres]. laymond does the opposite and gets 0.869, so both approaches work. **UNVERIFIED which is better.**
- **Label conventions matter more than vocabulary** [ds:riadmohamed42/jev-knee-labels README]. Their conventions: Effusion counts from moderate up (small/mild = 0); "incipient OA of all three compartments" = 1 for Medial and Lateral OA; traumatic bone-marrow oedema = Contusion but degenerative oedema does not; Hoffa fat-pad impingement = Synovitis; suspected/rule-out = 0.
- **Effusion over-calling.** laymond's v1 prompt predicted 0.88 effusion prevalence vs 0.60 in gold (specificity 30%). The v2 prompt counts only effusions "worth reporting (moderate/large, distending the suprapatellar recess)" and raised specificity to 43.5% [nb:laymond/rsna-knee-mri-weak-labeling-with-qwen2-5-32b].
- **Synovitis under-calling.** v1 sensitivity was 48%. v2 explicitly counts Hoffa's fat pad / infrapatellar oedema, plica, and synovial thickening, and reaches 63% [nb:laymond].
- **The Effusion field predicts gold Synovitis better than the Synovitis field does:** 0.7115 vs 0.6780. P(synovitis | effusion) = 0.63 vs 0.22 without. Filling only the undecided Synovitis cells from Effusion moved Synovitis 0.678 → 0.790 [ds:stevenleehans].
- **Co-occurrence fills on silent cells only** [nb:laymond]: PF OA ← Medial OA (0.60), Synovitis ← Medial OA (0.60), Synovitis ← Effusion (0.55). This raised PF OA AUC 0.728 → 0.840 and Synovitis 0.703 → 0.789.
- **Prompt diversity beats sampling.** A 3-prompt average scored 0.8693 vs 0.8486 for `samples=8` with one prompt. Multi-read "denoising" (`steps=4`) was the strongest single lever for fixing explicit negatives still scored 0.05–0.15. JEV seemed to attend to only about the first 1,200 characters of instructions, so rewriting the rules compactly at the front moved Lateral OA 0.65 → 0.77 [ds:riadmohamed42 README]. That last point is specific to that model.
- **The least-correlated label source adds the most.** lixin73 is the weakest alone (0.835) but has the lowest rank correlation with steven (0.718), and blending it in gave the biggest gain (0.887 → 0.893). "Standalone quality is not leg value; decorrelation is." Blend in probability space, not rank space, or the `2·|p−0.5|` confidence weight stops meaning anything [ds:stevenleehans].
- **Consensus weighting** [nb:prvsiyan `build_public_consensus`]:
  - target = mean of pilkwang, steven and lixin
  - agreement = 1 − 2·mean|source − consensus|
  - certainty = 2·|consensus − 0.5|
  - weight = 0.20 + 0.80·(0.65·agreement + 0.35·certainty)
  - gold rows are overwritten with gold at weight 3.0 (2.0 in later versions, which also oversample gold).
- **Masked agreement** [nb:dariushafshar]: train only on (study, label) pairs where all 3 sources agree after binarizing, and mask the rest out of the loss. 2,570 studies have at least 8 of 12 labels unanimous.
- **Calibrate to the gold annotators' policy** [nb:blacklions/report-teacher-anatomy-aware-hierarchical-multimod]: a small logistic model per target maps pseudo-label probability onto gold, because "medical report finding ≠ competition positive label". Also adds a pairwise ranking loss on top of weighted soft BCE. **Its measured benefit is UNVERIFIED.**

### 1c. Per-finding agreement with the 58 gold studies (AUC) [measured]

| Finding | gold pos/58 | pilkwang | steven v2 | steven v4 | lixin | laymond Qwen32B | JEV hybrid* | Consensus of pilk+steven v2+lixin |
|---|---|---|---|---|---|---|---|---|
| ACL | 24 | 0.990 | 0.993 | 0.987 | 0.931 | 0.981 | 0.992 | – |
| MCL | 9 | 0.976 | 0.964 | 0.968 | 0.921 | 0.971 | 0.982 | – |
| Medial Meniscus | 26 | 0.943 | 0.954 | 0.948 | 0.890 | 0.897 | 0.947 | – |
| Lateral Meniscus | 23 | 0.841 | 0.862 | 0.879 | 0.866 | 0.860 | 0.872 | – |
| Medial OA | 15 | 0.909 | 0.931 | 0.932 | 0.891 | 0.916 | 0.930 | – |
| Lateral OA | 11 | 0.789 | 0.830 | 0.833 | 0.824 | 0.795 | 0.855 | – |
| PF OA | 21 | 0.879 | 0.901 | 0.902 | 0.808 | 0.840 | 0.902 | – |
| Effusion | 35 | 0.824 | 0.853 | 0.877 | 0.696 | 0.849 | 0.877 | – |
| Synovitis | 27 | 0.695 | 0.790 | 0.790 | 0.676 | 0.789 | 0.785 | – |
| Baker's | 12 | 0.933 | 0.946 | 0.944 | 0.903 | 0.903 | 0.920 | – |
| Contusion | 19 | 0.758 | 0.831 | 0.860 | 0.814 | 0.745 | 0.870 | – |
| Fracture | 18 | 0.855 | 0.793 | 0.793 | 0.801 | 0.876 | 0.891 | – |
| **Macro** | – | **0.866** | **0.887** | **0.893** | **0.835** | **0.869** | **0.902** | **0.892** |

\*JEV hybrid chose sources per finding using these same 58 studies, so its number is optimistic.

95% bootstrap CIs (2,000 resamples) [measured]:
- pilkwang macro 0.827–0.900
- steven v2 macro 0.851–0.917
- lixin macro 0.805–0.865
- per-finding width examples (steven v2): Lateral OA 0.62–0.97, Synovitis 0.66–0.90, Fracture 0.63–0.93, Effusion 0.73–0.94

**Consistently hardest for report labels:** Synovitis (0.68–0.79), Lateral OA (0.79–0.86), Contusion (0.75–0.87), Effusion (0.70–0.88), Lateral Meniscus (0.84–0.88), and Fracture in some sources (0.79–0.89). **Easy:** ACL, MCL (0.96–0.99), Medial Meniscus, Baker's, Medial OA (0.90–0.95).

---

## 2. Label prevalence

### 2a. Gold 58 [measured; matches the counts in the laymond and steven cards]

| Finding | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| positives / 58 | 24 | 9 | 26 | 23 | 15 | 11 | 21 | 35 | 27 | 12 | 19 | 18 |
| rate | .414 | .155 | .448 | .397 | .259 | .190 | .362 | .603 | .466 | .207 | .328 | .310 |

Findings per study: mean 4.14, min 1, max 9. **No gold study is all-negative.** "Gold prevalence is the annotator's sampling, not disease prevalence" [ds:stevenleehans].

### 2b. LLM-labelled prevalence on the 4,349 non-gold studies (score > 0.5) [measured]

| Finding | pilkwang YES | steven v2 | lixin | laymond | gold58 |
|---|---|---|---|---|---|
| ACL | .275 | .219 | .200 | .215 | .414 |
| MCL | .172 | .156 | .146 | .149 | .155 |
| Medial Meniscus | .532 | .536 | .393 | .509 | .448 |
| Lateral Meniscus | .251 | .257 | .148 | .214 | .397 |
| Medial OA | .367 | .379 | .359 | .377 | .259 |
| Lateral OA | .246 | .289 | .257 | .204 | .190 |
| PF OA | .452 | .465 | .451 | .464 | .362 |
| Effusion | .602 | .599 | .590 | .329 | .603 |
| Synovitis | .130 | .593† | .123 | .569† | .466 |
| Baker's | .252 | .247 | .246 | .247 | .207 |
| Contusion | .290 | .248 | .164 | .266 | .328 |
| Fracture | .068 | .072 | .064 | .048 | .310 |

†After Effusion/OA co-occurrence fill.

**How often the report is silent** (pilkwang verdict UNK on non-gold studies) [measured]:

| Finding | UNK share |
|---|---|
| Synovitis | 84% |
| Fracture | 57% |
| Baker's | 46% |
| Lateral OA | 33% |
| Medial OA | 26% |
| Contusion | 21% |
| PF OA | 18% |
| all ligament/meniscus targets | 6–10% |

sadamtorres: "About 97% of studies share one value for [Fracture] ... A target with almost no variance produces almost no gradient", and Fracture sat at 0.635 AUC after fine-tuning [nb:sadamtorres]. **Fracture is a label problem, not a vision problem.**

### 2c. Per-finding difficulty for image models

- Frozen DINOv2 on gold-58: focal findings collapsed (Medial Meniscus 0.65, Fracture 0.66, MCL 0.68) while diffuse ones held up (Medial OA 0.88, Baker's 0.90, Effusion 0.87). Fine-tuning at 224 px gave Medial Meniscus +0.171, MCL +0.118, ACL +0.113, Contusion +0.099 [nb:sadamtorres].
- The renta CoAtNet trained on the full 4,349 studies gained most on Lateral Meniscus (+0.071), Fracture (+0.057), Lateral OA (+0.048) and Medial Meniscus (+0.035), "the findings that were capping us" [nb:renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid code comments].
- The top public stacks route Lateral Meniscus, ACL, Medial Meniscus, Lateral OA and Fracture to specialist CoAtNet arms [nb:beraterolelk/0-941-sota…, nb:starkhushi].

---

## 3. How the 58 gold studies are used, and the noise problem

Usage patterns seen:
1. **Validation only, never trained on** [nb:dreaddevelopment/knee-mri-training-the-twelve-finding-model]. The best epoch is picked by gold-58 macro AUC, and the top-3 epochs by gold AUC are weight-averaged. *Caveat: that is selection on the evaluation set.*
2. **Trained on at 3× weight, and scored only on gold studies that land in the holdout fold** [nb:pilkwang, nb:prvsiyan §7, nb:stevenleehans/rsna-knee-101-read-this-before-you-start]. stevenleehans points out a leak in an earlier source notebook: it "scores the annotated AUC over every gold study, including the ones in the training split, and then uses that number to choose the checkpoint".
3. **Held out and cross-fitted.** Each gold study is predicted only by a fold model that never saw it. yash `.903`: clean OOF 0.8544 vs cross-fitted gold 0.8568, "agreement between those two disjoint checks is much stronger evidence" [nb:prvsiyan §9].
4. **Choosing models by proxy score.** `0.7·CV AUC + 0.3·gold58 AUC` on 20 proxy trials using 10% of the data [nb:prvsiyan §9, yash].
5. **Choosing labellers or label sources per finding** [ds:riadmohamed42, ds:stevenleehans].

Evidence that tuning on 58 is noise:
- **Standard error.** Hanley–McNeil at A ≈ 0.85 with 9/58 positives gives SE ≈ 0.083, or ±0.16 at 95% for one finding [nb:prvsiyan §2.2]. The macro bootstrap CI is about ±0.08 per [nb:ulasdesouza], ±0.04 per [nb:sadamtorres], and about ±0.03–0.04 per our measurement (width 0.066 for steven v2).
- **renta0426.** A 3-arm blend showed "+0.010" on a 45-study gold set, but on the live LB CoAtNet alone scored 0.914 against 0.914–0.915 for every blend, so "the +0.010 was gold-set noise".
- **prvsiyan.** A per-finding multi-encoder overlay chosen on the 58 scored **0.887 public vs the 0.906 anchor**, "falsifying broad post-selection on only 58 image-labelled studies".
- **riadmohamed42.** A per-target blend scored higher on the 74 (0.9237) but lower on a 7-study holdout (0.9452 vs 0.9684), so the author kept the global blend.
- **starkhushi.** The per-label outer map "was tuned on the public split" and is fragile; keep a flat-weight version as the second final pick.
- **Where gold helps.** Report-label OOF understates real vision gains. Moving from 224 to 336 px gave OOF +0.0035 but LB +0.017, because a better model disagrees with the report labels exactly where the reports are wrong. Recommended protocol: use OOF on 4,407 to pick epochs and catch breakage, and gold-58 to decide whether a direction is worth pursuing [nb:sadamtorres].
- **Gold-58 → public LB offset.** Three systems landed about +0.044 above their gold-58 macro (0.771 → 0.776, 0.824 → 0.866, 0.857 → 0.903) [nb:sadamtorres]. **UNVERIFIED** as a general rule. The first pair is +0.005, not +0.044, so the claim is loose.

Other leakage traps:
- **Duplicate reports.** 46 report texts are shared by 2 or more studies, 177 studies in total (exact-text match) [measured]. Split folds by a report hash [nb:pilkwang, nb:prvsiyan]. prvsiyan normalizes dates and numbers before hashing.
- **Patients.** Patients map 1:1 to studies (4,407/4,407), per [nb:dariushafshar] citing wguesdon, so `StudyInstanceUID` is a safe grouping key. **UNVERIFIED by us.**
- **Site grouping.** A scanner/protocol fingerprint (e.g. Manufacturer / Model / field strength) can feed `StratifiedGroupKFold` [nb:blacklions, nb:beraterolelk EDA 6].
- **Public label files with gold pasted in** (yunus v5, barun) give AUC = 1.0 on gold. Never validate with them.

---

## 4. Facts about train.csv, the test set and the leaderboard

- **train.csv has 4,407 rows**, 4,407 unique `StudyInstanceUID`, **58 with all 12 labels** and 4,349 with only `Report` [measured; also stated in nb:prvsiyan, ds:stevenleehans and many others]. This settles CLAUDE.md open question 2 (UNVERIFIED only in that it comes from a third-party copy; recheck against the official file).
- **Report length** in characters: mean 1,098, median 977, IQR 588–1,460, min 52, max 4,743, no empty reports. Gold reports are longer: median 1,206 vs 974 [measured].
- **Languages: 9 languages in 3 scripts** [nb:prvsiyan, nb:pilkwang, nb:nathanjacob].

  | Language | Share (ulasdesouza) | Count (our crude guesser) |
  |---|---|---|
  | English | ~40% | 1,815 |
  | Spanish | ~16% | 576 |
  | Turkish | ~12% | 487 |
  | Croatian/Serbian | – | 405 |
  | Greek | ~7% | 321 |
  | German | ~6% | 258 |
  | Bulgarian/Russian (Cyrillic) | ~5% | 220 |
  | Dutch | ~3% | 152 |
  | French | ~1.4% | 64 |
  | unknown | – | 109 |

  Gold-58 by language: en 28, es 8, tr 6, hr 4, el 3, bg 3, nl 2, de 2, ? 2 [measured]. "58 studies, most of them English, cannot see a Greek and Spanish coverage fix" [nb:prvsiyan].
- **Sites:** "4,407 training studies from 16 sites across 5 continents" [nb:nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937]. **UNVERIFIED** (no source given). Language tracks site, so lexicon gaps become site bias [nb:prvsiyan, nb:pilkwang].
- **`Fluid_Sensitive` and `Fat_Suppression` are identical across all 24,371 train series rows** [nb:karnakbaevarthur/rsna-knee-eda-to-2-5d; nb:pilkwang says "agree on every row"]. **This contradicts the CLAUDE.md line "correlated but not equivalent"** (taken from the Kaggle Data page). Verify on the real file. If they are identical, recover fat-sat and weighting from DICOM headers (`SeriesDescription`, `ScanOptions`, TR/TE), as pilkwang does.
- **`Laterality` DICOM tag:** absent on about half the studies, by whole vendors [nb:pilkwang] (sadamtorres says empty in ~72%). Laterality comes from the image-centre x-coordinate instead. Relevant to label semantics, because medial/lateral labels are only learnable after that normalization.
- **Test set:** the visible `test.csv` has 3 placeholder studies, and the hidden rerun has **1,322 studies** [nb:renta0426]. That is consistent with Kaggle's "~1,300".
- **Public/private split and prevalence shift:** no notebook gives the public/private ratio or any measured prevalence shift. Kaggle only says prevalence "may differ" (quoted by pilkwang and prvsiyan). **UNVERIFIED / unknown.** Several notebooks warn that per-finding weights fitted to the public split "can do real damage privately" [nb:beraterolelk, nb:evgendvorkin].
- **Are test labels made the same way as the 58 gold** (image-read annotators)? Implied by several authors (sadamtorres, prvsiyan, ulasdesouza) but **UNVERIFIED**. If true, the gold annotators' policy (stricter Effusion and Contusion, image-visible Synovitis, higher-sensitivity Fracture) is what the leaderboard rewards.
- **Leaderboard context from these notebooks** (the leaderboard has moved since):
  - public anchors went 0.891 (pilkwang 20× DINOv2) → 0.906 (prvsiyan) → 0.914 (renta CoAtNet single arm) → 0.937–0.943 (public ensembles) [nb:prvsiyan, nb:renta0426, nb:starkhushi]
  - a single DINOv2-B fine-tune reached 0.866 at 224 px and 0.883 at 336 px [nb:sadamtorres]
  - the yash B3 system reached 0.903 [nb:prvsiyan §9]

---

## 5. Why public forks plateau at 0.941 (starkhushi) and what top labelling does

[nb:starkhushi/rsna-knee-why-public-forks-stop-at-0-941]:
- About **200 teams sit at exactly 0.941**, nearly all running a fork of mattiaangeli's "Bend the Knee to the Dinosaurs" ensemble with the maverickss26/renta0426 "0.941 recipe".
- **The plateau comes from everyone using the same members, not from labels.** The recipe is: a per-label outer map (CoAtNet weight per finding, Lateral Meniscus = 1.00), Raptor arm weights of 0.55/0.15, and an inner CoAt re-rank. One of its four edits is a no-op. starkhushi's hypothesis is to revert the arms to 0.60/0.10, as Mattia's own stack (0.942, later 0.946) did. **The score of that change was pending (UNVERIFIED).**
- **Silent degradation.** If `model_sources`/`kernel_sources` are missing, all 20 DINOv2 members drop and the score falls to 0.937.
- Adding its own 2.5D members (ResNet34/ConvNeXt/RadImageNet, holdout 0.85–0.87) at 15% weight scored 0.936. "Below ~0.90 solo, a member costs more than its diversity buys."
- **Labels in that stack:** every member traces back to the same public label sets (pilkwang, stevenleehans, lixin73; see mattiaangeli/sofiaanjenje credits). So **label quality is shared across all 0.941 forks. Better-than-public labels are one of the few ways to differ from them** (inference, not measured).

Top labelling tricks (ranked by how solid the evidence is):
1. **LLM beats regex**, +0.03 to +0.07 macro vs gold, almost all from non-English recall [ds:stevenleehans, nb:wguesdon].
2. **Use several decorrelated labellers**, blended per finding or with consensus weights (+0.006 for steven v2 → v4; consensus of 3 = 0.892) [ds:stevenleehans, measured].
3. **Fill silent Synovitis from Effusion/OA** (+0.11 Synovitis AUC, +0.009 macro) [ds:stevenleehans, nb:laymond].
4. **Tighten Effusion** to moderate or larger, and exclude degenerative oedema from Contusion [nb:laymond, ds:riadmohamed42, nb:prvsiyan §9].
5. **Soft/graded targets plus a confidence weight**, with unmentioned findings getting low weight rather than a hard 0 [all].
6. **More weakly labelled studies help:** 3,155 → 4,349 training studies gave +0.013 on the gold-58 gate (better in 92.7% of bootstraps) [nb:renta0426].
7. **The model can beat its teacher.** Image models end up above report-label OOF on the LB. Don't optimize purely for agreement with report labels [nb:sadamtorres].

---

## 6. Gold labels vs report labels: where they disagree

pilkwang verdicts against gold [measured]:

| Finding | gold+ : YES / NO / UNK | gold− : YES / NO / UNK |
|---|---|---|
| ACL | 23 / 0 / 0 | **9** / 21 / 4 |
| MCL | 9 / 0 / 0 | **9** / 33 / 6 |
| Medial Meniscus | 25 / 1 / 0 | 7 / 22 / 2 |
| Lateral Meniscus | 20 / 2 / 1 | **10** / 18 / 6 |
| Medial OA | 14 / 1 / 0 | 8 / 20 / 14 |
| Lateral OA | 9 / 2 / 0 | 7 / 17 / 22 |
| PF OA | 16 / 1 / 3 | 8 / 18 / 11 |
| Effusion | 34 / 0 / 0 | **16** / 5 / 2 |
| Synovitis | 13 / 0 / **14** | 3 / 0 / 27 |
| Baker's | 11 / 0 / 1 | 5 / 12 / 28 |
| Contusion | 16 / 1 / 1 | **20** / 10 / 9 |
| Fracture | 13 / 0 / 4 | 6 / 12 / 22 |

Reading the table:
- **The report almost never misses a gold positive it discusses.** Very few gold positives get a NO verdict. The misses are **silence**: Synovitis 14 of 27, Fracture 4 of 18.
- **Many gold negatives are asserted by the report.**
  - Effusion 16 of 23 and Contusion 20 of 39 gold negatives have a report YES. The annotator's cut-off is stricter (e.g. small/trace effusion or degenerative marrow oedema is marked 0).
  - Ligaments and menisci (ACL 9, MCL 9, LM 10) point the same way: sprains, low-grade or partial injuries and degeneration are probably not counted as gold positives. **UNVERIFIED interpretation.**
- **Consensus of 3 LLM sources binarized at 0.5, against gold** [measured]: 129 of 696 cells disagree (per finding: ACL 7, MCL 8, MM 6, LM 12, MOA 8, LOA 10, PFOA 12, Eff 16, Syn 18, Baker 6, Cont 15, Frac 11). False positives (gold 0, labels 1) outnumber false negatives in every finding except Synovitis (14 FN vs 4 FP).
- **Examples quoted by others:**
  - "a report saying *moderate joint effusion* against an annotation of 0, an annotation of Baker's cyst against a report that calls the study normal" [nb:prvsiyan]
  - "One study's report says 'Moderate joint effusion' yet Effusion=0; another describes a lateral meniscus complex tear yet Lateral Meniscus=0" [nb:ulasdesouza]
- **Size of the disagreement:** "Report and image disagree irreducibly maybe 10–15% of the time. The report reader therefore has a ceiling around 0.88–0.90" [nb:sadamtorres, estimate]. JEV notes "part of [Synovitis] evaluation gold comes from findings the report never states ... the same applies to some Lateral OA rows" [ds:riadmohamed42].

---

## 7. What this means for our labelling plan (Dom's 58-study teacher idea vs report-derived labels)

**Dom's plan as stated in CLAUDE.md:** fine-tune a labeller on the 58 verified studies, pseudo-label the other 4,349, then train the ViT on the enlarged set. Report-derived labels are the fallback and cross-check.

What the evidence says:

1. **An image-only teacher trained on the 58 studies will be much weaker than report labels.** A frozen RadImageNet ResNet-50 plus a head trained on gold only scored **0.66 macro AUC** with 5-fold CV [nb:dariushafshar]. Frozen DINOv2 trained on ~4.3k report labels only reached 0.77 on gold [nb:sadamtorres]. Public LLM report labels score **0.87–0.89** against the same gold without seeing a single pixel. Pseudo-labelling 4,349 studies with a 0.66 teacher would feed the ViT labels far noisier than the free report labels, with 58 positives or fewer per finding (MCL has 9, Lateral OA 11). **Recommendation: do not make an image teacher trained only on the 58 the primary label source.**
2. **Dom's idea works if the teacher reads the report, not the image.** Reports exist for all 4,349 unlabeled training studies, so a "teacher trained on 58" can be a *text* model. The strongest form of the idea:
   - an LLM (or several) reads each report into categorical states (UNK / NO / mild / moderate / severe / uncertain)
   - the **58 gold studies fit a small per-finding calibration** mapping those states (plus co-occurrence features such as Effusion → Synovitis) to the gold annotators' policy
   - this is blacklions' "gold calibration" and sadamtorres' "mild effusion → 0.45"
   - it keeps the part of Dom's idea that matters (learn the annotators' threshold from gold) and discards the part that cannot work (learning image features from 58 scans)
   - keep the model tiny (a few parameters per finding), or it will overfit 58 rows
3. **Use gold to set conventions, then leave it alone.** Per-finding choices made on 58 studies have repeatedly failed on the LB (§3). Decide a few global conventions (effusion cut-off, contusion excluding degenerative oedema, synovitis backoff, silence = low weight) and do not tune per finding past that.
4. **Report-derived labels should be the primary source, built from several decorrelated readers.**
   - Start from the public sets (pilkwang, steven v2, lixin, laymond Qwen32B). Their licenses: pilkwang CC0, lixin CC0, JEV CC0; **steven and laymond licenses not checked (UNVERIFIED)**. Check the winner's license rules (CC-BY-NC relicensing, "publicly available" external data).
   - Add **one reader of our own** so our labels differ from the 200 teams at 0.941.
   - Blend in probability space with agreement/certainty weights (prvsiyan's `build_public_consensus` formula is a ready template).
5. **Validation protocol for Dom's ViT:**
   - **(a)** Folds by study, grouped by normalized report hash (46 duplicate groups) and, if we can build it, by site/scanner fingerprint.
   - **(b)** Report-label OOF on about 4.3k studies selects epochs and catches breakage.
   - **(c)** Gold-58 is scored **only out of fold** (cross-fitted), always reported with a bootstrap CI, and used to decide directions, not to fine-tune weights.
   - **(d)** Do not train on gold for the model we validate on. Or cross-fit it, at 2–3× weight, as pilkwang/prvsiyan/stevenleehans do.
   - **(e)** Expect the LB to be above gold-58 by some offset (sadamtorres claims about +0.044, UNVERIFIED).
6. **Per-finding priorities for label work** (each finding at chance costs (M − 0.5)/12 ≈ 0.03):
   - **Synovitis:** mostly unreported, so model it from Effusion/OA/Hoffa proxies plus low confidence. The image model must learn it.
   - **Fracture:** 57% silent in reports, 31% positive in gold. Treat silence as weak evidence, not as a negative.
   - **Lateral OA:** needs compartment attribution plus OA inheritance.
   - **Contusion and Effusion:** apply the stricter annotator threshold.
   - **Lateral Meniscus:** the weakest meniscus in every source.
7. **Keep this section as a record of what we know.** Things we cannot verify yet: whether test labels come from the same annotators as the gold 58, the public/private split and prevalence shift, and whether "16 sites" is right. Re-check the Kaggle Data/Overview pages and the forum before building on any of them.

### Sources pulled (all under `scratchpad/nb/` and `scratchpad/ds/`, outside the repo)

- **Notebooks:** prvsiyan/rsna-knee-read-the-report-then-the-knee, pilkwang/rsna-knee-baseline-v1, renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid, starkhushi/rsna-knee-why-public-forks-stop-at-0-941, dreaddevelopment/knee-mri-training-the-twelve-finding-model, dariushafshar/rsna-report-pseudo-labels-add-0-022-auc-4-5, laymond/rsna-knee-mri-weak-labeling-with-qwen2-5-32b, ulasdesouza/01-report-labels, riadmohamed42/jev-knee-labels-verification, hosseini7798/rsna-knee-mri-labeling-typesafe-jev, blacklions/report-teacher-anatomy-aware-hierarchical-multimod, karnakbaevarthur/rsna-knee-eda-to-2-5d, jek1wantaufik/rsna-knee-mri-baseline-with-pseudo-labels (no markdown), sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee, wguesdon/rsna-knee-dinov2-at-meniscus-resolution, stevenleehans/rsna-knee-101-read-this-before-you-start, beraterolelk/0-941-sota-rsna-knee-dinov2-radimagenet, nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937.
- **Not pullable:** yashbishnoi98/rsna-knee-infer-v1 (private or renamed; details come only via prvsiyan §9).
- **Datasets:** pilkwang/rsna-knee-llm-labels, stevenleehans/rsna-knee-llm-report-labels, lixin73/rsna-knee-llm-report-labels-sol56, yunusgmsoy/rsna-knee-llm-labels-4-source-merged, laymond/rsna-knee-abnormality-qwen3-8b-weak-labels, riadmohamed42/jev-knee-labels, barun2104/rsna-knee-stratified-folds-and-llm-soft-labels.
