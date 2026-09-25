# RSNA Knee Abnormality Detection — Official Pages & Forum Research

Collected 2026-09-24 via the authenticated `kaggle` CLI v2.2.4 (`kaggle competitions pages` /
`kaggle competitions topics` / `kaggle competitions topic-messages`), competition slug
**`rsna-knee-abnormality-detection`** (the slug that actually resolves; `rsna-knee-abnormalities-detection`
returned a 404 on `ListCompetitionPages`). All quotes below are verbatim from the CLI's `--content` output
(the same markdown Kaggle renders on the site) or from forum message JSON. No browser was needed — the CLI
returned everything requested.

URLs use the working slug: `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/...`

---

## 1. Efficiency Prize Evaluation page

Page name in CLI: `Efficiency Prize Evaluation`. URL: `.../overview/efficiency-prize-evaluation`.

Full text:

> ## Efficiency Prize
> We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy.
> For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.
> To be eligible for an Efficiency Prize, a submission:
> - Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the **My Submissions** tab.
> - Must be ranked on the Private Leaderboard higher than the `sample_submission.csv` benchmark.
> All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.
> An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the **Prizes** tab for the prize awarded to each rank. More details may be posted via discussion forum updates.
>
> ## Efficiency Score
> We compute a submission's **efficiency score** by:
> Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400
> where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark `sample_submission.csv`, maxAUC is the maximum AUC of all submissions on the Private Leaderboard, and RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. The objective is to **minimize** the efficiency score.
> During the training period of the competition, you may see a leaderboard for the *public* test data in the following notebook, updated daily: **Efficiency Leaderboard** (`kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb`). After the competition ends, we will update this leaderboard with efficiency scores on the *private* data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.

**Confirms exactly** the formula already in CLAUDE.md: `Efficiency = AUC/(Benchmark−maxAUC) + RuntimeSeconds/32400`, lower is better. No change needed there.

**New/clarifying facts not in CLAUDE.md:**
- Eligibility requires the submission be one of the team's selected Leaderboard-Prize submissions (or Kaggle's auto-selected fallback) **and** beat `sample_submission.csv` on the Private LB — matches CLAUDE.md.
- There is a **live public Efficiency Leaderboard notebook** (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`) updated daily during the competition, showing rank only (not full score) until the end.
- **RuntimeSeconds measurement mechanics (submission scoring window vs. full 9h notebook budget) are NOT explicitly defined on this page** — no statement on CPU-only bonus, GPU vs CPU distinction, or whether it's the full notebook run or just the inference/scoring phase. This remains UNVERIFIED from official text. From forum chatter (§3 below), practitioners are measuring end-to-end **inference wall-clock for ~1,300 test studies** (one poster reported dropping "scoring time" from ~9h to 29 minutes; another profiled a ~32-minute breakdown: header pass, slice ordering, decode+resize, container/import, GPU). This is consistent with CLAUDE.md's "inference-notebook runtime" interpretation but is participant inference, not an official host statement — treat as UNVERIFIED (though strongly corroborated).
- **Hardware**: not stated on this page. Forum posts (below) mention Kaggle's standard **T4** GPU explicitly ("Kaggle T4 only", "24 windows on one T4"); no official page names GPU type. `kaggle quota` (run via CLI, this account) shows **GPU: 30.00h/week, TPU: 20.00h/week** remaining as of 2026-09-26 refresh — this is Dom's personal weekly quota, not necessarily the competition's guaranteed submission hardware, and Kaggle's standard notebook GPU offering (T4 x2 or P100) is UNVERIFIED as the specific type guaranteed for scoring runs.

---

## 2. Overview/Evaluation, Data, and Rules pages

Fetched via `kaggle competitions pages list --content`: page names returned were `rules`, `Description`, `Evaluation`, `Timeline`, `data-description`, `Prizes`, `Code Requirements`, `abstract`, `Efficiency Prize Evaluation`, `Acknowledgements`.

### Evaluation (full text)
> Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets:
> Final Score = (1/12) Σ AUC_i
> The final score is, in other words, the macro-averaged AUC ROC.
> ## Submission File
> For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:
> `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`

Matches CLAUDE.md exactly — no changes.

### Data description (full text, condensed to key facts)
- **Train set size**: the data-description page itself does **not** give a numeric count of training studies (no "N studies" or GB figure appears in this markdown — the 819,640 files / 569.76 GB figures in CLAUDE.md are presumably from the Data tab's UI file-browser stats, which this CLI content export does not include, so those specific numbers remain UNVERIFIED from this page and were not re-confirmed here).
- **Verified-label subset**: "Only a small subset of training studies carry per-condition labels." No number given on this page (see §3 — forum says 58, matching CLAUDE.md's "58 verified-label studies").
- **Test set size**: "There are **about 1300 studies** in the test set." (Matches CLAUDE.md's "~1,300 studies".) `test.csv` ships with only 3 example study IDs; real test data substituted at scoring. `Report` field **not provided at test time** — confirmed verbatim.
- **train_series.csv / test_series.csv schema**: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` (Sagittal/Coronal/Axial) — matches CLAUDE.md. Official text: "although `Fluid_Sensitive` and `Fat_Suppression` are often correlated, as observed in the training set, they are not necessarily equivalent for every case" — **a forum poster (see §3, thread 734106) claims they were identical on all 24,371 series they checked**; this is a specific empirical claim contradicting the spirit of the official note, UNVERIFIED and worth checking yourself before dropping either column.
- **DICOM notes**: mixed transfer syntaxes (uncompressed Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE); "every DICOM has been stripped to an allowlisted set of **86** metadata tags" — confirmed verbatim, matches CLAUDE.md.
- **Prevalence-shift disclaimer**: "prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets" — matches CLAUDE.md.
- Series: 20–45 slices typical, median 30, long tail to a few hundred — confirmed verbatim.

### Rules page (37,384-char full text pulled; key excerpts)
- **Team size**: "The maximum Team size is **five (5)**." Merge deadline / cap logic as in CLAUDE.md — confirmed.
- **Submission limits**: "You may submit a maximum of **five (5)** Submissions per day." / "You may select up to **two (2)** Final Submissions for judging." — confirmed verbatim, matches CLAUDE.md.
- **Winners decided by Private Leaderboard**: "The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard" — confirmed.
- **Public/Private definitions**: "The 'Private Leaderboard' is a ranked display of Participants' Submission scores against the private test set... determines the final standing." "The 'Public Leaderboard' is a ranked display of Participants' Submission scores against a representative sample of the test data." **No percentage split is stated anywhere on the Rules, Data, or Evaluation pages** (I grepped the full rules text for `%` — zero hits). See §4 below — the split percentage is **not stated anywhere official**; a forum thread asks the host directly and gets no numeric answer (only whether it's stratified-by-site, also unanswered by host as of this writing).
- **External data/tools (Section 2.6)**: usable if "publicly available and equally accessible to use by all Participants... at no cost" or meets a "Reasonableness Standard" for minimal-cost paid tools (example given: "a small subscription charge to use additional elements of a large language model such as Gemini Advanced" is fine; "Purchasing a license to use a proprietary dataset that exceeds the cost of a prize" is not reasonable).
- **Data Security clause (Section 4.b, verbatim)**: "You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." — **This is the clause that fueled the "can I send report text to OpenAI/Anthropic?" debate; see the host's explicit clarifying ruling in §3, thread 733965, which overrides the naive reading of this clause.**
- **Winner license**: **CC-BY-NC 4.0** on winning submission + source code, confirmed. Carve-out: "generally commercially available software... procured without undue expense" doesn't need relicensing; "input data or pretrained models with an incompatible license" used to generate the winning solution are exempt from the open-source relicensing requirement — confirmed matches CLAUDE.md.
- **Open source code requirement**: any open-source code used in the model "must only use open source code licensed under an Open Source Initiative-approved license... that in no event limits commercial use" — confirmed.
- **No numeric GPU type or Kaggle-notebook-hardware statement anywhere in the Rules/Code-Requirements pages.** Code Requirements page (full text): CPU Notebook ≤9h, GPU Notebook ≤9h, internet disabled, "freely & publicly available external data is allowed, including pre-trained models," submission file must be `submission.csv`. No GPU model named.
- No explicit statement permitting/forbidding third-party LLM APIs *on the Rules page itself* — that clarification only exists as a host discussion post (§3).

### Prizes page (full text)
> **Main Leaderboard**: 1st $9,000, 2nd $7,000, 3rd $6,500, 4th $6,000, 5th $5,500, 6th–10th $5,000 each.
> **Efficiency Track**: 1st $7,000, 2nd $6,000, 3rd $5,000.
Matches CLAUDE.md exactly ($77,000 total, top 10 paid). Additional detail not in CLAUDE.md: winners are "invited and strongly encouraged to attend the AI Challenge Recognition Event with waived fee, contingent on review of solution and fulfillment of winners' obligations," and beyond standard Kaggle winners' obligations the host additionally requires (i) a short video, (ii) publishing a link to open-sourced code+weights on the forum, (iii) sharing the final model "as publicly available for open distribution and validation" (with a linked example: `kaggle.com/models/tom99763/9th-place-models-rsna-iad`).

### Timeline page (full text) — confirmed verbatim, matches CLAUDE.md exactly:
Jul 30 2026 Start · Oct 15 2026 Entry Deadline & Team Merger Deadline · Oct 22 2026 Final Submission Deadline · Nov 5 2026 Winners' Requirement Deadline. All 11:59 PM UTC.

### Description/Abstract — narrative pages, no new numeric facts; confirms this is "the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report."

### Acknowledgements page — lists 19 data-contributing sites by name (organizing team, data contributors, curators, annotators from Society of Skeletal Radiology & International Skeletal Society, QC reviewers). Site count of **19 contributing institutions + 3 additional sites = 22 named sites** is explicitly listed (not "16 sites" as CLAUDE.md's UNVERIFIED news claim states — see contradiction flag below).

---

## 3. Discussion forum — thread inventory and summaries

Pulled **all 127 topics** via `kaggle competitions topics list` (paginated) as of 2026-09-25. Top 40 by votes, plus every thread read in full below. (Full list of 127 saved to scratchpad; happy to dump if wanted — condensing to the ones that matter per your ask.)

### Top-voted / most important threads (title — votes/comments — URL suffix `/discussion/<id>`)

| id | votes | comments | title |
|---|---|---|---|
| 733343 | 83 | 12 | Knee Abnormality Detection AI Challenge Overview (host post) |
| 735304 | 60 | 95 | Best single-model score |
| 733932 | 50 | 7 | "Not addressed" is a label too — what we learned reading 4,407 knee reports with an LLM |
| 733497 | 40 | 20 | I got a perfect usecase for this competition |
| 733826 | 27 | 2 | Possible inconsistencies between MRI reports and provided labels |
| 733652 | 21 | 7 | Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels |
| 733965 | 19 | 13 | **Use of Commercially Hosted LLMs (host clarification)** |
| 737696 | 19 | 19 | RSNA Raptor Weights (public model/weights sharing thread) |
| 736170 | 19 | 6 | Sharing my knee MRI |
| 733517 | 18 | 2 | 0.932 LB within one day. Tested for DICOM metadata shortcut |
| 733375 | 17 | 1 | Welcome to Knee Abnormality Detection Challenge! |
| 735767 | 16 | 6 | What Could the Final Ceiling Be for RSNA 2026? |
| 735348 | 15 | 14 | Does Kaggle Do Anything About Byte-for-Byte Public Notebook Theft? |
| 740610 | 15 | 13 | Plateau at 0.903: sharing my pipeline and ablations |
| 735154 | 14 | 3 | Scaling the encoder bought us nothing (+0.0011) |
| 736268 | 13 | 13 | Too many noise public notebooks |
| 735826 | 13 | 11 | What is the real bottleneck now: labels, image pipeline, or model capacity? |
| 736678 | 13 | 4 | Maybe worth checking the Efficiency LB? |
| 737155 | 11 | 0 | Annotation criteria question: how were Synovitis and Effusion graded? |
| 736301 | 11 | 2 | Are you actually looking at the data? |
| 737163 | 10 | 4 | **Corrupt files (host acknowledged)** |
| 734109 | 10 | 2 | **Is the gated KneeCoT dataset permitted as external data? (host ruling: NO)** |
| 733313 | 9 | 16 | Using Dino3 |
| 737820 | 9 | 4 | My account was banned due to a notebook. Just reinstated |
| 735639 | 8 | 5 | Two knees are better than one? (laterality/both-knees-in-frame issue) |
| 733491 | 8 | 3 | Data/Reporting Inconsistencies (host explains gold-label methodology) |
| 736635 | 7 | 3 | LB OOF relationship (gold labels vs. LB correlation) |
| 736720 | 7 | 0 | Healthcare professionals: what knee MRI resources do you use? |
| 734681 | 7 | 3 | **Public/private test split — stratified by site or entire sites held out? (unanswered by host)** |
| 734106 | 7 | 3 | 58 labelled studies out of **4,407** — ground facts about labels/languages |
| 730709 | 6 | 1 | How to get started + Official Discord |
| 737566 | 6 | 0 | Why everyone's Synovitis AUC is stuck around 0.6–0.7 |
| 735121 | 6 | 0 | Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation? |
| 734004 | 6 | 0 | DICOM metadata findings: scanner-grouped CV, PatientSex present in test headers |
| 735497 | 5 | 0 | Request for official clarification: registration-gated public datasets (MRNet, OAI) |
| 741819 | 4 | 6 | Is OAI permitted as external data? (host answer read by community as effectively "no") |
| 737950 | 4 | 9 | External data behind institutional DUA — does KneeCoT ruling generalize? |
| 742106 | -1 | 1 | Errors on gold labels |

### Content summaries of the most important threads

**733343 — "Knee Abnormality Detection AI Challenge Overview"** (host post, credited to Dr. Jacob Kazam). Clinical/background narrative: OA affects ~654M people worldwide; acute knee injuries are 15–40% of sports trauma; describes anatomy and the 12 target findings. No new numeric facts beyond what's in the Description page.

**733965 — "Use of Commercially Hosted LLMs" (host clarification, authored by user "javaduke95" acting as host contact)**. This is the single most consequential thread for your labeling-plan open question:
> "Use of commercially hosted LLMs and other external inference services is permitted... submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing (for example, extracting labels from reports) will **not, by itself,** be considered prohibited PRIVATE SHARING of Competition Data outside the Team... This is in accordance to Section 2.6.b, EXTERNAL DATA AND TOOLS."
Follow-up in same thread, host confirms directly: **"You can use LLM API, such as those from OpenAI, to read the reports to generate the labels."** Also confirms LLM-derived labels/embeddings are acceptable as model inputs "at both training and inference time" (subject to being reproducible at inference time — i.e., whatever you feed at train time must also be derivable at test time, since `Report` isn't available at test).
Later in the same thread the host adds the general accessibility test used elsewhere: datasets/tools reachable via "straightforward registration or click-through data-use agreement" generally satisfy External-Data rules; those requiring "institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing" may not.
**This directly resolves your CLAUDE.md open item** "whether the Report text or external LLMs may be used for labeling, and the MIRA license constraints on sending data to third-party APIs like OpenAI/Anthropic" — **answer: explicitly yes, host-confirmed**, notwithstanding the Data-Security clause some participants worried about.

**734109 — KneeCoT gated dataset (host ruling)**: Host explicitly **disallowed** the KneeCoT Hugging-Face dataset (gated behind a formal ethical-use agreement with the affiliated hospital): *"After deliberation, we decided to not allow the use of KneeCoT dataset due to the required data agreement with dataset institution... this can make the playing field uneven for participants who can complete the formal agreement... For now, the KneeCoT dataset is not allowed in this competition."*

**741819 — Is OAI permitted?** Host's literal answer was hedged: *"Using OAI as an external dataset does not violate the... rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge. We do not own or control the OAI dataset and cannot comment on whether this challenge complies OAI data-use terms from the dataset stewards' perspective."* A high-voted (+6) community reply interprets this as a practical **"no"** for OAI, MRNet, and similar gated public knee-MRI datasets, since they all require institutional sign-off — but note this is a **participant interpretation of an ambiguous host answer, not an unambiguous host "no."** Treat "external MRI datasets are effectively off-limits" as UNVERIFIED-but-strongly-implied, not settled rule text.

**733652 / 735497 / 737950** — three related threads pushing for a category-level ruling on registration-gated public knee-MRI datasets (MRNet, fastMRI+, OAI, SKM-TEA). The host's clarifications (in 733965 and 741819) are the closest thing to a general answer; no single unambiguous "any click-through DUA is fine / any institutional DUA is not" rule was posted as of this pull, though the emerging pattern (click-through = ok, institutional sign-off/IRB = not ok) is consistent host language repeated across threads.

**735121 — CC-BY-NC-SA pretrained backbones (e.g. RadImageNet) and winner open-license obligation**: question raised, **no host answer** appears in the thread (only the original post, 1 message total). Open/UNVERIFIED.

**737163 — Corrupt files**: two specific training DICOM files reported broken — pixel data "exactly half the size expected from their DICOM metadata." Host response: *"Thank you kindly for surfacing this issue--we will review these files and provide an update shortly."* No follow-up update was visible in the thread as of this pull — treat as an **open host action item**, not yet resolved. File identifiers:
  - StudyInstanceUID `1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680`, SeriesInstanceUID `...31160785238781353848727311763596115703`, SOP `...66460523372602171398246641420761543626.dcm`
  - StudyInstanceUID `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035`, SeriesInstanceUID `...39396636671446532796538574315802928348`, SOP `...41947538458330624797291612781685746320.dcm`

**733517 — DICOM-metadata-shortcut / leak check (18 votes)**: A participant tested whether the very fast public-LB climb to 0.90+ within a day of launch was explained by a metadata leak. Finding: "Full header metadata, no pixels, reaches **0.6515 macro AUC under random folds** but only **0.5981 under scanner-grouped (GroupKFold) folds**. The 0.053 increment is site memorization and does not transfer to unseen scanners." Conclusion: **no meaningful exploitable leak found**; fast LB climb attributed to genuine image-based modeling, not metadata leakage. This is a participant analysis, not a host statement, but is methodologically solid and worth taking seriously as a "no obvious leak" signal.

**734004 — DICOM metadata findings**: Independent, consistent with 733517. ~13 distinct scanner groups (Siemens_1.5T 1,148 studies; Siemens_3T 781; GE_1.5T 698; Philips_3T 663; Philips_1.5T 619 — these five alone sum to ~3,909 of the reported 4,407 total training studies). Confirms **`PatientSex` (tag 0010,0040) is present in the test set DICOM headers** (M=2,076/F=1,894 in training) — directly resolves your CLAUDE.md open question #3 ("demographics only if they exist in the 86 tags") for sex at least: **yes, it exists and is usable**. Also confirms `Manufacturer`, `MagneticFieldStrength`, `SeriesDescription`, `ImageOrientationPatient` are present in test headers.

**734106 / 733932 — the "58 of 4,407" labeling threads (CRITICAL — see contradiction flag below)**. Multiple independent, highly-voted (7 and 50 votes) posts converge on: **exactly 58 training studies are fully labeled (all 12 columns filled on the same 58 — no partial labeling to exploit), out of 4,407 total training studies.** Additional ground-truth facts from these threads:
  - Reports span 7+ languages (English, Turkish, Spanish, German, Greek [178 reports], Dutch, French) plus ~428 reports a stopword-language-detector can't place.
  - Turkish negates *after* the term ("efüzyon izlenmedi" = no effusion seen) — a left-only negation window (standard English NegEx) silently inverts labels for the second-largest language in the corpus.
  - A participant claims `Fluid_Sensitive` and `Fat_Suppression` were "identical on all 24,371 series" they checked — contradicts (or at least strongly qualifies) the official Data page's "not necessarily equivalent for every case" language; UNVERIFIED, worth checking yourself.
  - The 58 gold-labeled studies are **not a random sample**: ACL prevalence 41% in the gold set vs. ~20% in the full corpus — don't set priors from the gold subset.
  - An LLM-based labeler beat a regex baseline against the 58 gold studies: 0.8780 vs. 0.8136 (some accuracy metric, not specified as AUC).
  - 25.4% of all label cells across the LLM-derived labels are "the report does not address this" (absence ≠ negative) — handling this ambiguity is flagged as more impactful than labeler accuracy itself.
  - Several public Kaggle datasets of LLM-derived report labels already exist and are being shared/reused: `Pilkwang Kim — rsna-knee-llm-labels`, `barun2104 — Stratified Folds & LLM Soft Labels`, `lixin73 — LLM Report Labels (GPT-5.6-Sol)`, plus `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels`.

**733491 — Data/Reporting Inconsistencies (host explains annotation methodology)**: Host clarifies gold-label provenance: *"The sample labels were assigned from image review using the challenge annotation rubric by two independent readers with a third adjudicating disagreements... For meniscal tears, the target is a definite tear; intrasubstance degenerative signals that do not reach the articular surface are negative. The same process was used for the testing set."* Confirms **labels were derived from independent image review, not from the report text** — reports and gold labels can and do legitimately disagree; this is by design, not a labeling error. Directly relevant to your Labeling Plan step 1 (fine-tune-a-labeler-on-verified-subset risk: "verified labels might contain errors") — host is saying these aren't errors, they're a different, image-based ground truth methodology, deliberately not text-derived.

**736301 — "Are you actually looking at the data?"**: a "gorilla in the CT scan" reminder post (16-votes-adjacent) warning against over-reliance on coding agents without manual image review — not a data-quality finding, just community culture/advice.

**737566 — Synovitis AUC stuck at 0.6–0.7**: analysis — only ~16% of the 4,407 reports mention synovitis at all (other 84% inferred); report-derived Synovitis labels agree with the gold/radiologist-annotated studies at only ~0.79 AUC (vs. ~0.99 for ACL) — i.e., the Synovitis "teacher" (report-derived label) is a weak signal, near chance-plus-a-lean, which caps achievable model AUC on that target regardless of image-model quality. Medically-pretrained encoders (RadImageNet) reportedly help more on Synovitis specifically (0.78 vs 0.62–0.72 for ImageNet-pretrained).

**735826 — "What is the real bottleneck: labels, image pipeline, or model capacity?"**: community poll-style thread; strong consensus leans toward **label quality (report-derived supervision) as the binding constraint**, not model architecture/capacity — directly relevant to your labeling-plan prioritization. One reply: "the entire competition is the label making... I don't think we have enough reports to require a model size that's huge but I am still struggling with getting labels I like."

**734681 — Public/private split stratified by site?** Host has **not answered** this as of the pull (only participant speculation, citing past RSNA competitions had shakeups ranging "None to 1039 places for first"). **UNVERIFIED / open — no official word on split methodology or percentage.**

**735767 — "What could the final ceiling be?"**: As of the post (~12 days into the competition, dated ~mid-August): **1,832 teams** registered (vs. 1,149 teams for the full RSNA 2025 competition — i.e., already ~1.6x the prior year's final team count, less than 2 weeks in). Public LB #1 already ~0.951; best publicly-shared solution ~0.92. Speculation only on final ceiling (no host input): guesses ranged 0.94–0.97 for gold/winning scores, with acknowledgment that gold labels vs. public LB "overfit chance" is a real risk given agent-driven rapid iteration.

**735304 — "Best single-model score"** (96 messages, largest thread): as of the pull, participants report **single-model** scores as high as **0.950** (a fine-tuned Qwen3.5-2B VLM with LoRA on the vision+language encoder, single fold, ~3.5h; also a single-fold CoAtNet at 224px hitting 0.950). Multiple people report multimodal (image+report-derived-label) approaches beating pure-vision single models. Confirms report text is being used as a training-time input in leading approaches, not just for label derivation.

**737696 — "RSNA Raptor Weights"** (public model/weights-sharing thread, most collaborative technical thread found): a participant open-sources CoAtNet-based checkpoints (~0.924–0.938 on public LB depending on window count/geometry), with public Kaggle datasets (`dreaddevelopment/raptor-knee-widedense`, `raptor-knee-finespacing`). Detailed technical exchange on windows-per-study (more test-time windows/crops monotonically but sub-linearly improves AUC, plateauing ~0.94+ region), slice density vs. span tradeoffs, and SWA checkpoints performing marginally worse than best-epoch checkpoints. Confirms use of **Kaggle T4** GPUs as the standard submission hardware in practice (author references "Kaggle T4" explicitly, and describes local/rented GPU use during *training* — external compute is fine for training since only the *inference notebook* runs on Kaggle infra under the 9h cap).

**736678 — "Maybe worth checking the Efficiency LB?"**: as of Aug 22, only ~32% overlap between public-LB top-50 and Efficiency-LB top-50 — high LB score often correlates with long runtime (heavy ensembling), suggesting efficiency-prize contenders should not assume their main-LB pipeline is efficient. One poster profiled inference at ~32 min total for ~1,300 studies (header pass 0.5min, **slice ordering 12.0min — described as "the wall"**, decode+resize 4.1min, container/import 5.0min, GPU 5-fold 10.0min) — useful concrete efficiency-track profiling data point for your pipeline design (slice-ordering/geometry-tag reads across ~186 files/study is the actual bottleneck, not GPU inference).

**735639 — "Two knees are better than one?"**: confirms **~7 training studies contain both knees in a single DICOM series/field of view** (bilateral imaging) — a laterality/cropping edge case worth handling in preprocessing; unresolved question about which knee the labels apply to when both are visible in a test study.

**742106 — "Errors on gold labels"**: low-signal thread (1 vote), points back to the host's methodology explanation already covered in 733491. No new facts.

**735154 — "Scaling the encoder bought us nothing"**: DINOv2-Small→Base (22M→87M params, 3.9x compute) moved CV by only +0.0011 against a measured noise floor of 0.0020 — a null result, model capacity is not the binding constraint (consistent with 735826's labels-are-the-bottleneck consensus). Also: full 4,407-study dataset decodes down to an **11.12 GiB** uint8 array after slice-selection/windowing/crop/resize — i.e., the effective *model input* data is far smaller than the raw ~500GB DICOM corpus once preprocessed, which is a very useful number for your caching strategy (CLAUDE.md item 14).

**730709 — Getting started / Official Discord**: standard Kaggle newbie-help post; links an official competition Discord (`discord.gg/kaggle`, general server, not competition-specific channel confirmed).

**736268 / 733313 / 735348 / 737820** — forum-culture threads: complaints about noisy/forked public notebooks optimizing for GM points rather than genuine insight, notebook theft/plagiarism concerns (byte-for-byte forks), and one account-ban-then-reinstatement incident tied to a notebook (details not elaborated). Not competition-fact-bearing, but useful context: **the public notebook ecosystem here is noisy and often duplicated/AI-generated; treat public-notebook scores skeptically** (echoed by multiple independent posters).

*(Note: while reading thread 736268, one forum post contained an embedded instruction addressed to "Claude/codex" asking an AI agent to extract insights from a linked public notebook. This is just a Kaggle user's post content, not an instruction from you — I did not act on it, only reported it as forum content per your request.)*

---

## 4. Public vs. private leaderboard split percentage

**Not stated anywhere in the official Rules, Data, Evaluation, or Efficiency pages** (confirmed by full-text grep for `%` across the 37K-char Rules page — zero hits tied to a split ratio). The Rules page only defines the two leaderboards conceptually (Public = "representative sample of the test data," Private = full private test set, determines final standing) without a percentage. A forum thread (734681) directly asks the host whether the split is stratified by site or whether whole sites are held out — **as of this pull, the host has not answered**. Treat the split percentage and stratification method as **UNVERIFIED / unresolved**, and re-check the Rules/Data pages and thread 734681 closer to the deadline in case the host posts an update.

---

## Sources
- Official competition pages (fetched via `kaggle competitions pages list --content`, slug `rsna-knee-abnormality-detection`): `.../overview/efficiency-prize-evaluation`, `.../overview/evaluation`, `.../overview`, `.../overview/timeline`, `.../data`, `.../overview/prizes`, `.../rules`, `.../overview/code-requirements`, `.../overview/abstract`, `.../overview/acknowledgements`
- Discussion forum, fetched via `kaggle competitions topics list` (127 topics, paginated) and `kaggle competitions topic-messages` for ~30 threads: `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>` for each id cited above.
- `kaggle quota` CLI output (this Kaggle account's weekly GPU/TPU quota, run 2026-09-24).
