# Adapting ViTs to Multi-Series 3D Knee MRI — Research Notes

Prepared for the RSNA Knee Abnormality Detection project. All non-trivial claims are cited; anything without a citation and derived by reasoning alone is marked `[derived]`. Anything the sources didn't settle is marked `UNVERIFIED`.

---

## 1. Video/3D ViT tokenization: ViViT, TimeSformer, Video Swin, and anisotropic MRI patching

### ViViT (Arnab et al., ICCV 2021)
Source: [arXiv:2103.15691](https://arxiv.org/abs/2103.15691), [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2103.15691), [CVF version](https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html)

Two tokenization schemes:
- **Uniform frame sampling**: sample `n_t` frames, embed each independently with a standard 2D ViT patch embed, concatenate all tokens → `n_t · n_h · n_w` tokens. Purely spatial embedding; temporal fusion only happens inside the transformer.
- **Tubelet embedding**: extract non-overlapping spatio-temporal tubes of size `t × h × w` and linearly project each to `R^d` in one step, giving `⌊T/t⌋ · ⌊H/h⌋ · ⌊W/w⌋` tokens. This fuses spatio-temporal information *at the embedding stage* rather than relying purely on attention — smaller `t` gives more tokens and more compute.

Four model variants (increasing efficiency, decreasing token interaction density):
1. **Model 1 — unfactorized spatio-temporal attention**: all tokens go through standard joint self-attention. Full pairwise interaction from layer 1, but `O((n_t·n_h·n_w)²)` complexity — the most expensive.
2. **Model 2 — factorized encoder**: a spatial encoder attends only within a frame's tokens (temporal index fixed), producing one per-frame representation; a second, much shallower temporal encoder then attends across frame representations. This is a "late fusion" design, complexity `O((n_h·n_w)² + n_t²)`. This is architecturally very close to the classic 2.5D "per-slice CNN + slice-sequence aggregator" pattern (see §2), just with a ViT swapped in for the per-frame encoder.
3. **Model 3 — factorized self-attention**: keeps Model 1's per-layer structure but inside each block reshapes tokens and applies spatial self-attention then temporal self-attention sequentially. Same asymptotic complexity as Model 2 but interleaves spatial/temporal reasoning at every layer instead of only at the end.
4. **Model 4 — factorized dot-product attention**: splits attention heads — half compute dot-product attention over the spatial axis only, half over the temporal axis only — recombined afterward. Matches Model 2/3's compute while keeping Model 1's parameter count.

Arnab et al. found the factorized encoder (Model 2) gives the best accuracy/compute tradeoff in most regimes and is simplest to implement on top of a pretrained 2D ViT — directly relevant to us since it separates "understand one slice/frame" from "reason across slices," which lets the per-slice encoder be pretrained/reused while the temporal/slice aggregator is cheap and can be trained from scratch on our small labeled set.

### TimeSformer (Bertasius, Wang & Torresani, ICML 2021)
Source: [arXiv:2102.05095](https://arxiv.org/abs/2102.05095), [ICML PMLR version](https://proceedings.mlr.press/v139/bertasius21a.html), [official code](https://github.com/facebookresearch/TimeSformer)

**Divided space-time attention** — within each block, apply temporal attention and spatial attention as two separate, sequential sub-operations rather than one joint attention over all space-time tokens. This was found empirically to be the best of several design choices tested (joint space-time, divided, sparse local/global, axial), reducing complexity to `O(T²S + TS²)` (linear-ish tradeoff vs. `O(T²S²)` for joint attention) while improving accuracy over the joint variant on Kinetics-400/600. Conceptually equivalent to ViViT's Model 3.

### Video Swin Transformer (Liu et al., CVPR 2022)
Source: [openaccess CVPR PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf)

Uses 3D patches of size `2×4×4×3` (T×H×W×C) at the stem, giving `T/2 × H/4 × W/4` tokens, then replaces MHSA with **3D shifted-window** attention: partition tokens into non-overlapping `P×M×M` windows, compute attention locally within each window, and shift the window partition by `(P/2, M/2, M/2)` between consecutive layers so information can propagate across window boundaries without ever paying full global attention cost. Complexity is **linear** in token count because attention is local to a window; the shift schedule is what gives it an effectively growing receptive field across depth. This is the most directly transferable design for us if we go with dense 3D tokens over a whole series (or stacked multi-series volume) rather than factorizing into separate per-plane streams, because slice-count and spatial extent both grow token count fast (§2) and windowed attention caps the quadratic blow-up.

### 3D patch embedding for anisotropic MRI voxels
Knee MRI slice thickness is typically much larger than in-plane pixel spacing (in-plane MRI resolution is commonly sub-millimeter, e.g. ~0.3–0.5 mm, while slice thickness on clinical knee protocols is commonly 3–4 mm — **UNVERIFIED for this specific dataset**; needs to be checked once the 86-tag allowlist and `PixelSpacing`/`SliceThickness` tags are confirmed present, per the project's open questions). This anisotropy (in-plane spacing ≪ slice thickness) is the medical-imaging analogue of video's frame-rate vs. spatial-resolution mismatch, and argues against naively reusing a cubic `t×h×w` tubelet the way ViViT/Video Swin do for RGB video (where the 3 axes are more comparable in "information density"):
- A large tubelet depth (e.g. depth 4+ slices) mixes slices that are already spatially coarse relative to in-plane resolution, discarding information disproportionately along one axis versus the other two — `[derived]`.
- A tubelet depth of 1–2 slices is a more defensible starting point for a series-internal (single-plane) 3D tokenizer, keeping most of the token reduction on the spatial (h, w) axes via a larger in-plane patch size (e.g. 16×16) rather than the slice axis. This should be validated empirically (ablation), not assumed.
- Practical alternative used across the 2.5D knee-MRI literature (MRNet, ELNet — §4) is to treat slice-depth as effectively "tubelet depth = 1" and let a *separate* lightweight aggregator (RNN, max-pool, attention-MIL, or small transformer) handle the slice axis instead of baking it into the patch-embedding tubelet. Given our small labeled set (58 verified studies) and the need for cheap inference (efficiency prize), this 2.5D approach is lower-risk than full 3D tubelet embedding.

---

## 2. 2.5D (slice-encoder + slice-sequence aggregator) vs. full 3D attention — token-count math

**2.5D pattern**: a 2D (ViT or CNN) encoder processes each slice independently (shared weights across slices), producing one embedding per slice (or a small token set per slice); a second, lightweight sequence model (RNN, transformer, or attention-MIL pooling) aggregates across the slice axis to a per-series or per-study prediction. This is exactly what **MRNet** does with a CNN + max-pool (§4), and it is architecturally equivalent to ViViT's **factorized encoder** (Model 2, §1) with the per-frame encoder being 2D and the "temporal" encoder generalized to "slice/series" encoder.

**Full 3D / joint attention**: every patch token from every slice (and every series, if concatenated) attends to every other token in a single self-attention pass — Model 1 in ViViT terms.

### Worked token-count example (values chosen to match this project's data: 256×256 in-plane, 16×16 patches, ~30 slices/series median, 3 series/study — see CLAUDE.md data section)
- Patches per slice: `(256/16) × (256/16) = 16 × 16 = 256` patches/slice.
- Slices per series (median, per project data notes): 30.
- Series per study: 3 (one per plane/sequence, ignoring studies with more than 3 series or missing series — see §3).
- **Total tokens if every slice/series is flattened into one joint sequence** (plus 1 CLS token): `256 × 30 × 3 + 1 = 23,041` tokens.

**Memory of full (joint) self-attention**: the attention score matrix per head per layer is `N × N` where `N = 23,041`. `N² ≈ 5.31 × 10⁸` entries. At bf16 (2 bytes/entry) that is **~1.06 GB per head, per layer, per batch item** just for the attention matrix (before considering Q/K/V projections, MLP activations, or gradients). With even a modest 8–12 attention heads this is 8–13 GB **for one layer's attention scores alone** — this alone exceeds the entire 12 GB training VRAM budget from CLAUDE.md, before any other layer, the batch dimension, or the rest of the model. `[derived — arithmetic from the paper's stated O(N²) complexity, not from a specific benchmark source]`. This is why FlashAttention-style implementations that never materialize the full N×N matrix (§8) are close to mandatory even for testing full-3D attention, and why factorized/windowed variants (§1) or the 2.5D pattern exist in the first place.

**Memory of the factorized-encoder / 2.5D approach**: per-slice spatial attention only ever sees `256` tokens (`256² = 65,536` entries per head/layer — trivial, sub-megabyte), run independently (and batchably) across all 90 slices in the study; the slice/series-sequence aggregator then attends over just `90` slice-embeddings (`90² = 8,100` entries — negligible) or, if pooled per series first, as few as `3` series-embeddings. Total compute is dominated by the (well-understood, cheap) per-slice ViT forward passes, not by any large attention matrix. Complexity: `O(slices × patches²) + O(slices²)` instead of `O((slices × patches)²)` — for our numbers that's roughly `90 × 256² + 90² ≈ 5.9M + 8.1K` vs `5.31 × 10⁸`, a ~90× reduction in attention FLOPs/memory even before considering that the per-slice pass batches trivially.

**Practical implication for this project**: given the 12 GB training GPU and the ≤9 h / runtime-scored Kaggle inference constraint, the 2.5D pattern (or ViViT's Model 2 / factorized-encoder equivalent) is the much safer default; full joint 3D attention over a whole multi-series study is not tractable without aggressive token reduction (windowing, tubelets with large patch/tubelet size, ToMe, or a Perceiver-style latent bottleneck — §8). A middle ground worth ablating: 3D attention *within* a single series (fixing tubelet depth small, per §1) combined with a lightweight cross-series/cross-plane aggregator — this keeps the expensive full-attention token count to ~256×30=7,680 per series (`7,680² ≈ 5.9×10⁷` entries, ~118 MB/head/layer in bf16 — plausible on 12 GB with a handful of heads/layers) rather than ~23K tokens pooled across all series at once.

---

## 3. Multi-series / multi-view fusion strategies

Relevant to us because each study has up to several series spanning 3 planes (sagittal/coronal/axial) × fluid-sensitive/fat-suppressed variants (per `train_series.csv`/`test_series.csv` schema in CLAUDE.md), and **not every study necessarily has every plane/sequence present**.

Fusion strategies documented in the multi-view/multi-modal medical imaging literature:
- **Separate token streams with plane/sequence embeddings** (analogous to positional embeddings, but encoding *which plane/sequence* a token stream came from) fed into a shared or cross-attending encoder. This is the natural fit for a per-series ViT encoder plus a study-level aggregator, and lets the model learn plane-specific relevance per finding (relevant given §4's finding that different planes matter for different labels).
- **Channel stacking**: concatenate multiple planes/sequences as extra input channels at the pixel level. Requires spatial registration/alignment across series, which is generally not guaranteed for independently-acquired MRI series (different FOV, resolution, slice positions) — riskier for this dataset unless series are explicitly co-registered.
- **Late fusion**: process each series fully independently to a per-series prediction, then combine (e.g., logistic regression, as in MRNet — §4) — simplest, most robust to missing series, but forfeits cross-series interaction inside the network.
- **Cross-attention fusion**: cross-view/cross-modal attention lets tokens from one view attend to tokens from another without requiring spatial registration, addressing the "unregistered multi-view" problem explicitly. Example: cross-view transformers for unregistered multi-view medical images ([arXiv:2103.11390](https://arxiv.org/pdf/2103.11390)); XFMamba's cross-fusion approach for multi-view medical classification explicitly notes that "CNNs often employ a weight-sharing mechanism across different view features... limiting their flexibility... adapting to unseen or low-quality medical images," motivating attention-based cross-view fusion instead ([arXiv:2503.02619](https://arxiv.org/html/2503.02619)).
- **Per-series encoders with shared weights**: weight-sharing across series/planes (one CNN/ViT trunk applied independently per series, akin to MRNet's per-plane AlexNets, which in MRNet's case are *not* shared — MRNet trains a **separate** network per series/plane, see §4) is a design choice independent of the fusion mechanism above it; shared weights reduce parameter count and may generalize better across series with different content but risk under-fitting plane-specific features (e.g., ACL visibility is very sagittal-specific).

**Handling missing series**: this is a well-studied "missing modality" problem in medical multi-modal learning ([survey, arXiv:2409.07825](https://arxiv.org/pdf/2409.07825); [systematic review, PMC13099166](https://pmc.ncbi.nlm.nih.gov/articles/PMC13099166/)). Documented strategies, roughly in increasing complexity:
1. **Modality/series dropout during training** — randomly zero out an entire series stream in training so the model learns to function without it; simplest and most directly applicable here.
2. **Shared latent projection** — map each available series into a common representation space so downstream fusion doesn't need a fixed number of inputs.
3. **Ensemble/probabilistic marginalization** — train per-series unimodal models and combine only the available ones at inference (closest in spirit to MRNet's late-fusion logistic regression, naturally missing-series-tolerant).
4. **Generative/imputation methods** — reconstruct a missing series from available ones; higher complexity, probably not worth it for this project given time constraints.
`[Source summary; specific technique-to-outcome numbers are architecture/dataset-specific and were not verified against knee-MRI-specific literature — mark practical effect sizes UNVERIFIED.]`

**Recommendation given our constraints**: token-stream-per-series with a learned plane + sequence-type embedding (derived from `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` — already available in `train_series.csv`/`test_series.csv` per CLAUDE.md, so usable at test time), a shared per-slice 2D encoder (weight-sharing reduces params, helps with our small labeled set), cross-series attention or a small transformer aggregator at the study level, and series dropout during training to handle missing planes gracefully — combines the safest points from each strategy above. `[derived synthesis]`

---

## 4. Knee MRI deep-learning literature

### MRNet (Bien et al., 2018, Stanford)
Sources: [PLOS Medicine paper](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699), [PMC full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6258509/), [project page](https://stanfordmlgroup.github.io/projects/mrnet/)

- Dataset: 1,370 knee MRI exams from Stanford (2001–2012); 1,104 (80.6%) abnormal, 319 (23.3%) ACL tears, 508 (37.1%) meniscal tears.
- Architecture: each exam has 3 series — **sagittal T2, coronal T1, axial PD** (MRNet's specific protocol; may differ from RSNA's dataset). Input per series: `s × 3 × 256 × 256` (s = slice count, 3 = replicated grayscale→RGB channels for a pretrained 2D CNN backbone). Each slice → CNN feature extractor → `s × 256 × 7 × 7`; global average pool → `s × 256`; **max-pool across slices** → single 256-d vector per series; FC layer → per-series probability. The three series-level probabilities are combined via **logistic regression** (not averaging) into one exam-level prediction — i.e., classic late fusion, per-view weighting learned rather than fixed.
- Results (internal validation AUC): abnormality detection 0.937, ACL tear 0.965, meniscal tear 0.847 (95% CIs given in source). External validation (Croatian KneeMRI data, no retraining): ACL AUC 0.824, improving to 0.911 after fine-tuning on external data.
- This CNN-per-slice + max-pool-across-slices + late-fusion-across-series design is the canonical "2.5D" baseline the whole subsequent knee-MRI literature compares against.

### Follow-ups on MRNet
- **A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries using the MRNet Dataset** ([arXiv:2010.01947](https://arxiv.org/pdf/2010.01947)) — benchmarks alternative architectures against the original MRNet baseline on the same dataset.
- **ELNet (Efficiently-Layered Network)**, Tsai et al. 2020 ([arXiv:2005.02706](https://arxiv.org/abs/2005.02706), [code](https://github.com/mxtsai/ELNet)) — a lightweight (<1 MB) CNN trained from scratch (no transfer learning) that matches/beats MRNet using only a **single** imaging stack (axial or coronal alone) rather than all 3 planes, using "Multi-Slice normalization" (slice-independent normalization) and BlurPool anti-aliased downsampling. Directly relevant to the efficiency-prize goal: demonstrates strong accuracy is achievable from far less input than full 3-plane MRNet, at far lower parameter/compute cost.
- **TransMed** (Dai et al. 2021, [PMC8391808](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8391808/), [arXiv:2103.05940](https://arxiv.org/abs/2103.05940)) — described by its authors as the first ViT-based work for medical image classification; hybrid CNN+ViT, tested on both parotid tumor MRI and **knee injury classification** (on MRNet-derived data). Technique: overlay 3 neighboring 2D slices into a 3-channel pseudo-RGB image, patchify, run through a CNN for low-level patch features before a standard ViT for patch-relationship modeling — an interesting hybrid tubelet-like trick for injecting local depth context without a true 3D tubelet. Reported ~1.9% average-accuracy improvement over prior best CNN baselines on the knee task (~10.1% on the parotid task).
- Multi-view ViT for meniscal tear detection: "**Multi-views Knee MRI Classification Using Vision Transformers for Automated Meniscal Tear Detection**" ([Springer, 2026](https://link.springer.com/chapter/10.1007/978-3-032-21585-7_11)) and **MV2SwimNet**, a MobileNetV2+Swin-Transformer hybrid using windowed MHSA and hierarchical representations for meniscus tears ([PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/)) — both recent (2025/2026) confirmations that hybrid CNN+windowed-attention architectures are an active, competitive approach for this exact task family.
- **CTransFuse**: hybrid CNN-Transformer, dual-branch, for meniscus lesion segmentation + classification, reporting AUC 97.45% / AUPR 97.49% on their (non-RSNA) dataset under class imbalance ([PubMed 42184192](https://pubmed.ncbi.nlm.nih.gov/42184192/)) — numbers are dataset-specific and not directly comparable to what we should expect on RSNA's harder, more heterogeneous, multi-site data; cite as evidence of architecture family, not as an accuracy target. `UNVERIFIED as a target for our data.`

### Plane-vs-plane deep learning comparison (directly relevant, 2025)
**"A comparative analysis of sagittal, coronal, and axial MRI planes in diagnosing ACL and meniscal tears via a deep learning model: emphasizing the unexpected importance of the axial plane"** ([PubMed 40606344](https://pubmed.ncbi.nlm.nih.gov/40606344/), [full text, QIMS](https://qims.amegroups.org/article/view/137915/html)) — trained "TripleMRNet" on the MRNet dataset (1,130 train / 120 val cases) on all 7 combinations of the 3 planes:
- **ACL tears**: sagittal alone was the strongest single plane, but axial alone also performed strongly (i.e. axial is not just a redundant addition); the 3-plane combined model was best overall (accuracy 0.925, sensitivity 0.944, specificity 0.909, F1 0.919).
- **Meniscal tears**: **axial outperformed both other planes**, and **sagittal — the conventionally "primary" plane for menisci — performed worst alone** in this deep-learning comparison. This contradicts naive radiology-textbook priors (see next section) and is worth treating as an empirical prompt to *not* hard-code plane relevance, but to let a learned per-plane weighting (as in MRNet's logistic-regression fusion, or a learned per-plane attention gate) discover it from data.

### KneeMRI dataset (Šatjduhar et al.)
Source: original paper Stajduhar, M. et al., *Computer Methods and Programs in Biomedicine* 140:151–164 (2017); dataset description via [PMC12480240](https://pmc.ncbi.nlm.nih.gov/articles/PMC12480240/) and other citing works.
- 917 volumes (Clinical Hospital Centre Rijeka, Croatia, Siemens Avanto 1.5T, PD-weighted fat-suppressed, 2006–2014). Single sagittal series per exam, 320×320 or 290×300 in-plane, 21–45 slices (mean 31, SD 2.27).
- 3-class ACL label: healthy (690, 75.2%), partial tear (172, 18.8%), complete rupture (55, 6%).
- Used above as MRNet's external validation set (Bien et al.), and independently in several ACL-classification papers.

### fastMRI+ (knee)
Source: [Nature Scientific Data](https://www.nature.com/articles/s41597-022-01255-z), [arXiv:2109.03812](https://arxiv.org/abs/2109.03812), [PMC8983757](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8983757/)
- 16,154 bounding-box annotations + study-level labels across 22 pathology categories, annotated on all 1,172 fastMRI knee raw-data studies. Each exam has a **single coronal series** (PD or T2-weighted), slice-by-slice bounding boxes. Designed originally for MRI-reconstruction research (pairing raw k-space with clinical annotations), not a pure classification benchmark, but a source of pathology-location ground truth if useful for auxiliary supervision or sanity-checking a MIL attention map. Coronal-only, single-series-per-study — a caveat if used as pretraining/auxiliary data given the multi-series nature of our target task.

### OAI / MOAKS-based OA grading
- Deep-learning morphological phenotyping at scale on OAI data: [Nature Sci Rep](https://www.nature.com/articles/s41598-021-90292-6).
- MOAKS-based structured MRI feature models for predicting joint-space-narrowing progression report ROC AUC 0.70–0.76 across 2–8 year horizons using a multimodal transformer ([Frontiers](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1908547/full)) — note this is a *progression prediction* task (longitudinal), not single-timepoint classification, so not directly comparable to our per-study binary OA labels, but useful as an existence proof that transformer architectures work on OAI-scale MRI cohorts.
- General DL OA-grading accuracy from MOAKS-trained binary classifiers on cartilage/meniscal damage: reported accuracy ~0.75–0.83 despite class imbalance ([Frontiers, standard-radiograph-to-MRI-inference study](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1858407/full)). `UNVERIFIED as directly transferable to our 12-label multi-task, different-cohort setting — cite as rough plausibility range only.`

### Which plane/sequence matters per finding — radiology sources
Cross-checked against standard knee MRI protocol guidance and pathology-specific radiology literature (not a single unified paper — synthesized across sources below):
| Finding | Best plane(s) / sequence(s) | Source |
|---|---|---|
| ACL | Sagittal is standard-of-reference; axial also useful (and per the 2025 DL study above, axial alone can rival sagittal) | [Applied Radiology, imaging the knee ligaments](https://appliedradiology.com/articles/imaging-the-knee-ligaments); [PMC9127290](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9127290/); [DL comparative study](https://qims.amegroups.org/article/view/137915/html) |
| MCL (collateral ligaments) | Coronal | [Applied Radiology](https://appliedradiology.com/articles/imaging-the-knee-ligaments) |
| Medial / Lateral Meniscus | Classically sagittal *and* coronal (standard protocol emphasizes both); but the 2025 DL comparative study found **axial** markedly outperformed sagittal/coronal for meniscal tear detection in a deep model — a genuine conflict between classic radiology teaching and empirical DL results worth flagging, not resolving by assumption | [standard protocol guidance](https://appliedradiology.com/articles/imaging-the-knee-ligaments); [DL comparative study, axial superiority](https://qims.amegroups.org/article/view/137915/html) |
| PF (patellofemoral) OA | Axial for patellar cartilage; sagittal for trochlear cartilage — i.e., **both matter, for different sub-structures** of the same finding | [PMC2919651, systematised MRI approach to PFJ](https://pmc.ncbi.nlm.nih.gov/articles/PMC2919651/) |
| Medial/Lateral OA (tibiofemoral) | Not separately sourced here — general OA cartilage assessment typically draws on sagittal + coronal fluid-sensitive sequences per standard protocol; `UNVERIFIED — no compartment-specific tibiofemoral-OA plane study was found` in this pass. |
| Effusion | Fluid-sensitive sequences (T2/PD-FS or STIR) in any plane; sagittal and axial both explicitly documented as effective | [mrimaster.com Baker's cyst rupture](https://mrimaster.com/bakers-cyst-rupture-mri/) |
| Synovitis | Contrast-enhanced MRI is the clinical reference standard for synovitis (direct synovial enhancement); **non-contrast fluid-sensitive sequences (qDESS shown comparable to contrast-enhanced for mild synovitis: sensitivity 1.00, specificity 0.909; moderate+: sensitivity 0.727, specificity 1.00) are the best non-contrast proxy** | [Arthritis Research & Therapy, qDESS vs contrast-enhanced](https://arthritis-research.biomedcentral.com/articles/10.1186/s13075-021-02436-8) |
| Baker's (popliteal) cyst | Sagittal T2-FS and axial T2 both documented as effective; STIR highly sensitive to fluid, useful especially for ruptured cysts | [mrimaster.com](https://mrimaster.com/bakers-cyst-rupture-mri/); [PMC10540859](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10540859/) |
| Contusion (bone marrow edema) | Fat-suppressed fluid-sensitive sequences (FS-T2, PD-FS, or STIR) essential — non-fat-suppressed/T1 sequences can miss or underestimate lesion extent | [Radiology 1993, fat-sat FSE bone contusion detection](https://pubs.rsna.org/doi/10.1148/radiology.189.3.8234723); [AJR, coronal FS T2 vs STIR](https://www.ajronline.org/doi/10.2214/ajr.166.1.8571859) |
| Fracture | Not separately sourced in this pass beyond the general point that fat-suppressed fluid-sensitive sequences reveal marrow edema patterns associated with occult fracture; `UNVERIFIED — no fracture-specific plane study located`. |

**Practical takeaway**: `Fluid_Sensitive`/`Fat_Suppression`/`Anatomical_Plane` (available for both train and test per CLAUDE.md) are legitimately informative per-series conditioning signals, consistent with radiology practice — but the 2025 axial-plane finding above is a reminder not to hard-code "sagittal is best for menisci" as an architectural prior (e.g., don't drop axial series to save compute for meniscus-related labels) — let a learned per-plane attention/gating mechanism decide, and ablate.

---

## 5. Multiple-instance learning / attention-MIL pooling (Ilse et al. 2018)

Source: [ICML PMLR](https://proceedings.mlr.press/v80/ilse18a.html), [arXiv:1802.04712](https://arxiv.org/pdf/1802.04712), [official PyTorch code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)

Formulates multiple-instance learning (a bag of instances → one bag-level label) with a **permutation-invariant, learned attention-weighted pooling** operator instead of fixed pooling (mean/max). Because attention weights are produced per-instance, they double as an interpretability signal (which instances — here, which slices — drove the prediction), which is valuable for both debugging and, potentially, weak localization within a study.

Direct relevance to this project: our labels are **study-level** (12 binary findings per study), and a study is naturally a "bag" of slice instances (or series instances) with no verified per-slice ground truth. Attention-MIL pooling is a strict theoretical/practical upgrade over MRNet's fixed max-pool-across-slices (§4) as the slice-aggregation layer, and it is a more classical, cheaper-to-train alternative to a full learned transformer aggregator over slices when data is scarce (58 verified-label studies) — worth having as a baseline aggregator alongside a lightweight transformer aggregator (which is itself, per §1, ViViT's "factorized encoder" idea generalized). `[derived synthesis connecting Ilse et al. to this project's label structure]`

---

## 6. Medical 3D transformer architectures: UNETR, Swin UNETR, MedViT, TransMed

### UNETR (Hatamizadeh et al., WACV 2022)
Source: [arXiv:2103.10504](https://arxiv.org/abs/2103.10504), [CVF WACV PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf)
- Reformulates 3D **segmentation** as sequence-to-sequence prediction: the 3D input volume is split into uniform non-overlapping 3D patches, linearly projected, and passed through a pure transformer encoder; a CNN-style decoder reconnects to the encoder at multiple resolutions via skip connections (U-Net-shaped, transformer-encoder / CNN-decoder hybrid) to produce the segmentation map.
- This is a **segmentation**, not classification, architecture — not directly reusable for our per-study 12-label classification task, but its patch-embedding-of-a-3D-volume approach is a reference point for how much of a knee volume a single 3D-ViT patch embedding stage can practically ingest, and its skip-connection decoder pattern would matter if we ever add a segmentation auxiliary task (e.g., weak meniscus/ACL localization) — not currently in scope per CLAUDE.md's labeling plan.

### Swin UNETR (also Hatamizadeh et al. lineage, MICCAI BrainLes workshop 2022)
Source: [arXiv:2201.01266](https://arxiv.org/abs/2201.01266)
- Same sequence-to-sequence 3D segmentation reformulation as UNETR, but swaps the plain ViT encoder for a **hierarchical Swin (shifted-window) transformer**, giving multi-scale features "for free" via the pyramid structure plus linear-complexity local attention (§1's Video Swin windowing idea, applied to static 3D medical volumes rather than video). Outperformed CNN baselines and ranked among top models in the BraTS 21 segmentation validation phase. Same caveat as UNETR: it's a segmentation architecture, but the Swin-hierarchical-encoder-for-3D-medical-volumes pattern is directly reusable as our per-series (or per-study) *classification* backbone if we adopt windowed attention — feed the Swin-hierarchical encoder's final [class]-token-equivalent (e.g. global-pooled last stage) into a classification head instead of a segmentation decoder.

### MedViT (Manzari et al., Computers in Biology and Medicine 2023)
Source: [arXiv:2302.09462](https://arxiv.org/abs/2302.09462), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482523002561), [code](https://github.com/Omid-Nejati/MedViT)
- Hybrid CNN+Transformer for 2D medical image **classification** — convolutional stages give locality/inductive bias, transformer stages give global connectivity, in a hierarchical pyramid. Includes a "patch moment changer" augmentation (permuting feature mean/variance across mini-batch) specifically to improve robustness/generalization on small, heterogeneous medical datasets — evaluated on MedMNIST-2D (12 datasets). Relevant less as an architecture to copy wholesale and more as a validated instance of the "CNN stem/stages + transformer stages" hybrid pattern outperforming pure ViT on small/medical data, reinforcing that a pure from-scratch ViT will likely need either large data augmentation, strong regularization, or a light convolutional stem to be competitive given our small labeled set — consistent with the small-dataset-ViT literature below.

### TransMed
Covered in §4 (parotid tumor + knee injury classification) — the earliest ViT-for-medical-classification work located, and notably already validated on a knee task.

### 3D ViT classifiers on MRI / small-data considerations
- General finding across multiple 2023–2026 papers: pure ViTs lack CNNs' locality inductive bias and are data-hungry, tending to overfit on small medical datasets unless mitigated ([MDViT, arXiv:2307.02100](https://arxiv.org/pdf/2307.02100); [depth-wise-conv-in-ViT, arXiv:2407.19394](https://arxiv.org/pdf/2407.19394); [CoMViT, arXiv:2510.27442](https://arxiv.org/html/2510.27442)).
- Mitigations documented: convolutional tokenizer/stem instead of a plain linear patch embed; shifted patch tokenization (SPT) + locality self-attention (LSA), reported to let a ViT "learn from scratch even on small datasets"; multi-domain knowledge aggregation/adapters; compact architectures with 5–20× parameter reduction (CoMViT) without accuracy loss.
- Given this project's **58 verified-label studies** and reliance on pseudo-labeling / report-derived labels to scale up (per CLAUDE.md's labeling plan), the small-data-ViT literature is directly actionable: favor a convolutional (or hybrid) patch stem over a naive linear patch embed, and treat aggressive regularization / strong augmentation as necessary rather than optional, at least until the enlarged pseudo-labeled/report-derived training set is validated.

---

## 7. Laterality: is horizontal flip safe? — reasoned by anatomy, not directly sourced

No single source addresses knee-MRI-specific flip augmentation directly; the general medical-imaging-augmentation literature confirms the underlying hazard class ("laterality loses meaning when mirrored... the most common bug with flips is label semantics") ([synthesis of medical augmentation guidance](https://blog.roboflow.com/how-flip-augmentation-improves-model-performance/); general principle also echoed for retinal and cardiac imaging in the same search pass). Because no source addressed knee MRI specifically, the following is `[derived]` from anatomy and DICOM plane geometry, and should be verified against actual `ImageOrientationPatient`/`ImagePositionPatient` conventions once confirmed present in the allowed 86 DICOM tags (per CLAUDE.md's open question #3) before being trusted in code.

**Setup**: a standard patient/DICOM coordinate frame has three anatomical axes — left–right (LR, = medial–lateral within a single leg), anterior–posterior (AP), and superior–inferior (SI). For each of the three acquisition planes, two axes lie *in-plane* and one is the *through-plane* (slice-stacking) axis:
- **Axial**: in-plane = LR, AP; through-plane (slice order) = SI.
- **Coronal**: in-plane = LR, SI; through-plane (slice order) = AP.
- **Sagittal**: in-plane = AP, SI; through-plane (slice order) = LR.

**Key anatomical fact used**: the human body has approximate bilateral mirror symmetry across the LR axis (a mirrored left knee looks like a plausible right knee), but has **no such symmetry across the AP or SI axes** (a mirrored anterior-posterior or superior-inferior knee does not look like a real, differently-oriented knee — the patella, femoral condyles, tibial plateau, etc. have fixed, non-mirror-symmetric AP/SI structure).

**Consequence for our 12 labels**: crucially, `Medial Meniscus`/`Lateral Meniscus` and `Medial OA`/`Lateral OA` labels are anatomical (defined relative to the body's midline), not defined relative to "which side of the image." Mirroring a knee along the LR axis turns a left knee into a (synthetic) right knee, but a structure that was medial stays medial and a structure that was lateral stays lateral — it just now appears on the opposite side of the *image*. Since our labels are attached to the study, not to an image-column position, an LR-axis mirror does not require relabeling. Also note: left-vs-right knee identity itself is **not** one of the 12 prediction targets, so there is no separate "laterality label" that a flip could invalidate.

**Conclusions per augmentation axis** `[derived]`:
| Plane | In-plane LR flip | In-plane UD (AP or SI) flip | Slice-order reversal |
|---|---|---|---|
| **Axial** (in-plane = LR & AP) | **Safe** — mirrors medial/lateral, no relabel needed | **NOT safe** if this is an AP flip — no AP mirror symmetry exists | Reverses SI slice order — **NOT safe**, no SI mirror symmetry |
| **Coronal** (in-plane = LR & SI) | **Safe** — mirrors medial/lateral, no relabel needed | **NOT safe** if this is an SI flip — no SI mirror symmetry | Reverses AP slice order — **NOT safe**, no AP mirror symmetry |
| **Sagittal** (in-plane = AP & SI) | N/A — LR is not an in-plane axis here | **NOT safe either way** — flipping AP or SI in-plane axes has no mirror symmetry to justify it | Reverses LR slice order — **Safe**, equivalent to the same LR mirror argument as above (this is how you "flip" a sagittal series, since LR is its through-plane axis) |

**Practical caveat before implementing**: this table assumes the in-plane row/column axes can be reliably mapped to LR/AP/SI per study. That mapping is *not* guaranteed from pixel-array orientation alone — it depends on the DICOM `ImageOrientationPatient` direction cosines (and patient positioning, e.g. supine vs prone, feet-first vs head-first), which is exactly the tag CLAUDE.md flags as `TODO verify` (whether orientation tags survive the 86-tag allowlist). **Do not hard-code "column axis = LR" without checking `ImageOrientationPatient` per study first** — get this wrong and the "safe" flip table above silently becomes unsafe. `[derived, unverified against actual per-study DICOM tags]`

---

## 8. Efficient attention for long token sequences

Directly relevant given §2's token-count math (a naive joint-attention approach blows past the 12 GB training budget, and the efficiency prize explicitly scores end-to-end Kaggle-notebook inference runtime per CLAUDE.md).

- **FlashAttention / PyTorch SDPA**: an IO-aware exact-attention algorithm that never materializes the full `N×N` attention matrix, instead tiling the computation and using online softmax + selective recomputation, giving reported **2–4× speedups and 10–20× memory reduction** versus naive attention ([summary from search, consistent across sources](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)). Available natively via `torch.nn.functional.scaled_dot_product_attention` since PyTorch ≥2.2, which auto-selects among FlashAttention-2, memory-efficient attention, or a C++ fallback kernel depending on input shapes ([PyTorch docs](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html)). This is a **free, exact** (not approximate) optimization — there is essentially no reason not to route all attention through SDPA rather than a hand-rolled softmax(QKᵀ)V, both for training memory headroom and for inference latency (efficiency prize).
- **Windowed attention** (Swin/Video Swin, §1 and §6): local attention within fixed windows plus a shift schedule for cross-window propagation, giving **linear** (not quadratic) complexity in token count. Directly applicable if we go with dense 3D or long joint-token sequences rather than the 2.5D pattern.
- **Token Merging (ToMe)**, Bolya et al., ICLR 2023 ([arXiv:2210.09461](https://arxiv.org/abs/2210.09461), [code](https://github.com/facebookresearch/ToMe)): a **training-free** (can also be used during training) method that merges the most similar tokens (via key-vector similarity) at each layer, reported to **2× throughput on ViT-L/H at high resolution** and **2.2× on video ViT-L**, with only 0.2–0.3% accuracy drop; also usable during MAE fine-tuning for up to 2× training speedup. Attractive as a drop-in inference-time optimization for the efficiency prize with minimal implementation risk since it requires no retraining if applied only at inference (though better results come from training-time integration).
- **Token pruning**: dynamically drops (rather than merges) low-importance tokens; complexity reduction is similar in spirit to ToMe but discards rather than preserves information via averaging — general surveys note attention cost grows quadratically with token count, motivating pruning particularly as input resolution/patch count grows ([survey collection](https://github.com/ZLKong/Awesome-Collection-Token-Reduction)). ToMe is generally reported as more accurate than pruning at equivalent speedup because merging preserves information via averaging rather than discarding it outright ([ToMe paper](https://arxiv.org/pdf/2210.09461)).
- **Perceiver / Perceiver IO** (Jaegle et al., ICML 2021 / arXiv:2107.14795): cross-attends a large input token set into a **much smaller, fixed-size latent array**; all subsequent self-attention happens only within that small latent array, decoupling compute from raw input size — complexity becomes **linear** in input token count (only the initial cross-attention touches all input tokens; each such cross-attention is `O(input_tokens × latents)`, not `O(input_tokens²)`) ([arXiv:2103.03206 Perceiver](https://arxiv.org/abs/2103.03206), [Perceiver IO, arXiv:2107.14795](https://arxiv.org/pdf/2107.14795)). This is the most aggressive of the options here and is a strong candidate if we ever want a *single* architecture that ingests all series/slices/tokens of a study jointly (rather than the factorized 2.5D approach) while staying tractable — the latent bottleneck plays a role structurally similar to ViViT's Model 2 "temporal encoder" or an attention-MIL pooling layer (§5), but as a general, learnable, input-agnostic mechanism rather than one tailored to a fixed slice/series axis.

**Runtime/memory implication ranking for this project** `[derived synthesis]`: (1) route everything through SDPA/FlashAttention regardless of the higher-level architecture choice — it's free; (2) prefer the 2.5D / factorized-encoder pattern as the primary way to control token count (§2), since it sidesteps needing exotic attention variants for the expensive part of the sequence; (3) if a fuller 3D/joint-token design is pursued for accuracy reasons, add windowed attention (Swin-style) or a Perceiver-style latent bottleneck as the mechanism that makes it tractable; (4) treat ToMe as a cheap inference-time lever specifically for the efficiency-prize runtime term, applicable on top of whichever architecture is chosen, with the caveat that any accuracy loss trades directly against the AUC term of the efficiency formula (`Efficiency = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400` per CLAUDE.md) — so it should be tuned once both AUC and runtime are being measured together, not before.

---

## Recommended architecture for our knee ViT (with token budget math)

Given: 12 GB training VRAM, from-scratch ViT as primary goal with a pretrained-backbone fallback, 58 verified-label studies scaling via pseudo-labeling/report-derived labels, ≤9 h Kaggle inference budget with runtime scored for the efficiency prize, ~1,300 test studies, and per-study series counts/plane availability that may vary.

**Recommended pattern: 2.5D factorized-encoder (ViViT Model-2-style) with attention-MIL / lightweight-transformer aggregation, per-series plane+sequence conditioning, and series dropout for missing planes.**

1. **Per-slice encoder** (shared weights across all slices and, at least initially, across all series/planes): a ViT-Small/Base-scale encoder with a **convolutional patch stem** rather than a naive linear patch embed (per §6's small-data-ViT literature), operating on individual 2D slices at, e.g., 256×256 with 16×16 patches → 256 tokens/slice + 1 [CLS] → attention cost per slice is `257² ≈ 66K` entries/head/layer — trivial, and this pass batches cheaply across all ~90 slices in a study (or across a training minibatch of studies).
2. **Per-slice output**: take the [CLS] token (or a small pooled token set) per slice as its embedding, tagged with a learned embedding of that slice's series' `Anatomical_Plane` + `Fluid_Sensitive` + `Fat_Suppression` metadata (available at test time per CLAUDE.md) plus a slice-position embedding within its series.
3. **Study-level aggregator**: feed all slice embeddings from all series present in the study (median ~90, but variable — handle missing series naturally since this is just "fewer tokens in the bag") into either (a) an attention-MIL pooling layer (Ilse et al., §5) for a cheap, interpretable, small-data-friendly baseline, or (b) a small transformer encoder over the slice-embedding sequence (ViViT's factorized "temporal" encoder equivalent) for a higher-capacity option once enough (pseudo-)labeled data exists to support it — attention cost here is at most `~90² ≈ 8,100` entries/head/layer, negligible. Train and compare both; they are cheap enough to ablate against each other directly.
4. **Head**: linear → 12 logits → sigmoid + BCE (per CLAUDE.md, already settled).
5. **Missing-series handling**: series dropout during training (randomly drop an entire series' slice-embeddings from the bag) so the aggregator learns to function with whatever subset of planes a given study provides — cheapest, most directly applicable strategy from §3.
6. **Attention implementation**: route every attention call through `torch.nn.functional.scaled_dot_product_attention` (SDPA/FlashAttention) — no architecture cost, direct memory/speed win, matters both for training on 12 GB and for the efficiency-prize runtime term.
7. **If accuracy plateaus and a richer spatial-temporal signal is needed inside a single series** (e.g., sagittal series specifically, where adjacent-slice context plausibly helps ACL/meniscus visibility): consider a small tubelet depth (2, ablate) *within that one series only* (never across series, and never across the whole study), attention-windowed (Swin-style, §1/§6) to keep cost bounded — this stays within the ~7,680-token-per-series full-attention regime computed in §2 (`~118 MB/head/layer` in bf16), which is plausible on 12 GB with careful batch-size/gradient-accumulation tuning, rather than the ~23K-token whole-study joint-attention regime which is not (`~1 GB/head/layer` from §2's math, exceeding budget once multiplied across heads and layers).
8. **What to explicitly avoid as a first attempt**: joint full self-attention over all series/slices/patches of a study concatenated into one sequence (~23K tokens per §2) — the memory math above shows this alone can exceed the entire 12 GB budget in a single attention layer, before considering the rest of the network, batch size, or gradient memory. If full 3D/joint modeling is desired later for an ablation, gate it behind FlashAttention/SDPA plus either windowed attention or a Perceiver-style latent bottleneck (§8), not naive dense attention.
9. **Efficiency-prize interaction**: once the ViT (or pretrained-backbone fallback) reaches a stable AUC, apply Token Merging (ToMe, §8) as a first-pass, low-risk inference-time speedup (training-free option available) before considering mixed precision/quantization (already planned per CLAUDE.md) — re-measure both AUC and wall-clock runtime together, since the efficiency formula trades one against the other.

`[This recommendation section is a derived synthesis of the sourced material above, tailored to this project's stated hardware/data/labeling constraints from CLAUDE.md — not itself independently sourced.]`

---

## Sources

- ViViT: [arXiv:2103.15691](https://arxiv.org/abs/2103.15691) · [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2103.15691) · [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html)
- TimeSformer: [arXiv:2102.05095](https://arxiv.org/abs/2102.05095) · [ICML PMLR](https://proceedings.mlr.press/v139/bertasius21a.html) · [code](https://github.com/facebookresearch/TimeSformer)
- Video Swin Transformer: [CVPR 2022 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf)
- MRNet: [PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6258509/) · [project page](https://stanfordmlgroup.github.io/projects/mrnet/)
- MRNet comparative follow-up: [arXiv:2010.01947](https://arxiv.org/pdf/2010.01947)
- ELNet: [arXiv:2005.02706](https://arxiv.org/abs/2005.02706) · [code](https://github.com/mxtsai/ELNet)
- TransMed: [PMC8391808](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8391808/) · [arXiv:2103.05940](https://arxiv.org/abs/2103.05940)
- Multi-view ViT meniscal tear: [Springer 2026](https://link.springer.com/chapter/10.1007/978-3-032-21585-7_11)
- MV2SwimNet: [PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/)
- CTransFuse: [PubMed 42184192](https://pubmed.ncbi.nlm.nih.gov/42184192/)
- Plane-comparison DL study (axial importance): [PubMed 40606344](https://pubmed.ncbi.nlm.nih.gov/40606344/) · [full text QIMS](https://qims.amegroups.org/article/view/137915/html)
- KneeMRI dataset background: [PMC12480240](https://pmc.ncbi.nlm.nih.gov/articles/PMC12480240/)
- fastMRI+: [Nature Sci Data](https://www.nature.com/articles/s41597-022-01255-z) · [arXiv:2109.03812](https://arxiv.org/abs/2109.03812) · [PMC8983757](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8983757/)
- OAI morphological phenotyping: [Nature Sci Rep](https://www.nature.com/articles/s41598-021-90292-6)
- MOAKS progression transformer: [Frontiers](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1908547/full)
- Standard radiograph→MRI OA inference: [Frontiers](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1858407/full)
- Knee ligament imaging planes: [Applied Radiology](https://appliedradiology.com/articles/imaging-the-knee-ligaments) · [PMC9127290](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9127290/)
- Patellofemoral joint MRI planes: [PMC2919651](https://pmc.ncbi.nlm.nih.gov/articles/PMC2919651/)
- Bone contusion/marrow edema imaging: [Radiology 1993](https://pubs.rsna.org/doi/10.1148/radiology.189.3.8234723) · [AJR](https://www.ajronline.org/doi/10.2214/ajr.166.1.8571859)
- Baker's cyst imaging: [mrimaster.com](https://mrimaster.com/bakers-cyst-rupture-mri/) · [PMC10540859](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10540859/)
- Synovitis qDESS vs contrast-enhanced: [Arthritis Research & Therapy](https://arthritis-research.biomedcentral.com/articles/10.1186/s13075-021-02436-8)
- Ilse et al. attention-MIL: [arXiv:1802.04712](https://arxiv.org/pdf/1802.04712) · [ICML PMLR](https://proceedings.mlr.press/v80/ilse18a.html) · [code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
- UNETR: [arXiv:2103.10504](https://arxiv.org/abs/2103.10504) · [WACV PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf)
- Swin UNETR: [arXiv:2201.01266](https://arxiv.org/abs/2201.01266)
- MedViT: [arXiv:2302.09462](https://arxiv.org/abs/2302.09462) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482523002561) · [code](https://github.com/Omid-Nejati/MedViT)
- Small-dataset ViT training: [MDViT arXiv:2307.02100](https://arxiv.org/pdf/2307.02100) · [Depth-wise-conv ViT arXiv:2407.19394](https://arxiv.org/pdf/2407.19394) · [CoMViT arXiv:2510.27442](https://arxiv.org/html/2510.27442)
- Missing-modality handling: [survey arXiv:2409.07825](https://arxiv.org/pdf/2409.07825) · [systematic review PMC13099166](https://pmc.ncbi.nlm.nih.gov/articles/PMC13099166/)
- Cross-view fusion (unregistered): [arXiv:2103.11390](https://arxiv.org/pdf/2103.11390) · [XFMamba arXiv:2503.02619](https://arxiv.org/html/2503.02619)
- Medical flip-augmentation laterality risk (general principle): [Roboflow blog on flip augmentation](https://blog.roboflow.com/how-flip-augmentation-improves-model-performance/)
- FlashAttention / SDPA: [PyTorch SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) · [PyTorch SDPA docs](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html)
- Token Merging (ToMe): [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) · [code](https://github.com/facebookresearch/ToMe)
- Token pruning survey: [Awesome-Collection-Token-Reduction](https://github.com/ZLKong/Awesome-Collection-Token-Reduction)
- Perceiver / Perceiver IO: [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) · [arXiv:2107.14795](https://arxiv.org/pdf/2107.14795)
