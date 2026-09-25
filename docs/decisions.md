# Decisions log

Every decision Dom has made, dated, with the reason. **Future sessions: don't re-litigate these.**
If new evidence makes one worth revisiting, ask Dom and add a new dated entry; never edit old ones.

| Date | Decision | Why / evidence | Source |
|---|---|---|---|
| 2026-09-23 | Primary architecture = Vision Transformer built from scratch; pretrained fine-tuning is the fallback | Performance/consistency view + learning value + write-up value | CLAUDE.md [Dom] |
| 2026-09-23 | Start in float32, get accuracy first, then shrink for efficiency | Efficiency is scored on runtime, but AUC comes first | CLAUDE.md [Dom] |
| 2026-09-23 | No grayscale inversion by default; only within the same scan type (ablation) | FS vs PD-FS signal differ | CLAUDE.md [Dom] |
| 2026-09-23 | "Injury type encoding" idea dropped | It is the task itself | CLAUDE.md [Dom] |
| 2026-09-24 | **Claude writes the code, Dom reviews** | Speed to submission; learning via review guides + learning notes | S00 AskUserQuestion |
| 2026-09-24 | **Run a pretrained-backbone track (Track A) in parallel from the start**, sharing the data pipeline with the scratch-ViT track (Track B) | Public notebooks already score 0.941–0.947 with pretrained backbones; 28 days left | S00 AskUserQuestion |
| 2026-09-24 | **Future sessions run Claude Code on the Windows box** | Next to the data and GPU | S00 AskUserQuestion |
| 2026-09-24 | **Code syncs via the private GitHub repo `dominictedrow/knees`** | Versioned handoffs between machines/sessions | S00 AskUserQuestion |
| 2026-09-24 | **The Windows box can train 24/7 until Oct 22** | Makes multi-day MAE pretraining feasible | S00 AskUserQuestion |
| 2026-09-24 | All hyperparameters are tuned later; plans hold placeholders only | Dom's instruction | S00 prompt |
| 2026-09-24 | **Labels: report-derived first.** Main labels come from LLM report readers. The 58 gold studies fit only a tiny per-finding calibration (e.g. effusion threshold, Effusion→Synovitis fill) and are scored out-of-fold with bootstrap CIs. Dom's 58-teacher idea is deferred to a later **noisy-student** round (a model trained on report labels relabels the corpus from images, blended with the report labels) | A 58-study image model scored 0.66 macro AUC vs 0.87–0.89 for LLM report labels; tuning on the 58 hurt the LB (docs/kaggle/labels_from_notebooks.md; docs/research/ssl_and_labels.md §7) | S00 AskUserQuestion |
| 2026-09-24 | **Label sources: our own LLM reader + vetted public label sets, blended.** Public sets only after a license check; **exclude any set with gold answers pasted in** (yunusgmsoy v5, barun2104) and never validate with them | Blending decorrelated readers measurably helped | S00 AskUserQuestion |
| 2026-09-24 | **Our own LLM reader = Claude API** (key in `.env` on the Windows box, never committed) | Host allows commercial LLM APIs for labeling (forum 733965); strong multilingual reading; cheap for ~4.4k reports | S00 AskUserQuestion |
| 2026-09-24 | **"From scratch" (Track B) = our own code + our own weights**: every module hand-written, random init, then MAE self-pretraining on competition slices only. No external pretrained weights in Track B | Purest learning/write-up story; MAE self-pretraining is the evidence-backed way to make it viable | S00 AskUserQuestion |
| 2026-09-24 | **Final 2 submissions = 1 accuracy (best-AUC ensemble) + 1 efficient (single fast model at fp16)** | The efficiency prize needs a selected final submission; winners pick categorically faster models | S00 AskUserQuestion |
| 2026-09-24 | **Solo** (no team merge) | Dom's choice | S00 AskUserQuestion |
| 2026-09-24 | **Claude API labeling budget ≤ ~$50** in total (incl. re-runs and the noisy-student round); pilot 100 reports first to measure cost | Dom's choice | S00 AskUserQuestion |
| 2026-09-24 | **Training precision: bf16 autocast from day one** (fp32 master weights) in both tracks. Supersedes "start in float32" for *training*; "shrink later" now means inference only (fp16 on T4 → quantization) | 12 GB VRAM; research + public practice | S00 AskUserQuestion |
| 2026-09-24 | **Slice input default C=3: three physically adjacent slices of the same series as channels (2.5D)**; C=1 is an ablation. Refines checklist #11 (still grayscale; not plane stacking) | All strong public arms use 2.5D triplets; cheap local depth context | S00 AskUserQuestion |
| 2026-09-24 | **Cache v1 stores every slice of every series at 384 px** (130 mm crop, uint8, ≈120 GB) on a fast internal drive of the Windows box | Matches Track A's proven 384-px CoAtNet; Track B/MAE downsample at load; re-caching costs hours, downsampling costs nothing | S02 AskUserQuestion |
