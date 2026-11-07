# xAI Workflow in Instana

## Overview

xAI's Grok models are called at two distinct stages of the Instana pipeline: **visual frame classification** (Grok vision) and **language generation** (Grok text). Neither stage invents numbers — all engagement rates, rankings, and performance deltas are pure arithmetic computed from Instagram Graph API data. Grok is only responsible for what it can genuinely do: describe what it sees and explain patterns in plain language.

---

## Stage 1 — Video Frame Analysis (Vision)

### When
After a reel has been downloaded and its frames extracted, the frame analyzer runs as part of the per-video pipeline. It is triggered once per reel, during the `grok_analysis` step of the scan job.

### Where
`src/server/pipeline/grokAnalyzer.ts` → `analyzeFrames()`

Called from `src/server/pipeline/pipelineOrchestrator.ts` after:
1. Video download
2. Frame extraction (20 evenly-spaced frames per video)
3. Audio detection
4. OCR text extraction (Tesseract, first 3 frames)
5. Edit-cut detection (frame differencing) — `cutsPerSecond` measured here, before Grok runs

### How

**Model:** `grok-4.3` (vision)
**Temperature:** `0`
**Max tokens:** `1500`
**Frames sent:** 16 (≥30s video) · 10 (15–30s) · 6 (<15s), each resized to 512×512 JPEG

Each frame is base64-encoded and passed as an `image_url` content part alongside the prompt below. The frame images are prepended to the user message in order; the text prompt follows at the end.

`cutsPerSecond` is computed by `detectCutsPerSecond` before `analyzeFrames` is called. It must be passed into `analyzeFrames` as a parameter and included in the prompt context — Grok must cross-check its pacing classification against this objective measurement.

---

### System Message

```
You are a video format classifier. Return ONLY valid JSON matching the schema. No markdown, no explanation.
```

---

### User Message

```
You are classifying an Instagram Reel to identify the recurring production template it belongs to.
Your primary goal is to recognise format consistency — two videos shot with the same setup, same
background, and same text overlay style are the SAME FORMAT even if their topics differ completely.

Structural and visual production choices take priority over semantic content. A talking-head tutorial
and a talking-head motivational post filmed in the same location with the same framing and text layout
are one format, not two. Do not allow differences in topic, spoken content, or hook wording to
override structural similarity.

Frames provided: {N} equally-spaced stills across a {duration}s video ({frame labels}).
OCR detected text: {text from Tesseract, or "none"}
Audio track present: {yes | unknown}
Measured edit-cut rate: {cutsPerSecond} cuts/sec

Examine every frame for production template signals: camera framing and angle, subject position in
frame, background environment (be specific — not just "plain" but e.g. "white wall, no props"),
lighting rig inferred from catchlights or shadows, text overlay layout (position, font weight, number
of visible layers), color grade consistency across frames.

If OCR text is present, use its position, casing, and typographic style to anchor `text_overlays`
classification — a recurring text overlay pattern (same position, same font weight, same caption
structure) is a strong format identity signal. Classify `text_overlays.style` and
`text_overlays.position` to match that pattern precisely.

For `pacing`: cross-check your visual assessment against the measured cut rate above — they must agree.
A measured rate above 1.5 cuts/sec is fast or very_fast; below 0.3 cuts/sec is slow or static.

For `content_category`: classify by delivery mechanism, not topic. A step-by-step instructional is
"tutorial"; general knowledge sharing without steps is "educational". If the delivery mechanism is
ambiguous between two similar categories (e.g. "educational" vs "motivational"), prefer the broader
or more common one — a consistent label across similar videos matters more than the precise semantic
distinction. Do not let topic differences push structurally identical videos into different categories.

For `narrative_structure`: classify the structural shape of the video, not its content. A video that
opens with a hook then resolves a question is "hook_reveal" regardless of subject matter. If the
structure fits both "hook_reveal" and "single_take", prefer "hook_reveal" — it is more specific.

Return ONLY valid JSON — no markdown, no explanation:
{
  "footage_type": "talking_head"|"b_roll"|"animation"|"screen_recording"|"mixed",
  "content_category": "tutorial"|"vlog"|"entertainment"|"product_showcase"|"lifestyle"|"motivational"|"educational"|"review"|"comedy"|"other",
  "narrative_structure": "hook_reveal"|"problem_solution"|"story"|"listicle"|"before_after"|"single_take"|"montage"|"other",
  "subject": {
    "type": "single_person"|"group"|"product"|"scenery"|"abstract"|"text_only",
    "face_to_camera": boolean,
    "presenter_style": "direct_monologue"|"interview"|"voiceover_only"|"silent"|"reaction"|"none"
  },
  "visual_style": {
    "color_tone": "warm"|"cool"|"neutral"|"high_contrast"|"desaturated"|"vibrant",
    "aesthetic": "cinematic"|"raw"|"polished"|"minimal"|"chaotic"|"corporate"
  },
  "scene": {
    "background": "plain"|"natural"|"urban"|"studio"|"digital"|"other",
    "filming_style": "handheld"|"tripod"|"drone"|"static"|"screen_capture",
    "lighting": "natural"|"studio"|"artificial"|"low_light"|"mixed",
    "setting": "indoor"|"outdoor"|"mixed"
  },
  "text_overlays": {
    "present": boolean,
    "has_subtitles": boolean,
    "overlay_count": integer,
    "primary_text": "exact wording of the largest or most prominent text overlay, or empty string",
    "hook_text": "text visible in Frame 1 or Frame 2, or empty string",
    "position": "top"|"center"|"bottom"|"full_screen"|"scattered"|"none",
    "style": "bold_caption"|"subtitle"|"handwritten"|"minimal"|"graphic_heavy",
    "animated": boolean
  },
  "audio_type": "music"|"voiceover"|"music_and_voiceover"|"silent"|"trending_audio",
  "voiceover_style": "educational"|"conversational"|"dramatic"|"hype"|"none",
  "hook_strength": "strong"|"medium"|"weak",
  "hook_type": "question"|"bold_statement"|"action"|"visual_surprise"|"text_only"|"product_reveal"|"story_open",
  "pacing": "very_fast"|"fast"|"medium"|"slow"|"static",
  "production_quality": "high"|"medium"|"low",
  "structural_fingerprint": ["3–5 observable production choices that identify this as a recurring template on sight: camera framing (e.g. 'tight face crop, center frame'), background detail (e.g. 'white wall, no props'), text overlay pattern (e.g. 'bold white caption, bottom third, 2 layers'), lighting signature (e.g. 'ring light catchlights'), audio signature (e.g. 'voiceover over music bed'). Observable production choices only — no content topics, no descriptions of what is said or shown."],
  "confidence": 0.0–1.0
}
```

---

### What Happens With the Response

The JSON is parsed into a `GrokFrameAnalysis` object (the `distinguishing_features` field is renamed to `structural_fingerprint` in the interface) and used in two ways:

1. **Database storage** — key fields (`footage_type`, `audio_type`, `hook_strength`, `production_quality`, `scene`, `text_overlays`) are written to the `video_analyses` table. The full raw JSON is stored in `grok_raw_json`.

2. **Feature vector encoding** — the analysis is encoded into a 111-dimensional `Float32Array` by `src/server/clustering/featureEncoder.ts`. Each categorical field becomes a one-hot group; booleans and normalized scalars occupy single dimensions. This vector is stored in the `feature_vectors` table and later fed into k-means clustering to form content format groups.

**111-dim vector layout:**

| Dims | Field | Classes | Notes |
|---|---|---|---|
| 0–4 | footage_type | 5 | High reliability — structural |
| 5–14 | content_category | 10 | Semantic drift risk — see fragmentation note below |
| 15–22 | narrative_structure | 8 | |
| 23–28 | subject.type | 6 | |
| 29 | subject.face_to_camera | 1 (bool) | High reliability |
| 30–34 | subject.presenter_style | 5 | `none` → all zeros |
| 35–40 | scene.background | 6 | High reliability — structural |
| 41–45 | scene.filming_style | 5 | High reliability — structural |
| 46–50 | scene.lighting | 5 | |
| 51–53 | scene.setting | 3 | |
| 54–59 | visual_style.color_tone | 6 | Low reliability — subjective |
| 60–65 | visual_style.aesthetic | 6 | Low reliability — subjective |
| 66–70 | audio_type | 5 | `trending_audio` is a distinct 5th class — do NOT collapse to `music` |
| 71–75 | voiceover_style | 5 | |
| 76–78 | hook_strength | 3 | |
| 79–85 | hook_type | 7 | |
| 86–90 | pacing | 5 | Must agree with measured cuts/sec |
| 91–93 | production_quality | 3 | |
| 94 | text_overlays.present | 1 (bool) | High reliability |
| 95 | text_overlays.has_subtitles | 1 (bool) | |
| 96 | text_overlays.animated | 1 (bool) | |
| 97–101 | text_overlays.style | 5 | |
| 102–107 | text_overlays.position | 6 | |
| 108 | text_overlays.overlay_count | 1 (scalar) | Bucketed: 0=none · 0.33=1–2 · 0.67=3–5 · 1.0=6+ |
| 109 | cuts_per_second (normalized) | 1 (scalar) | Objective measurement: `Math.min(cutsPerSecond / 5.0, 1.0)` |
| 110 | confidence | 1 (scalar) | |

**Content_category fragmentation risk**: with 10 dims, a single category mismatch between structurally identical videos (e.g. "tutorial" vs "educational") contributes a euclidean distance of √2 ≈ 1.41 — enough to split a tight cluster. The prompt instruction to prefer broader, consistent labels reduces this; the dimension weighting below further contains it.

If no API key is configured, a neutral stub vector is used so the pipeline still runs end-to-end without blocking.

---

## Stage 2 — Format Naming (Text)

### When
After k-means clustering assigns all analyzed reels into format groups, the label generator runs once per cluster set to give each group a human-readable name.

### Where
`src/server/clustering/labelGenerator.ts` → `generateFormatLabels()`

### How

**Model:** `grok-3-mini` (text)
**Temperature:** `0`
**Max tokens:** `256`

Each cluster's centroid is decoded back into readable characteristics via `decodeCentroid()` and passed as a structured description. Grok names each cluster in 3–6 words based on the combination of properties that caused those reels to group together.

`decodeCentroid` must reflect the updated audio_type list (5 classes including `trending_audio`) and must decode the `overlay_count` bucket back to a human-readable range for the cluster description.

**Example cluster description fed to Grok:**
```
Cluster 0: footage=talking head, category=tutorial, structure=problem solution,
subject=single person, presenter=direct monologue, face_to_camera=true,
background=plain, filming=static, lighting=artificial, setting=indoor,
color=neutral, aesthetic=polished, audio=voiceover, voiceover=educational,
hook=strong (question), pacing=medium, quality=high, cuts/sec=0.8,
text_overlays=present (bold_caption, bottom, ~2 overlays)
```

---

## Stage 3 — Performance Insights (Text)

### When
After clusters are formed and engagement rates are computed, the insights generator runs once per account to explain why the worst-performing format underperforms and what can be improved across all formats.

### Where
`src/server/analytics/insightsGenerator.ts` → `generateInsights()`

### How

**Model:** `grok-3-mini` (text)
**Temperature:** `0`
**Max tokens:** `512`

Grok receives the pre-computed best and worst format characteristics alongside their factual engagement figures (calculated as `(likes + comments + reposts + sends) / plays × 100`). It returns structured JSON with 2–3 weakness bullet points and 2–3 actionable recommendations. All numbers in the prompt are real — Grok does not estimate or fabricate engagement figures.

---

## Clustering Engine

### Where
`src/server/clustering/clusterEngine.ts`

### k selection
`k = Math.max(2, Math.min(8, Math.round(n / 5)))` — not `n / 3`.

Dividing by 3 over-fragments small accounts: 19 videos → k=6 with the old formula, meaning clusters average only 3 members, easily splitting a 9-video format group across 2–3 clusters. Dividing by 5 gives k=4 for the same account, keeping natural format groups intact. The cap of 8 (not 10) prevents excessive fragmentation on larger accounts.

### Dimension weighting
Before computing euclidean distance, scale vector dimensions by reliability:

- **2×**: dims 0–4 (footage_type), 29 (face_to_camera), 35–40 (background), 41–45 (filming_style), 94 (text_overlays.present)
- **0.5×**: dims 54–59 (color_tone), 60–65 (aesthetic)
- **1×**: all other dims

Weighting is applied by multiplying each dimension value before distance computation — not by post-processing assignments. This ensures that high-reliability structural signals (what the camera is pointed at, how it's mounted, what's in the background) dominate over subjective style fields (how warm the color grade feels) which Grok assesses inconsistently across near-identical videos.

---

## Key Constraints

| Rule | Reason |
|---|---|
| Grok vision only classifies structure — never estimates engagement | Engagement is pure math; AI-generated numbers would corrupt rankings |
| `temperature: 0` on all calls | Deterministic output ensures consistent feature vectors across re-scans |
| Frames resized to 512×512 before sending | Reduces token cost while preserving enough detail for structural classification |
| `structural_fingerprint` field required (replaces `distinguishing_features`) | Forces Grok to name observable production choices, not content topics — cluster labels reflect templates, not themes |
| `cutsPerSecond` passed to `analyzeFrames` and included in prompt | Grok must cross-check its pacing classification against the objective measurement; this eliminates pacing disagreements on structurally identical videos |
| `trending_audio` encoded as a distinct 5th class | Collapsing it to `music` loses a real clustering signal |
| `overlay_count` bucketed into the feature vector | Collected by the pipeline but previously unused; 4-bucket scalar adds precision without over-fitting |
| k = `round(n / 5)`, max 8 | Prevents micro-cluster fragmentation on small accounts — format groups need ≥3 members to be meaningful |
| High-reliability dims weighted 2× in distance computation | Structural facts dominate over subjective style fields that vary across near-identical videos |
| Content_category: prefer broader label when ambiguous | Semantic drift between overlapping classes (tutorial/educational) is the primary cause of identical videos landing in different clusters |
| Stub analysis used when no API key present | Pipeline runs end-to-end for testing without blocking on a missing key |
| Tokens from Instagram never sent to xAI | Credentials are stored in macOS Keychain via `keytar` and never passed to any AI model |
