# Instana — Current Codebase Snapshot

**Date:** 2026-05-20  
**Version:** 0.1.0

---

## Overview

Electron desktop app that analyses an Instagram account's reels via the Instagram Graph API, classifies videos with xAI Grok vision, clusters them into "formats", and generates engagement insights. Daily auto-scan included.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Desktop | Electron 41.5 + vite-plugin-electron |
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS 4 + Framer Motion + React Router v7 + Zustand |
| Backend | Express 5 (inside Electron main process) |
| DB | SQLite via better-sqlite3 + drizzle-orm |
| Auth | keytar + Instagram Graph API OAuth |
| Video | fluent-ffmpeg + sharp + tesseract.js |
| AI | xAI Grok via openai SDK (baseURL: https://api.x.ai/v1) |
| Clustering | hnswlib-node + k-means (111-dim feature vectors) |
| Scheduling | node-cron |

---

## Scripts

```bash
npm run dev               # Electron + Vite dev server + Express
npm run db:generate       # Drizzle-kit schema → SQL migrations
npm run db:migrate        # Apply migrations to data/instana.db
npm run electron:rebuild  # Rebuild native modules for Electron
```

---

## Directory Structure

```
src/
  electron/
    main.ts              # Electron entry — creates BrowserWindow, starts Express
    preload.ts           # Exposes apiPort to renderer via contextBridge
  server/
    index.ts             # Express setup, route mounting, port discovery
    logger.ts            # Winston logger (console + file at logs/instana.log)
    db/
      index.ts           # better-sqlite3 + drizzle init, runs migrations
      schema.ts          # All table definitions
    config/
      settings.ts        # App-wide constants (frame counts, model names, paths)
    auth/
      oauth.ts           # Instagram OAuth flow + token refresh + connectWithToken
      tokenStore.ts      # keytar wrapper for ig_access_token, grok_api_key, etc.
    instagram/
      client.ts          # Axios instance with token injection + 401/429 retry
      mediaFetcher.ts    # Paginates /media endpoint, upserts into media table
      insightsFetcher.ts # Fetches plays/likes/comments/reposts/sends per media (batches of 5)
    pipeline/
      pipelineOrchestrator.ts  # Concurrent scan loop — 3 videos in parallel, SSE events
      videoDownloader.ts       # Downloads video to os.tmpdir()/instana/{mediaId}/
      frameExtractor.ts        # ffmpeg frame extraction — 20 frames, all extracted in parallel
      audioExtractor.ts        # ffprobe audio stream detection
      ocrProcessor.ts          # tesseract.js OCR — singleton worker, reused across all videos
      editPatternDetector.ts   # Frame diff to count cuts/sec — all pairs compared in parallel
      grokAnalyzer.ts          # Sends evenly-strided frames to Grok vision — singleton client
    clustering/
      featureEncoder.ts        # Encodes GrokFrameAnalysis → 111-dim Float32Array + DIM_WEIGHTS
      clusterEngine.ts         # k-means++ (20 iter) + weighted distance + HNSW write
      coherencePass.ts         # Post-cluster outlier detection via grok-3-mini per cluster
      labelGenerator.ts        # Calls grok-3-mini to name each cluster from its characteristics
      formatRepository.ts      # rebuildFormats, getFormats, getFormatDetail
    analytics/
      engagementCalculator.ts  # (likes+comments+reposts+sends)/plays*100
      insightsGenerator.ts     # Grok text call to explain best/worst format weaknesses
      snapshotRecorder.ts      # Records avg engagement + format count per scan
    routes/
      auth.ts        # GET /status, POST /connect, POST /connect-token, DELETE /disconnect
      profile.ts     # GET/POST /api/profile
      media.ts       # POST /fetch, GET /, GET /:id
      pipeline.ts    # POST /run, GET /status/:id, GET /events/:id (SSE)
      formats.ts     # POST /rebuild, GET /, GET /:id
      analytics.ts   # GET /insights, POST /insights/refresh, GET /engagement, GET /snapshots
      scheduler.ts   # GET /status, POST /toggle, POST /run-now
      sections.ts    # CRUD for sections + background media/insights fetch
    scheduler/
      dailyScanner.ts  # node-cron daily scan → pipeline → rebuildFormats → generateInsights → snapshot
  renderer/
    main.tsx           # ReactDOM root, BrowserRouter
    App.tsx            # Route tree, onboarding gate
    api/client.ts      # apiFetch wrapper, reads apiPort from electronAPI
    store/
      appStore.ts      # Zustand — profile, account, isOnboarded, loadProfile
      pipelineStore.ts # Zustand — jobId, progress, status
    constants/
      styles.ts        # IG_GRADIENT, TEXT_INPUT_CLASS, HEADING_CLASS, VIDEO_SIZE
    pages/
      OnboardingPage.tsx        # Name → token → connect flow
      HomePage.tsx              # Countdown timer, top 5 formats, AI insights, schedule modal
      LibraryPage.tsx           # Scan button + progress + format grid
      ScanningPage.tsx          # Standalone scan page — triggers full pipeline including rebuildFormats
      FormatDescriptionPage.tsx # Format profile, member list, AI weaknesses
      ProfilePage.tsx           # Grok key, IG token update, scheduler, engagement graph
    components/
      layout/
        AppShell.tsx    # Sidebar toggle + Outlet
        Sidebar.tsx     # Slide-in nav (Home, Library, Scan, Profile)
        BackButton.tsx  # navigate(-1)
      shared/
        FormatCard.tsx        # Thumbnail + label + engagement badge
        EngagementBadge.tsx   # White monospace rate display
        VideoThumbnail.tsx    # 180×320 9:16 thumbnail box
        InstagramButton.tsx   # IG gradient button
        Modal.tsx             # Backdrop + centred panel
        TextInput.tsx         # Styled input using TEXT_INPUT_CLASS
```

---

## Database Schema

### `profiles`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| username | text NOT NULL | |
| pfp_url | text | |
| created_at | timestamp | default now |

### `accounts`
| Column | Type | Notes |
|---|---|---|
| id | text PK | Instagram user ID |
| profile_id | integer FK → profiles | |
| username | text NOT NULL | |
| section_id | integer FK → sections | nullable |
| followers_count | integer | |
| last_sync_at | timestamp | |

### `sections`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| name | text NOT NULL | |
| color | text NOT NULL | |
| ig_profile_url | text NOT NULL | |
| account_id | text | |
| created_at | timestamp | |

### `media`
| Column | Type | Notes |
|---|---|---|
| id | text PK | Instagram media ID |
| account_id | text FK → accounts | |
| media_type | text NOT NULL | REEL or VIDEO |
| media_url | text | signed URL, expires in hours |
| thumbnail_url | text | |
| caption | text | |
| timestamp | timestamp | |

### `media_insights`
| Column | Type | Notes |
|---|---|---|
| media_id | text PK FK → media | |
| views | integer | unused (null) |
| plays | integer | reel play count |
| likes | integer | total_interactions from API |
| comments | integer | always 0 (not separately available) |
| reposts | integer | reshares (API: `saved`) |
| sends | integer | DM sends (API: `shares`) |
| engagement_rate | real | (likes+comments+reposts+sends)/plays*100 |

### `video_analyses`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| media_id | text FK → media | |
| detected_text | text | OCR output |
| cuts_per_second | real | computed by editPatternDetector before Grok runs |
| footage_type | text | talking_head / b_roll / animation / screen_recording / mixed |
| background_type | text | |
| scene_json | text | full scene object from Grok |
| text_style_json | text | full text_overlays object from Grok |
| audio_type | text | music / voiceover / music_and_voiceover / silent / trending_audio |
| hook_strength | text | strong / medium / weak |
| production_quality | text | high / medium / low |
| grok_raw_json | text | full GrokFrameAnalysis JSON including structural_fingerprint |
| status | text | pending / processing / done / error |

### `feature_vectors`
| Column | Type | Notes |
|---|---|---|
| media_id | text PK FK → media | |
| vector | blob | 111 × float32 = 444 bytes |
| dimension | integer | always 111 (schema default: 111) |

### `formats`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| account_id | text FK → accounts | |
| label | text | AI-generated by grok-3-mini from cluster centroid characteristics |
| centroid_json | text | JSON array of 111 floats |
| video_count | integer | |
| avg_engagement | real | |
| top_media_id | text FK → media | highest engagement member |
| bottom_media_id | text FK → media | lowest engagement member |
| created_at | timestamp | |

### `format_memberships`
| Column | Type | Notes |
|---|---|---|
| media_id | text FK → media | |
| format_id | integer FK → formats | |
| distance | real | unweighted euclidean distance to centroid (all 111 dims) |

### `insights`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| account_id | text FK → accounts | |
| best_format_id | integer FK → formats | nulled on format rebuild |
| worst_format_id | integer FK → formats | nulled on format rebuild |
| report_json | text | full InsightReport serialised |
| generated_at | timestamp | |

### `sync_jobs`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| account_id | text FK → accounts | |
| triggered_by | text | 'manual' or 'scheduler' |
| status | text | pending / running / done / error |
| media_fetched | integer | |
| analyzed | integer | |
| started_at | timestamp | |
| finished_at | timestamp | |

### `scan_snapshots`
| Column | Type | Notes |
|---|---|---|
| id | integer PK autoincrement | |
| account_id | text FK → accounts | |
| scanned_at | timestamp | |
| avg_engagement_rate | real | |
| formats_count | integer | |

---

## API Routes

### Auth — `/api/auth`
| Method | Path | Description |
|---|---|---|
| GET | /status | Returns `{ connected, userId, expiresAt }` |
| POST | /connect | OAuth browser flow (opens Instagram login) |
| POST | /connect-token | Connect via pasted long-lived token |
| DELETE | /disconnect | Clears keychain token |

### Profile — `/api/profile`
| Method | Path | Description |
|---|---|---|
| GET | / | Returns `{ profile, account }` or null |
| POST | / | Create/update profile (username, pfpUrl) |

### Media — `/api/media`
| Method | Path | Description |
|---|---|---|
| POST | /fetch | Fetches all media + insights from Instagram API |
| GET | / | Lists media (limit/offset params) |
| GET | /:id | Single media with insights + analysis |

### Pipeline — `/api/pipeline`
| Method | Path | Description |
|---|---|---|
| POST | /run | Starts a force scan. Inserts `sync_jobs` row, responds immediately with `{ jobId }`, runs pipeline → rebuildFormats → generateInsights → recordScanSnapshot in background |
| GET | /status/:jobId | Returns `{ status, progress, accountId }` |
| GET | /events/:jobId | SSE stream of PipelineEvent objects |

### Formats — `/api/formats`
| Method | Path | Description |
|---|---|---|
| POST | /rebuild | Runs clusterVectors + coherencePass + writes formats to DB |
| GET | / | Lists all FormatSummary for the account |
| GET | /:id | Returns FormatDetail with members + characteristics |

### Analytics — `/api/analytics`
| Method | Path | Description |
|---|---|---|
| GET | /insights | Returns cached InsightReport (generates on first call) |
| POST | /insights/refresh | Re-calls Grok text model, saves new report |
| GET | /engagement | Format list sorted by avgEngagement desc |
| GET | /snapshots | Historical scan snapshots for engagement chart |

### Scheduler — `/api/scheduler`
| Method | Path | Description |
|---|---|---|
| GET | /status | Returns `{ running, nextRun, lastRun, config }` |
| POST | /toggle | Enable/disable + set hour/minute |
| POST | /run-now | Triggers immediate scheduled scan |

### Settings — `/api/settings`
| Method | Path | Description |
|---|---|---|
| GET | / | Returns `{ hasGrokKey, frames }` |
| POST | /grok-key | Saves xAI key to keychain |

---

## Scan Pipeline Flow

### 1. Start

**POST /api/pipeline/run** inserts a `sync_jobs` row, responds immediately with `{ jobId }`, then runs the full pipeline chain in the background:

```
runPipeline(jobId, accountId, force=true)
  → rebuildFormats(accountId)
  → generateInsights(accountId)
  → recordScanSnapshot(accountId)
```

Progress streams to the client via `GET /api/pipeline/events/:jobId` (SSE). SSE emitters are tracked in `pipelineOrchestrator.ts` and removed immediately when the pipeline finishes. Job state (`activeJobs` map, progress %, status) is tracked in `routes/pipeline.ts` and cleaned up 5 minutes after completion.

---

### 2. runPipeline(jobId, accountId, force=true)

**Setup (sequential):**

1. `fetchAllMedia` — paginates `/me/media`, upserts into `media` table.
2. `fetchAllInsights` — fetches `plays`, `total_interactions`, `saved`, `shares` per media item, **batched 5 requests at a time** to stay within Graph API rate limits; computes and upserts `engagement_rate` into `media_insights`.
3. Query all video media (`mediaType = REEL | VIDEO`) for the account.
4. Query all `video_analyses` with `status=done` for this account, extract `structural_fingerprint` arrays from `grok_raw_json` — these become **known template anchors** passed to every Grok vision call in this scan. Loaded before any force-delete so a re-scan still benefits from prior classifications.
5. If `force=true`: delete all existing `video_analyses` rows for the account.
6. `resetGrokClient()` — clears the singleton Grok client so any key change since last scan takes effect.
7. `initOcrWorker()` — starts a single shared Tesseract worker (loads English language data once, reused for all videos).

**Per-video processing (up to 3 videos concurrently):**

Videos already in `status=done` or `status=error` are skipped unless `force=true`. Up to 3 videos are in-flight simultaneously — while one waits on the Grok API, the next video's download and frame work proceeds.

| Step | SSE step name | Detail |
|---|---|---|
| Download | `downloading` | Video saved to `os.tmpdir()/instana/{mediaId}/video.mp4` |
| Frames + audio | `extracting_frames` | **In parallel:** (a) Extract 20 frames via ffmpeg — timestamps at `(i + 0.5) * interval` to avoid black frames at t=0, all 20 extracted simultaneously using per-frame `seekInput`; (b) Detect audio presence via ffprobe |
| OCR + cuts | `ocr` | **In parallel:** (a) Tesseract OCR on the first 3 frames using the shared singleton worker; (b) Frame-diff all adjacent pairs simultaneously to compute `cutsPerSecond` |
| Grok vision | `grok_analysis` | Select evenly-strided frames across the full duration (16/10/6 by duration); resize each to 512×512 JPEG; call Grok vision with known template anchors injected into the prompt |
| Save + clean | `done` | Upsert `video_analyses` (status=done), encode 111-dim feature vector, upsert `feature_vectors`, delete temp files |

On error: `video_analyses.status` set to `error`, temp files cleaned, SSE `error` event emitted. Pipeline continues with remaining videos.

The Tesseract worker is terminated after all videos complete (or if the pipeline throws).

---

### 3. rebuildFormats(accountId)

1. Load all `feature_vectors` for the account (joined through `media` to filter by `account_id`).
2. Run **k-means++**: `k = max(2, min(8, round(n / 5)))`, 20 iterations. Distance is weighted — see [Dimension Weighting](#dimension-weighting) below.
3. Best-effort HNSW index write to `{INSTANA_USER_DATA}/hnsw/{accountId}.bin` (10s timeout — may be skipped in Electron).
4. **Coherence pass** (`coherencePass.ts`): for each cluster with ≥3 members, fetch their `structural_fingerprint` arrays from `video_analyses`, call `grok-3-mini` to flag outliers (videos whose fingerprints don't match the majority production template), then reassign flagged videos to the nearest other centroid by weighted euclidean distance.
5. Decode all cluster centroids into human-readable characteristics via `decodeCentroid()` and call `grok-3-mini` **once** (all clusters in a single prompt) to generate descriptive format names. Falls back to `footageType` strings if no API key or unexpected response.
6. Delete existing formats + memberships for the account; insert new ones with `avgEngagement`, `topMediaId`, `bottomMediaId` computed from `media_insights`.

---

### 4. generateInsights(accountId)

- Reads best and worst formats (by `avgEngagement`) from the newly-written `formats` table.
- Decodes their centroids into characteristics via `decodeCentroid()`.
- Calls `grok-3-mini` with pre-computed engagement figures: 2–3 weakness bullet points and 2–3 actionable recommendations. All numbers are real — Grok never estimates engagement figures.
- Inserts a new row into `insights`. Appends `allFormats` (all clusters sorted by engagement) to the `InsightReport` for the engagement ranking UI.
- Falls back gracefully with placeholder strings if no API key is configured.

**InsightReport shape:**
```typescript
interface InsightReport {
  bestFormat: { id, label, avgEngagement, videoCount }
  worstFormat: { id, label, avgEngagement, videoCount }
  worstFormatWeaknesses: string[]
  overallRecommendations: string[]
  generatedAt: string   // ISO timestamp
  allFormats: { id, label, avgEngagement, videoCount }[]
}
```

---

### 5. recordScanSnapshot

Saves current `avgEngagementRate` (mean across all media) and `formatsCount` to `scan_snapshots` for the engagement-over-time chart.

---

## Daily Scanner

`dailyScanner.ts` runs via node-cron at the configured hour/minute.

Flow: `runPipeline(force=false)` (incremental — skips already-analysed videos) → `rebuildFormats` → `generateInsights` → `recordScanSnapshot`

Config persisted at `data/scheduler.json`: `{ enabled, hour, minute }`

---

## Engagement Formula

```
engagementRate = (likes + comments + reposts + sends) / plays * 100
```

- **likes** = `total_interactions` from Instagram API
- **comments** = always 0 (not separately available via the insights endpoint)
- **reposts** = reshares (Instagram API metric: `saved`)
- **sends** = DM sends (Instagram API metric: `shares`)
- **plays** = reel play count (Instagram API metric: `plays`)

All numbers are sourced directly from the Instagram Graph API. No AI-generated estimates.

---

## xAI Workflow — Three Stages

### Stage 1 — Video Frame Classification (Vision)

**Model:** `grok-4.3`  
**Temperature:** `0` · **Max tokens:** `1500`  
**When:** Once per reel, during the `grok_analysis` step of each video  
**File:** `src/server/pipeline/grokAnalyzer.ts` → `analyzeFrames()`

**Frame selection:** `strideSelect(framePaths, frameCount)` — picks evenly-spaced indices across all 20 extracted frames. 16 frames for ≥30s videos, 10 for 15–30s, 6 for <15s. Each frame is resized to 512×512 JPEG before base64-encoding. This ensures the full video duration is represented regardless of length.

**`cutsPerSecond`** is computed by `detectCutsPerSecond` before `analyzeFrames` is called and is passed into the function explicitly. It is included in the prompt so Grok must cross-check its pacing classification against the objective measurement.

**Known template anchors:** Before the scan loop begins, all `structural_fingerprint` arrays from prior `done` analyses for the account are loaded from `grok_raw_json` in the DB. These are injected into every Grok vision call as a "Known production templates" block, instructing Grok to reuse exact phrases when the video matches an existing template. This gives Grok consistent vocabulary across scans, reducing the classification drift that causes structurally identical videos to land in separate clusters.

The `OpenAI` client is created once per scan and reused. The API key is read from the macOS Keychain (or `GROK_API_KEY` env var) on the first call and cached. Both are reset at the start of each new scan via `resetGrokClient()`.

If no API key is configured, `STUB_ANALYSIS` is returned (all mid-range values, `confidence: 0.0`) so the pipeline completes end-to-end.

**System message:**
```
You are a video format classifier. Return ONLY valid JSON matching the schema. No markdown, no explanation.
```

**User message structure:**
```
You are classifying an Instagram Reel to identify the recurring production template it belongs to.
Your primary goal is to recognise format consistency — two videos shot with the same setup, same
background, and same text overlay style are the SAME FORMAT even if their topics differ completely.

Structural and visual production choices take priority over semantic content...

[Known production templates already identified in this account:]
[- Template 1: ["tight face crop, center frame", "white wall, no props", ...]]
[- Template 2: [...]]
[If this video matches one of the templates above, your structural_fingerprint MUST reproduce those exact phrases.]

Frames provided: {N} equally-spaced stills across a {duration}s video ({frame labels}).
OCR detected text: {Tesseract output, or "none"}
Audio track present: {yes | unknown}
Measured edit-cut rate: {cutsPerSecond} cuts/sec

[Examination instructions covering: camera framing, subject position, background environment,
 lighting, text overlay layout, color grade, pacing cross-check against measured cut rate,
 content_category preference for broader label when ambiguous,
 narrative_structure specificity rules]

Return ONLY valid JSON — no markdown, no explanation:
{ footage_type, content_category, narrative_structure, subject, visual_style, scene,
  text_overlays, audio_type, voiceover_style, hook_strength, hook_type, pacing,
  production_quality, structural_fingerprint, confidence }
```

**`structural_fingerprint`** — 3–5 observable production choices that identify this as a recurring template on sight: camera framing, background detail, text overlay pattern, lighting signature, audio signature. Observable production choices only — no content topics, no descriptions of what is said or shown. When known templates are provided, phrases are reused verbatim for matching templates.

**Output (`GrokFrameAnalysis`) is used for:**
1. Writing key fields to `video_analyses` (`footage_type`, `audio_type`, `hook_strength`, `production_quality`, `scene_json`, `text_style_json`, `grok_raw_json`)
2. Encoding a 111-dimensional feature vector stored in `feature_vectors`
3. Supplying `structural_fingerprint` for future scans' known-template anchors and for the coherence pass

---

### Stage 2 — Coherence Pass (Text, post-cluster)

**Model:** `grok-3-mini`  
**Temperature:** `0` · **Max tokens:** `128`  
**When:** Once per cluster with ≥3 members, immediately after k-means, before label generation  
**File:** `src/server/clustering/coherencePass.ts` → `runCoherencePass()`

For each qualifying cluster, the `structural_fingerprint` arrays of all members are fetched from `video_analyses` and presented to `grok-3-mini` in a single prompt:

```
These {N} videos were automatically clustered as a single content format. Review their structural
fingerprints and flag any that clearly belong to a different production template.

Video 1: ["tight face crop, center frame", "white wall, no props", ...]
Video 2: [...]
...

Return ONLY valid JSON: {"outliers": []} — list the 1-based video numbers that do NOT share
the same production template as the majority. Return [] if all belong together.
```

Flagged outliers are reassigned to the nearest other centroid by weighted euclidean distance. Clusters with <3 members are skipped. The coherence pass runs once per cluster, in sequence. The result is a corrected `assignments` array fed to the downstream label and membership steps.

---

### Stage 3 — Format Naming (Text)

**Model:** `grok-3-mini`  
**Temperature:** `0` · **Max tokens:** `256`  
**When:** Once per format rebuild, after the coherence pass  
**File:** `src/server/clustering/labelGenerator.ts` → `generateFormatLabels()`

Each cluster centroid is decoded into human-readable characteristics by `decodeCentroid()`. All cluster descriptions are sent to Grok **in a single prompt** — one API call for the entire account's format set.

**Example cluster description in the prompt:**
```
Cluster 0: footage=talking head, category=tutorial, structure=problem solution,
subject=single person, presenter=direct monologue, face_to_camera=true,
background=plain, filming=static, lighting=artificial, setting=indoor,
color=neutral, aesthetic=polished, audio=voiceover, voiceover=educational,
hook=strong (question), pacing=medium, quality=high, cuts/sec=0.8
```

**Response format:** `{ "labels": ["name for cluster 0", "name for cluster 1", ...] }`

Falls back to `footageType` strings if no API key, if the response shape doesn't match, or if the call throws.

---

### Stage 4 — Performance Insights (Text)

**Model:** `grok-3-mini`  
**Temperature:** `0` · **Max tokens:** `512`  
**When:** Automatically after every scan. Also on demand via `POST /api/analytics/insights/refresh`.  
**File:** `src/server/analytics/insightsGenerator.ts` → `generateInsights()`

Grok receives the pre-computed best and worst format characteristics alongside their factual engagement figures. It returns 2–3 weakness bullet points and 2–3 actionable recommendations. All numbers in the prompt are real — Grok does not estimate or fabricate engagement figures.

**Prompt input:**
```
Best format: "{label}" — {X.XX}% avg engagement ({N} videos)
  footage=..., category=..., [full characteristics]

Worst format: "{label}" — {X.XX}% avg engagement ({N} videos)
  footage=..., category=..., [full characteristics]

The best outperforms the worst by {N}x.
```

**Response format:**
```json
{
  "worst_format_weaknesses": ["2–3 short points"],
  "overall_recommendations": ["2–3 actionable tips"]
}
```

Falls back gracefully (placeholder strings) if no API key is configured.

---

### xAI Constraints

| Rule | Reason |
|---|---|
| Grok vision only classifies structure — never estimates engagement | Engagement is pure math; AI-generated numbers would corrupt rankings |
| `temperature: 0` on all calls | Deterministic output ensures consistent feature vectors and coherence decisions across re-scans |
| Frames resized to 512×512 JPEG before sending | Reduces token cost while preserving enough detail for structural classification |
| Frames selected by stride across all extracted frames, not by slicing from the front | Ensures Grok sees the full video duration; a short video's 6 frames span 0–100% not 0–30% |
| `cutsPerSecond` passed to `analyzeFrames` and included in the prompt | Grok must cross-check its pacing classification against the objective measurement; eliminates pacing disagreements on structurally identical videos |
| `structural_fingerprint` field required in vision response | Forces Grok to name observable production choices, not content topics; phrases are reused verbatim as template anchors in subsequent scans |
| Known template fingerprints injected into each vision call | Gives Grok cross-video awareness so structurally identical videos use consistent phrasing across concurrent and future scans |
| `trending_audio` encoded as a distinct 5th class | Collapsing it to `music` loses a real clustering signal |
| `overlay_count` bucketed into the feature vector | Adds text-overlay density as a precision signal without over-fitting |
| Dimension weighting applied before euclidean distance | High-reliability structural facts dominate over subjective style fields that vary across near-identical videos |
| Coherence pass runs after k-means before label generation | Catches videos misclassified into the wrong cluster due to semantic label drift; uses cheap text model, not vision |
| Stub analysis used when no API key present | Pipeline runs end-to-end for testing without blocking on a missing key |
| Instagram tokens never sent to xAI | Credentials stored in macOS Keychain via keytar; only the Grok API key goes to xAI |

---

## Feature Vector Layout (111 dims)

Vectors are `Float32Array`, stored as 444-byte blobs. Encoding: `featureEncoder.ts` → `encodeAnalysis()`. Decoding: `decodeCentroid()`.

| Dims | Field | Classes / Notes | Reliability |
|---|---|---|---|
| 0–4 | footage_type | talking_head, b_roll, animation, screen_recording, mixed | HIGH — 2× weight |
| 5–14 | content_category | tutorial, vlog, entertainment, product_showcase, lifestyle, motivational, educational, review, comedy, other | semantic drift risk |
| 15–22 | narrative_structure | hook_reveal, problem_solution, story, listicle, before_after, single_take, montage, other | |
| 23–28 | subject.type | single_person, group, product, scenery, abstract, text_only | |
| 29 | subject.face_to_camera | 0 or 1 | HIGH — 2× weight |
| 30–34 | subject.presenter_style | direct_monologue, interview, voiceover_only, silent, reaction (`none` → all zeros) | |
| 35–40 | scene.background | plain, natural, urban, studio, digital, other | HIGH — 2× weight |
| 41–45 | scene.filming_style | handheld, tripod, drone, static, screen_capture | HIGH — 2× weight |
| 46–50 | scene.lighting | natural, studio, artificial, low_light, mixed | |
| 51–53 | scene.setting | indoor, outdoor, mixed | |
| 54–59 | visual_style.color_tone | warm, cool, neutral, high_contrast, desaturated, vibrant | LOW — 0.5× weight |
| 60–65 | visual_style.aesthetic | cinematic, raw, polished, minimal, chaotic, corporate | LOW — 0.5× weight |
| 66–70 | audio_type | music, voiceover, music_and_voiceover, silent, **trending_audio** (distinct 5th class) | |
| 71–75 | voiceover_style | educational, conversational, dramatic, hype, none | |
| 76–78 | hook_strength | strong, medium, weak | |
| 79–85 | hook_type | question, bold_statement, action, visual_surprise, text_only, product_reveal, story_open | |
| 86–90 | pacing | very_fast, fast, medium, slow, static | must agree with measured cuts/sec |
| 91–93 | production_quality | high, medium, low | |
| 94 | text_overlays.present | 0 or 1 | HIGH — 2× weight |
| 95 | text_overlays.has_subtitles | 0 or 1 | |
| 96 | text_overlays.animated | 0 or 1 | |
| 97–101 | text_overlays.style | bold_caption, subtitle, handwritten, minimal, graphic_heavy | |
| 102–107 | text_overlays.position | top, center, bottom, full_screen, scattered, none | |
| 108 | text_overlays.overlay_count | bucketed scalar — 0=none, 0.33=1–2, 0.67=3–5, 1.0=6+ | |
| 109 | cuts_per_second | normalized: `Math.min(cutsPerSecond / 5.0, 1.0)` | objective measurement |
| 110 | confidence | 0.0–1.0 from Grok | |

### Dimension Weighting

Applied inside `euclideanSq` in both `clusterEngine.ts` and `coherencePass.ts` by multiplying each dimension's difference by its weight before squaring. Centroids are computed from raw unweighted means — weighting only affects distance comparisons.

```
2.0×  dims 0–4    footage_type
2.0×  dim  29     face_to_camera
2.0×  dims 35–40  scene.background
2.0×  dims 41–45  scene.filming_style
2.0×  dim  94     text_overlays.present
0.5×  dims 54–59  visual_style.color_tone
0.5×  dims 60–65  visual_style.aesthetic
1.0×  all other dims
```

---

## Clustering

**k formula:** `k = Math.max(2, Math.min(8, Math.round(n / 5)))`

Dividing by 5 keeps clusters averaging ≥5 members on small accounts. The cap of 8 prevents fragmentation on larger accounts. With the old `floor(n/3)`, 19 videos produced k=6 (averaging 3 members per cluster), easily splitting a 9-video format group across 2–3 clusters.

**Algorithm:** k-means++ initialisation (probability-weighted seeding), 20 Lloyd iterations with weighted euclidean distance, followed by the coherence pass before format DB writes.

**HNSW index:** Written best-effort to `{INSTANA_USER_DATA}/hnsw/{accountId}.bin` with a 10s timeout. Clustering and format writes complete regardless.

---

## Grok Models

| Model | Used for | Max tokens | Calls per scan |
|---|---|---|---|
| `grok-4.3` | Vision classification of each video's frames | 1500 | 1 per video |
| `grok-3-mini` | Coherence pass — outlier detection per cluster | 128 | 1 per cluster ≥3 members |
| `grok-3-mini` | Format label generation (all clusters in one prompt) | 256 | 1 |
| `grok-3-mini` | Insights — weaknesses and recommendations | 512 | 1 |

All calls use `temperature: 0`. Vision falls back to `STUB_ANALYSIS` (neutral values, `confidence: 0.0`) if no API key. Text calls fall back to placeholder strings.

---

## Key Config

`src/server/config/settings.ts`:
- `frames.totalExtracted`: `20` (extracted per video via ffmpeg, all in parallel)
- `frames.sentToGrok.long`: `16` (≥30s videos — evenly strided from all 20)
- `frames.sentToGrok.medium`: `10` (15–30s)
- `frames.sentToGrok.short`: `6` (<15s)
- `grok.visionModel`: `grok-4.3`
- `grok.textModel`: `grok-3-mini`
- `grok.temperature`: `0`
- `grok.maxTokens`: `1500` (vision; text models use their own lower limits)
- `db.path`: resolved at runtime as `process.env['INSTANA_USER_DATA'] + '/instana.db'` (not in `settings.ts`)

---

## Known Issues

1. **Instagram Insights API** — all media may return "not available". Account requires `instagram_manage_insights` permission (Business/Creator account + re-OAuth). Engagement rates are 0 until resolved.
2. **HNSW writeIndex** — hangs indefinitely inside Electron (native NODE_MODULE_VERSION mismatch). Mitigated with a 10s timeout; clustering and format DB writes complete successfully, only the `.bin` file is skipped.
3. **Existing feature vectors invalidated after dim bump** — any vectors stored before the 109 → 111 dimension change must be regenerated. Run `npm run db:generate && npm run db:migrate` then trigger a force re-scan.
