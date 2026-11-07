# Instana — Full Implementation Plan

## Overview
Electron desktop app analyzing an Instagram account's reels/posts via Graph API. Videos are classified by xAI Grok vision, clustered into "formats", and scored by engagement rate. Daily auto-scan included. Build backend + frontend together per phase so every feature is testable immediately.

---

## Directory Structure

```
/
├── index.html
├── vite.config.ts
├── tsconfig.json
├── tailwind.css
├── drizzle.config.ts
├── .env / .env.example
│
├── src/
│   ├── electron/
│   │   ├── main.ts              # Creates BrowserWindow, calls startServer()
│   │   └── preload.ts           # contextBridge (empty passthrough for now)
│   │
│   ├── server/                  # Express backend
│   │   ├── index.ts             # Express app factory + startServer()
│   │   ├── router.ts
│   │   ├── logger.ts            # Winston
│   │   ├── db/
│   │   │   ├── schema.ts        # All Drizzle table definitions
│   │   │   └── index.ts         # DB singleton (WAL, FK on)
│   │   ├── config/
│   │   │   └── settings.ts      # Port, cron, frame counts
│   │   ├── auth/
│   │   │   ├── oauth.ts         # Instagram OAuth PKCE, token exchange/refresh
│   │   │   └── tokenStore.ts    # keytar wrappers
│   │   ├── instagram/
│   │   │   ├── client.ts        # Axios + auth + 429 retry
│   │   │   ├── mediaFetcher.ts  # Paginate /me/media
│   │   │   └── insightsFetcher.ts
│   │   ├── pipeline/
│   │   │   ├── videoDownloader.ts
│   │   │   ├── frameExtractor.ts   # FFmpeg → up to 10 JPEGs
│   │   │   ├── audioExtractor.ts
│   │   │   ├── ocrProcessor.ts     # Tesseract.js PSM 11
│   │   │   ├── editPatternDetector.ts  # Frame diff → cuts/sec
│   │   │   ├── grokAnalyzer.ts     # xAI vision → GrokFrameAnalysis
│   │   │   └── pipelineOrchestrator.ts
│   │   ├── clustering/
│   │   │   ├── featureEncoder.ts   # → Float32Array[32]
│   │   │   ├── clusterEngine.ts    # k-means++ over HNSW
│   │   │   └── formatRepository.ts
│   │   ├── analytics/
│   │   │   ├── engagementCalculator.ts  # pure math
│   │   │   └── insightsGenerator.ts     # Grok text for weakness explanation only
│   │   ├── scheduler/
│   │   │   └── dailyScanner.ts
│   │   └── routes/
│   │       ├── auth.ts
│   │       ├── media.ts
│   │       ├── pipeline.ts      # SSE for progress
│   │       ├── analytics.ts
│   │       ├── formats.ts
│   │       ├── sections.ts
│   │       └── scheduler.ts
│   │
│   └── renderer/                # React frontend
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts        # Typed fetch wrappers
│       ├── store/
│       │   ├── appStore.ts
│       │   ├── libraryStore.ts
│       │   └── pipelineStore.ts
│       ├── constants/styles.ts  # IG gradient, shared class strings
│       ├── components/
│       │   ├── layout/          # AppShell, Sidebar, BackButton
│       │   └── shared/          # InstagramButton, TextInput, VideoThumbnail,
│       │                        # FormatCard, SectionRect, EngagementBadge, Modal
│       └── pages/
│           ├── OnboardingPage.tsx
│           ├── HomePage.tsx
│           ├── LibraryPage.tsx
│           ├── AddSectionPage.tsx
│           ├── SectionPage.tsx
│           ├── ScanningPage.tsx
│           ├── FormatDescriptionPage.tsx
│           └── ProfilePage.tsx
│
└── data/hnsw/                   # Persisted HNSW index files
```

---

## Database Schema

| Table | Key Columns |
|---|---|
| `profiles` | `id`, `username`, `pfp_url`, `created_at` |
| `accounts` | `id` (IG user_id), `profile_id`, `username`, `section_id`, `followers_count`, `last_sync_at` |
| `sections` | `id`, `name`, `color`, `ig_profile_url`, `account_id`, `created_at` |
| `media` | `id`, `account_id`, `media_type`, `media_url`, `thumbnail_url`, `caption`, `timestamp` |
| `media_insights` | `media_id` PK, `views`, `likes`, `comments`, `shares`, `saves`, `engagement_rate` |
| `video_analyses` | `id`, `media_id`, `detected_text`, `cuts_per_second`, `footage_type`, `background_type`, `text_style_json`, `audio_type`, `hook_strength`, `production_quality`, `grok_raw_json`, `status` |
| `feature_vectors` | `media_id` PK, `vector` blob, `dimension` |
| `formats` | `id`, `account_id`, `label`, `centroid_json`, `video_count`, `avg_engagement`, `top_media_id`, `bottom_media_id`, `created_at` |
| `format_memberships` | `media_id`, `format_id`, `distance` |
| `insights` | `id`, `account_id`, `best_format_id`, `worst_format_id`, `report_json`, `generated_at` |
| `sync_jobs` | `id`, `account_id`, `triggered_by`, `status`, `media_fetched`, `analyzed`, `started_at`, `finished_at` |

---

## REST API Routes

| Method | Path | Description |
|---|---|---|
| GET | `/api/auth/status` | Token status, username, expiry |
| POST | `/api/auth/connect` | Start OAuth flow |
| DELETE | `/api/auth/disconnect` | Revoke token |
| POST | `/api/settings/grok-key` | Save xAI API key |
| GET | `/api/settings` | Non-secret config |
| POST | `/api/media/fetch` | Trigger media + insights fetch |
| GET | `/api/media` | Paginated media list |
| GET | `/api/media/:id` | Single media + insights + analysis |
| POST | `/api/pipeline/run` | Start analysis → `{ jobId }` |
| GET | `/api/pipeline/status/:jobId` | Job status + progress % |
| GET | `/api/pipeline/events/:jobId` | **SSE** real-time progress stream |
| POST | `/api/formats/rebuild` | Re-run clustering |
| GET | `/api/formats` | All formats with stats |
| GET | `/api/formats/:id` | Single format + member videos |
| GET | `/api/analytics/insights` | Latest InsightReport |
| GET | `/api/analytics/engagement` | Chart-ready data |
| GET | `/api/sections` | All sections |
| POST | `/api/sections` | Create section + trigger scan |
| PATCH | `/api/sections/:id` | Edit section |
| GET | `/api/scheduler/status` | `{ running, nextRun, lastRun }` |
| POST | `/api/scheduler/toggle` | Enable/disable + set cron |
| POST | `/api/scheduler/run-now` | Manual trigger |
| GET | `/api/profile` | Local profile |
| POST | `/api/profile` | Create/update profile |

---

## AI Rules
- **Grok vision** (`grok-2-vision-1212`): classifies footage type, background, text style, audio type, hook strength. Returns strict JSON `GrokFrameAnalysis`. Never generates scores.
- **Grok text** (`grok-3-mini`): called once after clustering to explain why worst format underperforms. Input = pre-calculated numbers. Output = `worst_format_weaknesses[]` + `overall_recommendations[]`.
- **All engagement rates, rankings, comparisons = pure math.** No AI involved.

### Grok Vision Prompt (strict JSON, no markdown)
```
System: "Return ONLY valid JSON matching the schema. No markdown, no explanation."
User: "Analyze these N frames from an Instagram Reel.
OCR detected: [text overlays]
Audio present: yes/unknown

Return ONLY:
{
  footage_type: 'talking_head'|'b_roll'|'animation'|'screen_recording'|'mixed',
  background_type: 'plain'|'natural'|'urban'|'studio'|'digital'|'other',
  text_style: { present, position, color, style, animated },
  audio_type: 'music'|'voiceover'|'music_and_voiceover'|'silent'|'trending_audio',
  hook_strength: 'strong'|'medium'|'weak',
  production_quality: 'high'|'medium'|'low',
  confidence: 0.0–1.0
}"
```

### Feature Vector (32-dim Float32Array)
| Dims | Feature | Encoding |
|---|---|---|
| 0–4 | footage_type | one-hot 5 |
| 5–10 | background_type | one-hot 6 |
| 11–14 | audio_type | one-hot 4 |
| 15–17 | hook_strength | one-hot 3 |
| 18–20 | production_quality | one-hot 3 |
| 21–23 | text_style.position | one-hot 3 |
| 24–27 | text_style.style | one-hot 4 |
| 28 | text_style.present | 0/1 |
| 29 | text_style.animated | 0/1 |
| 30 | cuts_per_second / 5.0 | normalized |
| 31 | grok confidence | 0–1 |

### Clustering Algorithm
1. Load all feature vectors for the account from DB
2. Build HNSW index (`cosine` space, dim=32)
3. k-means++ initialization, k = `min(10, floor(n/3))`, min 2
4. 20 iterations of k-means
5. Map cluster assignments → format labels (argmax of one-hot dims)
6. Persist index to `data/hnsw/{accountId}.bin`

---

## UI Design Rules
- Black background on every page
- All text white
- Instagram gradient (exact): `linear-gradient(45deg, #FCAF45, #FD1D1D, #E1306C, #833AB4, #405DE6)`
- Text inputs: grey background (`bg-neutral-700`) + thin white outline (`border border-white/30`)
- All video thumbnails: same fixed size, 9:16 ratio (180×320px)
- All headings same class, all titles same class (defined in `constants/styles.ts`)
- Back button on every page (top-left)
- Sidebar: hamburger (top-left) → Home, Library, Scan, Profile links

### Pages
| Page | Route | Key Elements |
|---|---|---|
| Onboarding | `/onboarding` | Username, pfp upload, IG link, scan button — shown only on first run |
| Home | `/home` | "Analyze Insta!" title, top 5 formats list, 5 recent FormatCards, IG gradient countdown timer |
| Library | `/library` | Search bar, section grid (3/row, rounded, custom color), + add button, right-click to edit |
| Add Section | `/library/add` | Name input, color picker, IG profile URL, "Scan & Finish" button |
| Section | `/library/section/:id` | Scan button, search bar, FormatCards grid |
| Scanning | `/scan` | Grey box + IG gradient scan button → Framer Motion drop animation → formats appear |
| Format Description | `/format/:id` | Left: thumbnail + engagement rate + video links list. Right: AI summary + pros/cons (cons = light red tint) |
| Profile | `/profile` | Account info, Grok API key input, scheduler toggle |

---

## Build Phases (backend + frontend together)

### Phase 1 — Project Setup
- Config files: `vite.config.ts`, `tsconfig.json`, `tailwind.css`, `drizzle.config.ts`, `index.html`, `.env.example`
- `package.json` scripts: `dev`, `build`, `db:generate`, `db:migrate`
- Backend: `db/schema.ts`, `db/index.ts`, `logger.ts`, `config/settings.ts`, `server/index.ts` (bare Express, `/api/health`)
- Frontend: `renderer/main.tsx`, `App.tsx` (black bg + router), `constants/styles.ts`, all layout + shared components
- Electron: `electron/main.ts` (BrowserWindow + startServer), `electron/preload.ts`
- **Verify**: `npm run dev` → Electron window opens, black bg, sidebar works, `/api/health` → 200

### Phase 2 — Auth + Onboarding
- Backend: `auth/tokenStore.ts`, `auth/oauth.ts`, `routes/auth.ts`, `routes/profile.ts`
- Frontend: `store/appStore.ts`, `OnboardingPage.tsx`, `ProfilePage.tsx`, onboarding route guard in `App.tsx`
- **Verify**: First run → Onboarding → OAuth → token in macOS Keychain, row in `accounts` table

### Phase 3 — Media Fetch + Library Structure
- Backend: `instagram/client.ts`, `instagram/mediaFetcher.ts`, `instagram/insightsFetcher.ts`, `routes/media.ts`, `routes/sections.ts`
- Frontend: `store/libraryStore.ts`, `LibraryPage.tsx`, `AddSectionPage.tsx`, `SectionPage.tsx` (skeleton)
- **Verify**: Create section → media rows in SQLite → section rect appears in Library

### Phase 4 — Video Pipeline + Scanning UI
- Backend: all 6 pipeline modules + `pipelineOrchestrator.ts` + `routes/pipeline.ts` (SSE)
- Frontend: `store/pipelineStore.ts`, `ScanningPage.tsx` (grey box + Framer Motion), progress in SectionPage
- **Verify**: Scan → SSE events stream → `video_analyses.status='done'` → 10 frames extracted for 30s+ video → temp files cleaned

### Phase 5 — Clustering + Format Display
- Backend: `featureEncoder.ts`, `clusterEngine.ts`, `formatRepository.ts`, `engagementCalculator.ts`, `routes/formats.ts`
- Frontend: `FormatDescriptionPage.tsx` (thumbnail, engagement rate, video links, pros/cons), SectionPage with real FormatCards
- **Verify**: Rebuild clusters → `formats` table populated → FormatDescriptionPage shows correct engagement rate

### Phase 6 — Insights + Home Page
- Backend: `insightsGenerator.ts`, `routes/analytics.ts`
- Frontend: `HomePage.tsx` (top 5 formats, recent FormatCards, countdown timer), wire AI summary + pros/cons into FormatDescriptionPage
- **Verify**: `GET /api/analytics/insights` → numbers match DB → Home page top 5 ranked correctly

### Phase 7 — Scheduler + Timer
- Backend: `dailyScanner.ts`, `routes/scheduler.ts`
- Frontend: CountdownTimer click → Modal with time picker, ProfilePage scheduler toggle
- **Verify**: Set scan for 1 min → `sync_jobs` row with `triggered_by: 'scheduler'`

---

## Engagement Formula
```
engagement_rate = (likes + comments + shares + saves) / views * 100
```
Computed in `engagementCalculator.ts`. Stored in `media_insights.engagement_rate`. All format rankings use this value directly — no AI involved.

## Frames to Grok
- Extract: up to 10 frames (FFmpeg, evenly distributed)
- Send to Grok: 8 frames for ≥30s, 5 for 15–30s, 3 for <15s
- Resize to 512×512 via sharp before base64 encoding
- Configurable in `src/server/config/settings.ts`

## Token / Auth Flow
1. OAuth → short-lived token
2. Exchange via `ig_exchange_token` → 60-day long-lived token
3. Store in macOS Keychain via `keytar`
4. Refresh via `ig_refresh_token` when within 7 days of expiry
5. Axios client interceptor handles 401 → auto-refresh + retry
