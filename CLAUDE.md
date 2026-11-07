# Instana — Claude Code Guide

## Implementation Plan
**Read `plan.md` before starting any work.** It contains the complete implementation plan:
- Full directory structure and every file to create
- Database schema (all tables + columns)
- All REST API routes
- Build phases in order (Phase 1–7) — always work within the current phase before moving to the next
- AI prompt design, feature vector layout, clustering algorithm
- UI page specs and design rules
- Verification checklist per phase

When implementing: check which phase is current, build backend + frontend together for that phase, verify before moving on.

## Project
Electron desktop app that analyzes an Instagram account's reels/posts via the Graph API, classifies videos with xAI Grok vision, clusters them into "formats", and generates number-driven engagement insights. Daily auto-scan included.

## Tech Stack
| Layer | Tool |
|---|---|
| Desktop | Electron 41.5 + `vite-plugin-electron` (dev mode only, no packaging yet) |
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS 4 + Framer Motion + React Router v7 + Zustand |
| Backend | Express 5 (started inside Electron main process) |
| DB | SQLite via `better-sqlite3` + `drizzle-orm` |
| Auth | `keytar` + Instagram Graph API OAuth |
| Video | `fluent-ffmpeg` + `sharp` + `tesseract.js` |
| AI | xAI Grok via `openai` SDK (`baseURL: 'https://api.x.ai/v1'`) |
| Clustering | `hnswlib-node` + k-means (32-dim feature vectors) |
| Scheduling | `node-cron` |

## Dev Workflow
```bash
npm run dev          # Electron + Vite dev server + Express (all in one)
npm run db:generate  # Drizzle-kit schema → SQL migrations
npm run db:migrate   # Apply migrations to data/instana.db
```
- Hot reload works for `src/renderer/` (React)
- Main process changes restart Electron automatically via `vite-plugin-electron`
- Express server starts on a random port; Electron passes it to the renderer

## Key File Locations
| What | Where |
|---|---|
| Electron entry | `src/electron/main.ts` |
| Express entry | `src/server/index.ts` |
| React entry | `src/renderer/main.tsx` |
| DB schema | `src/server/db/schema.ts` |
| Shared types | `src/shared/types.ts` |
| Config / env | `src/server/config/settings.ts` + `.env` |
| HNSW indexes | `data/hnsw/{accountId}.bin` |
| Logs | `logs/` |

## Coding Rules
- **TypeScript strict mode** throughout — no `any`, no unchecked casts
- **No AI-generated numbers** — engagement rates and rankings are pure math; Grok only classifies video structure and explains qualitative weaknesses
- **No comments** unless the WHY is non-obvious (hidden constraint, workaround, invariant)
- **No backwards-compat shims** — delete unused code completely
- **No error handling for impossible cases** — trust internal guarantees
- All DB access via Drizzle ORM — no raw SQL strings
- All Instagram tokens stored via `keytar` — never in `.env` or DB
- Temp video files go in `os.tmpdir()/instana/{mediaId}/` and are cleaned up after analysis

## AI Constraints
- Grok vision model: `grok-2-vision-1212`, `temperature: 0`, `max_tokens: 512`
- Frames sent to Grok: 8 (≥30s video), 5 (15–30s), 3 (<15s) — configurable in `settings.ts`
- Grok text model: `grok-3-mini` — only called to explain format weaknesses using pre-computed numbers
- Never send raw user data or account credentials to any AI model

## Instagram Graph API
- Base URL: `https://graph.instagram.com/v21.0`
- Reels use `plays` metric (not `video_views`) — insightsFetcher must detect media type
- Tokens: short-lived → `ig_exchange_token` → 60-day long-lived → `ig_refresh_token` before expiry
- Required permissions: `instagram_basic`, `instagram_manage_insights`

## Design System
```typescript
// src/renderer/constants/styles.ts
IG_GRADIENT = 'linear-gradient(45deg, #FCAF45, #FD1D1D, #E1306C, #833AB4, #405DE6)'
TEXT_INPUT_CLASS = 'bg-neutral-700 border border-white/30 rounded text-white px-3 py-2'
HEADING_CLASS = 'text-white font-semibold text-xl tracking-tight'
VIDEO_SIZE = { width: 180, height: 320 }  // all thumbnails same 9:16 ratio
```
- Black background everywhere
- All text white
- All Instagram gradient elements use the exact gradient above
- All text inputs: grey background + thin white outline

## Environment Variables
Copy `.env.example` → `.env`. Required keys:
- `GROK_API_KEY` — xAI API key (also stored in keytar after first setup)
- `IG_APP_ID` + `IG_APP_SECRET` — Facebook Developer App credentials
- `PORT` — Express port (default: random available port)

## Testing Checklist (run after each phase)
1. `npm run dev` → Electron window opens, black bg, sidebar works
2. `GET /api/health` → 200
3. After auth: token in macOS Keychain, row in `accounts` table
4. After scan: `video_analyses.status = 'done'`, temp files cleaned
5. Engagement rate hand-check: `(likes+comments+shares+saves)/views*100`
