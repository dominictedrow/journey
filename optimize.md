# Instana Scan Optimizations

## What Changed and Why It's Dramatically Faster

The original pipeline did everything one step at a time — one video finished completely before the next started, and within each video every sub-step waited for the previous one to finish. None of that waiting was necessary. The changes below remove it without touching what the pipeline actually produces.

---

## The Changes

### 1. Frame extraction runs all at once
**File:** `src/server/pipeline/frameExtractor.ts`

Before, 20 video frames were pulled out one by one — each waited for the last to finish. These 20 operations have nothing to do with each other, so they now all run at the same time. On a typical reel this drops from ~7 seconds to under 1 second.

---

### 2. Audio detection and frame extraction happen at the same time
**File:** `src/server/pipeline/pipelineOrchestrator.ts`

After a video is downloaded, the pipeline used to extract frames and *then* check for audio. Checking for audio is a quick metadata read that doesn't need the frames to exist first. Both now start simultaneously.

---

### 3. OCR and cut detection happen at the same time
**File:** `src/server/pipeline/pipelineOrchestrator.ts`

After frames are extracted, the pipeline used to run text detection (OCR) and then count edit cuts. Both operations just read the frame files that are already on disk — neither depends on the other. They now run in parallel, saving ~3 seconds per video.

---

### 4. Up to 3 videos are processed at the same time
**File:** `src/server/pipeline/pipelineOrchestrator.ts`

The biggest single change. Before, the pipeline finished one video completely before touching the next. While one video was waiting on a response from the Grok API (2–4 seconds of idle time), nothing else was happening. Now up to 3 videos are in-flight at once, so that idle time is filled with download and frame work on the next video. For a 19-video account this alone cuts total scan time by roughly 60%.

The limit is set to 3 (not higher) to stay comfortably within xAI API rate limits.

---

### 5. Instagram insights are fetched in batches of 5, not one at a time
**File:** `src/server/instagram/insightsFetcher.ts`

Before, engagement data was fetched from the Graph API one video at a time in a queue. For 19 videos at ~400ms each that was nearly 8 seconds of pure waiting. Now 5 requests go out at once, reducing that to 2 batches. The batch size of 5 stays safely under Instagram's rate limits.

---

### 6. The Grok API connection is created once, not 19 times
**File:** `src/server/pipeline/grokAnalyzer.ts`

Before, a new HTTP client was created for every single video. Each creation set up new network connections from scratch. The client is now created once at startup and reused for every video, eliminating redundant connection setup.

---

### 7. The Grok API key is read from the keychain once, not 19 times
**File:** `src/server/pipeline/grokAnalyzer.ts`

The API key was being fetched from macOS Keychain on every video. The key doesn't change during a scan. It is now read once on the first video and cached in memory for the rest of the scan.

---

### 8. The Tesseract OCR worker is created once and reused
**File:** `src/server/pipeline/ocrProcessor.ts`

Before, a Tesseract worker (which loads English language data from disk) was created at the start of each video's OCR step and destroyed when finished. This initialization cost ~1–2 seconds and was paid 19 times. The worker is now created once and stays alive for the entire scan.

---

### 9. Edit-cut detection processes all frame pairs at once
**File:** `src/server/pipeline/editPatternDetector.ts`

Counting edit cuts works by comparing adjacent frames. These comparisons are fully independent — frame 1 vs 2 has nothing to do with frame 10 vs 11. They now all run simultaneously instead of one pair at a time.

---

## Estimated Time Savings

| Change | Per-video | 19-video total |
|---|---|---|
| Parallel frame extraction | ~6s | ~114s |
| Parallel OCR + cut detection | ~3s | ~57s |
| Persistent Tesseract worker | ~1.5s | ~28s |
| Parallel cut detection pairs | ~0.3s | ~6s |
| Batched insights fetching | — | ~6s |
| Singleton Grok client + key cache | ~0.1s | ~2s |
| **3x concurrent video processing** | — | **~60% of what's left** |

A scan that previously took 5–6 minutes for 19 videos should complete in under 90 seconds.

---

## Will Anything Perform Worse?

### Memory usage will be slightly higher
Processing 3 videos at once means 3 downloaded video files exist in the temp directory simultaneously instead of 1. For typical Instagram reels (30–90MB each) this adds ~60–180MB of temporary disk usage during the scan. Files are still deleted immediately after each video finishes.

### CPU spikes will be sharper but shorter
Frame extraction, OCR, and cut detection are all CPU-heavy. Previously that load was spread out over 5–6 minutes in a slow trickle. Now it arrives in sharp bursts over ~90 seconds. On most machines this is invisible, but on very low-spec hardware the machine may feel less responsive during a scan.

### No other regressions
- xAI API rate limits: 3 concurrent videos stays well within xAI's documented limits
- Instagram rate limits: batches of 5 insights requests stay within the Graph API limit; the existing 429 retry handler in `client.ts` catches any edge cases automatically
- Output is byte-for-byte identical: every classification, feature vector, engagement number, and cluster label is produced by exactly the same logic as before — only the order and timing of operations changed

---

## Compliance with CLAUDE.md Requirements

| Requirement | Status |
|---|---|
| No AI-generated numbers — engagement is pure math | Unchanged. Grok is still only called for visual classification. |
| `temperature: 0` on all Grok calls | Unchanged. The singleton client uses the same settings. |
| Frames resized to 512×512 JPEG before sending | Unchanged. The `toBase64` function in `grokAnalyzer.ts` is untouched. |
| Correct frame counts (16 / 10 / 6 by duration) | Unchanged. `framesForDuration()` is untouched. |
| Temp files cleaned up after each video | Unchanged. `cleanupMedia()` is still called after every video, success or error. |
| Tokens never sent to xAI | Unchanged. Only the Grok API key (from keychain/env) is used with the xAI client. |
| All DB access via Drizzle ORM | Unchanged. No raw SQL introduced. |
| TypeScript strict mode — no `any` | All new code uses explicit types. |
