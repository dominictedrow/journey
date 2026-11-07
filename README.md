# asl: sign-language recognition

> **Part of [journey](https://github.com/dominictedrow/journey)**, Dom's ML and software portfolio.
>
> **How it was built: 🤖 vibe coded.** AI coding agents (mostly Claude Code) wrote the code. I picked the goal, made the design calls, and steered and tested the work.
>
> **When:** 2026-07 → 2026-08 · **Branch:** `vibe/asl` · [All vibe projects](https://github.com/dominictedrow/journey/tree/vibe-projects)

Recognizes American Sign Language from video. MediaPipe hand/body landmarks are pulled from WLASL, MS-ASL and ASL-Citizen clips, then a sequence classifier is trained on them. Live inference segments gestures from a webcam and speaks the result.

Pipeline code: `src/` (ingest → dataset → train → infer). The original repo vendored the third-party `youtube-dl` downloader for fetching source videos; it is left out here because it is not my code (install it separately to re-download the datasets).
