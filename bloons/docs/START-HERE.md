# START HERE

**Phase:** M: Mac/Windows platform switch (`docs/PLATFORM-FLAG-PLAN.md`). Phase X (Steam pop farm) is paused, see handoff 061. All harness code is in `harness/` (run commands from there); `models/` is for Dom's model.
**Status:** Steps 1-5 done (offline). `Bot(platform="mac"|"win")` (default win), `--platform` on every tool. Mac `Game` matches Windows' API, and the UI-key parity test is green. Tests on the Mac: 130 offline + 3 steam.
**Latest handoff:** [docs/handoffs/063-2026-11-22.md](handoffs/063-2026-11-22.md)

**Next action:** Step 6, live on the Mac (Dom opens BTD6+ on Monkey Meadow, windowed 1470x830): `../.venv/bin/python scripts/smoke_platform.py`, then `scripts/watch_state.py logs/watch --platform mac`.

Run modes: Windows `python tools/drive.py serve` (default) · Mac `python tools/drive.py serve --platform mac`, or `Bot(platform="mac")` in code.
Kill switch: `harness\tools\stop.cmd` on Windows, `sh harness/tools/stop.sh` on the Mac (only Dom deletes harness/logs/STOP).
