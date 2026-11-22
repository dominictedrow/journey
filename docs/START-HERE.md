# START HERE

**Phase:** M: Mac/Windows platform switch (`docs/PLATFORM-FLAG-PLAN.md`). Phase X (Steam pop farm) is paused, see handoff 061. All harness code is in `harness/` (run commands from there); `models/` is for Dom's model.
**Status:** Steps 1-3 done. `Bot(platform="mac"|"win")` (default win), `--platform` on every tool, `BLOONS_PLATFORM` env. Offline tests are green on the Mac (128 + 3 steam).
**Latest handoff:** [docs/handoffs/062-2026-11-22.md](handoffs/062-2026-11-22.md)

**Next action:** PLATFORM-FLAG-PLAN step 4: Mac `Game` parity (`relaunch`, `read_number`, `width/height`, `keep_awake`), `tools/stop.sh`, AST parity test.

Run modes: Windows `python tools/drive.py serve` (default) · Mac `python tools/drive.py serve --platform mac`, or `Bot(platform="mac")` in code.
Kill switch: `harness\tools\stop.cmd` on Windows, `touch harness/logs/STOP` on the Mac (only Dom deletes it).
