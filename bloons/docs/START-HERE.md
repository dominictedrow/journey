# START HERE

**Phase:** X: pop farm for the Gift Box on Dom's Steam account. All harness code is in `harness/` (run commands from there); `models/` is for Dom's model.
**Status:** Stopped by Dom in game 3. med4.txt (no farms/village) won 2/2; game 3 stalled after the post-victory menus (see 061). logs/STOP is set.
**Latest handoff:** [docs/handoffs/061-2026-11-15.md](handoffs/061-2026-11-15.md)

**Next action:** When Dom says go: pause -> Restart the open game, `python tools/drive.py serve --difficulty medium` (background), `bash logs/loop.sh logs/med4.txt`. When `logs/gift.png` shows Mermonkey ready: claim it, select Beast Handler, then Dartling.

Kill switch: `harness\tools\stop.cmd` (writes harness/logs/STOP; only Dom deletes it).
