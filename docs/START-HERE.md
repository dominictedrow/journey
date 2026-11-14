# START HERE

**Phase:** X (XP farm on Dom's Steam account), Claude plays by hand via `harness/tools/drive.py`. All harness code is in `harness/` (run commands from there); `models/` is for Dom's model.
**Status:** Hard Meadow **Magic Monkeys Only** (`new monkey_meadow hard magic_only`): the Ninja + Alchemist build (`logs/g4*.sh`) held r40 at 99 lives, lost r63. Dom stopped the session at the defeat screen; XP not spent.
**Latest handoff:** [docs/handoffs/059-2026-11-11.md](handoffs/059-2026-11-11.md)

**Next action:** When Dom says go: start `python tools/drive.py serve --difficulty hard` (background), go Home, `bash logs/xpgrid.sh alchemist` / `ninja_monkey`, re-run g4 with an Arcane Spike Wizard (4-0-2) by r55 and a spend loop that never holds cash (RESEARCH 2026-11-11).

Kill switch: `harness\tools\stop.cmd` (writes harness/logs/STOP; only Dom deletes it).
