"""Install/uninstall launchd agents for cwork's scheduled skills.

Run:
    python scripts/schedule.py install     # write + load a plist per scheduled skill
    python scripts/schedule.py uninstall   # unload + remove those plists
    python scripts/schedule.py status      # launchctl list | grep cwork

Reads `app/registry.js` (via `node -e`) so the registry stays the single
source of truth for which skills are scheduled and when — this script adds
no scheduling info of its own. Only entries with `kind: 'scheduled'` and a
`schedule` field get an agent; on-demand skills and Assignments (no schedule
at all — it's manual CRUD) are skipped.

Each agent runs `conda run -n c_work python -m skills.<id>` with no CLI
input, exactly the scheduled-skill invocation contract in plan.md. The conda
executable's absolute path is resolved here (via PATH, at install time) and
baked into the plist, since launchd agents don't inherit a login shell's PATH.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = Path.home() / "Library" / "Application Support" / "cwork" / "logs"
LABEL_PREFIX = "com.cwork"

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{conda_path}</string>
        <string>run</string>
        <string>-n</string>
        <string>c_work</string>
        <string>python</string>
        <string>-m</string>
        <string>skills.{skill_id}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{repo_root}</string>
    <key>StartCalendarInterval</key>
    <dict>
{calendar_entries}
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/{skill_id}.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/{skill_id}.err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def _load_registry():
    result = subprocess.run(
        ["node", "-e", "console.log(JSON.stringify(require('./app/registry.js')))"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _scheduled_skills():
    return [s for s in _load_registry() if s.get("kind") == "scheduled" and s.get("schedule")]


def _plist_path(skill_id):
    return LAUNCH_AGENTS_DIR / f"{LABEL_PREFIX}.{skill_id}.plist"


def _uid():
    return subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()


def _agent_target(skill_id):
    return f"gui/{_uid()}/{LABEL_PREFIX}.{skill_id}"


def _render_plist(skill_id, schedule, conda_path):
    fields = {"Hour": schedule.get("hour"), "Minute": schedule.get("minute", 0)}
    if "day" in schedule:
        fields["Day"] = schedule["day"]
    calendar_entries = "\n".join(
        f"        <key>{k}</key>\n        <integer>{v}</integer>"
        for k, v in fields.items()
        if v is not None
    )
    return PLIST_TEMPLATE.format(
        label=f"{LABEL_PREFIX}.{skill_id}",
        conda_path=conda_path,
        skill_id=skill_id,
        repo_root=REPO_ROOT,
        calendar_entries=calendar_entries,
        log_dir=LOG_DIR,
    )


def install():
    conda_path = shutil.which("conda")
    if not conda_path:
        print("conda not found on PATH — activate/init conda first, then re-run.", file=sys.stderr)
        sys.exit(1)

    skills = _scheduled_skills()
    if not skills:
        print("No scheduled skills found in app/registry.js.", file=sys.stderr)
        return

    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    for skill in skills:
        skill_id = skill["id"]
        plist_path = _plist_path(skill_id)
        plist_path.write_text(_render_plist(skill_id, skill["schedule"], conda_path))

        subprocess.run(["launchctl", "bootout", _agent_target(skill_id)], capture_output=True)
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{_uid()}", str(plist_path)],
            check=True,
        )
        print(f"installed + loaded {LABEL_PREFIX}.{skill_id} ({plist_path})", file=sys.stderr)


def uninstall():
    skills = _scheduled_skills()
    for skill in skills:
        skill_id = skill["id"]
        subprocess.run(["launchctl", "bootout", _agent_target(skill_id)], capture_output=True)
        plist_path = _plist_path(skill_id)
        if plist_path.exists():
            plist_path.unlink()
        print(f"unloaded + removed {LABEL_PREFIX}.{skill_id}", file=sys.stderr)


def status():
    subprocess.run(["bash", "-c", f"launchctl list | grep {LABEL_PREFIX} || true"])


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "uninstall", "status"):
        print(f"usage: python {Path(__file__).name} install|uninstall|status", file=sys.stderr)
        sys.exit(1)

    {"install": install, "uninstall": uninstall, "status": status}[sys.argv[1]]()


if __name__ == "__main__":
    main()
