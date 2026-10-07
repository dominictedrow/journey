"""launchd entry point for a scheduled skill: run it, then cache its envelope.

Run (from the repo root, inside the c_work env):
    python scripts/run_scheduled.py <skill_id>

The app's Home tab and skill tabs read results from the `skill_runs` table,
which app/main.js only writes for in-app runs. This wrapper runs
`python -m skills.<id>` exactly as the scheduled-skill contract in plan.md
describes, then upserts the envelope into `skill_runs` the same way
db.recordSkillRun does — including a failure envelope when the skill crashes
or prints something that isn't JSON — so scheduled results show up in the UI.
The envelope is still echoed to stdout so the launchd .out.log keeps working.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from skills._lib import store  # noqa: E402


def _failure(skill_id, error):
    return {
        "skill": skill_id,
        "ok": False,
        "ran_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": "Skill failed to produce a valid envelope",
        "sections": [],
        "error": error,
    }


def _record(skill_id, envelope):
    conn = store.connect()
    conn.execute(
        "INSERT INTO skill_runs (skill_id, envelope, ok, ran_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(skill_id) DO UPDATE SET "
        "envelope = excluded.envelope, ok = excluded.ok, ran_at = excluded.ran_at",
        (skill_id, json.dumps(envelope), 1 if envelope.get("ok") else 0),
    )
    conn.commit()
    conn.close()


def main():
    if len(sys.argv) != 2:
        print(f"usage: python {Path(__file__).name} <skill_id>", file=sys.stderr)
        sys.exit(1)
    skill_id = sys.argv[1]

    # stderr passes straight through to launchd's .err.log; only stdout (the
    # envelope) is captured.
    result = subprocess.run(
        [sys.executable, "-m", f"skills.{skill_id}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        envelope = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        envelope = _failure(skill_id, f"process exited with code {result.returncode}")

    _record(skill_id, envelope)
    print(json.dumps(envelope))
    sys.exit(0 if envelope.get("ok") else 1)


if __name__ == "__main__":
    main()
