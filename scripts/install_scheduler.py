"""Install the collector scheduler as a login service (component doc section 7).

macOS: writes ``~/Library/LaunchAgents/com.qt.scheduler.plist`` (RunAtLoad, KeepAlive, logs to
``logs/``) and loads it with ``launchctl bootstrap``. Windows (Task Scheduler) is Phase 12.

    uv run python scripts/install_scheduler.py              # install + load (replaces old one)
    uv run python scripts/install_scheduler.py --dry-run    # print the plist only
    uv run python scripts/install_scheduler.py --uninstall
    uv run python scripts/install_scheduler.py --status
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = "com.qt.scheduler"


def venv_python() -> Path:
    env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    venv = Path(env).expanduser() if env else Path.home() / ".venvs" / "quant_trader"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        raise SystemExit(f"venv python not found at {py}; run `uv sync` first")
    return py


def _path_env() -> str:
    dirs = []
    for tool in ("docker", "uv", "git"):  # backup shells out to docker compose
        found = shutil.which(tool)
        if found:
            dirs.append(str(Path(found).parent))
    dirs += ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    return os.pathsep.join(dict.fromkeys(dirs))


def service_log_dir() -> Path:
    """launchd's own stdout/stderr files. Kept off the repo drive: launchd can't open files on an
    external (exFAT) volume when it spawns the job and fails with EX_CONFIG (78). The app's
    real logs still go to ``logs/`` via QT_HOME."""
    return Path.home() / "Library" / "Logs" / "quant_trader"


CAFFEINATE = Path("/usr/bin/caffeinate")


def program(py: Path) -> list[str]:
    """The scheduler, wrapped in ``caffeinate -is`` so the Mac doesn't idle-sleep (-i), or sleep
    at all on AC power (-s), while it runs: sleep stops the news loop and the 5-day gate (D-030).
    caffeinate exits when its child does, so KeepAlive still restarts a crash. Closing the lid
    still sleeps the Mac unless it's in clamshell mode with an external display."""
    cmd = [str(py), "-m", "qt.orchestrator.scheduler"]
    return [str(CAFFEINATE), "-is", *cmd] if CAFFEINATE.exists() else cmd


def plist(py: Path) -> dict:
    logs = service_log_dir()
    return {
        "Label": LABEL,
        "ProgramArguments": program(py),
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {
            "QT_PROFILE": "paper",
            "QT_HOME": str(REPO),
            "PATH": _path_env(),
            "PYTHONUNBUFFERED": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,  # launchd waits this long between crash restarts
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "scheduler.out.log"),
        "StandardErrorPath": str(logs / "scheduler.err.log"),
    }


def agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=check)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _wait_unloaded(timeout_s: float = 30.0) -> bool:
    """bootout returns before the old process has exited; bootstrapping during that window
    fails with "5: Input/output error". Wait until launchd no longer knows the label."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _launchctl("print", f"{_domain()}/{LABEL}", check=False).returncode != 0:
            return True
        time.sleep(0.5)
    return False


def install_macos(dry_run: bool) -> int:
    data = plistlib.dumps(plist(venv_python())).decode()
    if dry_run:
        print(data)
        return 0
    p = agent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    service_log_dir().mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{LABEL}", check=False)  # replace a running copy
    if not _wait_unloaded():
        print(f"{LABEL} did not unload within 30 s; not reinstalling", file=sys.stderr)
        return 1
    p.write_text(data, encoding="utf-8")
    r = _launchctl("bootstrap", _domain(), str(p), check=False)
    if r.returncode != 0:
        print(f"launchctl bootstrap failed: {r.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"installed and loaded {p}")
    return 0


def uninstall_macos() -> int:
    _launchctl("bootout", f"{_domain()}/{LABEL}", check=False)
    p = agent_path()
    if p.exists():
        p.unlink()
    print(f"unloaded {LABEL}; removed {p}")
    return 0


def status_macos() -> int:
    r = _launchctl("print", f"{_domain()}/{LABEL}", check=False)
    if r.returncode != 0:
        print(f"{LABEL} is not loaded")
        return 1
    keep = ("state =", "pid =", "last exit code", "runs =")
    print("\n".join(ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith(keep)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install the qt collector scheduler service.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if sys.platform == "darwin":
        if args.uninstall:
            return uninstall_macos()
        if args.status:
            return status_macos()
        return install_macos(args.dry_run)
    if os.name == "nt":
        raise NotImplementedError("Windows Task Scheduler registration is Phase 12")
    raise NotImplementedError(f"unsupported platform {sys.platform}")


if __name__ == "__main__":
    sys.exit(main())
