"""Which platform the harness drives and which calibration the data files come from.

PLATFORM: 'win' (Steam BTD6, the default) or 'mac' (BTD6+, Apple Arcade). Set it with
Bot(platform=...), a tool's --platform flag (bloons/cli.py) or BLOONS_PLATFORM ('steam' is
accepted for 'win'). Both call use() before anything reads a data file; once one has been
read the platform is fixed for the process, and use() with another one raises.

NAME, the calibration: 'steam' (1920x1080) for win, 'mac' (1470x797) for mac. BLOONS_CALIB
overrides it alone (the offline tests pin 'mac': their fixtures are Mac frames). Calibrated files
have a Mac twin: ui.json -> ui.mac.json, templates/ -> templates/mac/. Files without a twin
(towers.json, ...) are shared."""
import os
from pathlib import Path

DATA = Path(__file__).parent / "data"
PLATFORMS = ("win", "mac")
DEFAULT = "win"
_ALIASES = {"mac": "mac", "win": "win", "windows": "win", "steam": "win"}


def resolve_platform(value=None):
    """'mac' | 'win' from a flag/env/argument value; None -> DEFAULT. Raises ValueError on junk."""
    if not value:
        return DEFAULT
    p = _ALIASES.get(str(value).strip().lower())
    if p is None:
        raise ValueError(f"unknown platform {value!r}: use one of {', '.join(PLATFORMS)}")
    return p


def resolve_calib(platform, override=None):
    """Calibration name for a platform; `override` (BLOONS_CALIB) wins."""
    if override:
        if override not in ("mac", "steam"):
            raise ValueError(f"unknown BLOONS_CALIB {override!r}: use mac or steam")
        return override
    return "steam" if platform == "win" else "mac"


def _set(platform):
    global PLATFORM, NAME, TEMPLATES
    PLATFORM = platform
    NAME = resolve_calib(platform, os.environ.get("BLOONS_CALIB"))
    TEMPLATES = DATA / "templates" / "mac" if NAME == "mac" else DATA / "templates"


_used = False                      # True once path() has handed out a data file
_set(resolve_platform(os.environ.get("BLOONS_PLATFORM")))


def use(platform):
    """Switch to 'mac' | 'win' (None: keep the current one). Only before any data is loaded;
    after that a different platform raises RuntimeError. Returns the platform in effect."""
    if platform is None:
        return PLATFORM
    p = resolve_platform(platform)
    if p != PLATFORM:
        if _used:
            raise RuntimeError(f"platform={p!r} asked for, but the harness already loaded {PLATFORM!r} "
                               "data in this process; pick the platform before anything else "
                               "(first Bot(platform=...), or the --platform flag)")
        _set(p)
    os.environ["BLOONS_PLATFORM"] = p      # child processes follow
    return p


def path(rel, write=False):
    """data/<rel> for this calibration. Reading on mac falls back to the shared file when there's
    no .mac twin; `write=True` always names the twin on mac, so Mac output never lands in a
    Steam file."""
    global _used
    _used = True
    p = DATA / rel
    if NAME == "mac" and p.suffix == ".json":
        twin = p.with_name(p.name[:-5] + ".mac.json")
        if write or twin.exists():
            return twin
    return p


def describe():
    """One line for run logs: which mode this process is in."""
    return (f"platform={PLATFORM} calib={NAME} "
            f"templates={TEMPLATES.relative_to(DATA).as_posix()}")
