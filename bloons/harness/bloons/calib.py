"""Which calibration the data files come from: 'steam' (Windows, 1920x1080) or 'mac' (BTD6+,
1470x797). Default by OS; BLOONS_CALIB overrides (the offline tests pin 'mac': their fixtures
are Mac frames). Calibrated files have a Mac twin: ui.json -> ui.mac.json, templates/ ->
templates/mac/. Files without a twin (towers.json, ...) are shared."""
import os
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
NAME = os.environ.get("BLOONS_CALIB") or ("steam" if sys.platform == "win32" else "mac")


def path(rel):
    """data/<rel> for this calibration."""
    p = DATA / rel
    if NAME == "mac" and p.suffix == ".json":
        twin = p.with_name(p.name[:-5] + ".mac.json")
        if twin.exists():
            return twin
    return p


TEMPLATES = DATA / "templates" / "mac" if NAME == "mac" else DATA / "templates"
