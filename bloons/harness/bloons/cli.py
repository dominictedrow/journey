"""--platform mac|win for every tool (default win). Call pre_parse() before the tool loads any
harness module:

    from bloons import cli; cli.pre_parse()
    from bloons.harness import BloonsHarness      # now loads that platform's calibration

It takes the flag out of sys.argv (so each tool's own argparse never sees it) and hands it to
calib.use(). Without the flag, BLOONS_PLATFORM or the default (win) stays in effect.
"""
import sys

from . import calib

FLAG = "--platform"
HELP = "win (Steam BTD6, default) or mac (BTD6+); BLOONS_PLATFORM also works"


def _take(argv):
    """Remove the flag from argv in place; return its value or None (last one wins)."""
    value, i = None, 1
    while i < len(argv):
        a = argv[i]
        if a == FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{FLAG} needs a value: win or mac")
            value = argv[i + 1]
            del argv[i:i + 2]
        elif a.startswith(FLAG + "="):
            value = a.split("=", 1)[1]
            del argv[i]
        else:
            i += 1
    return value


def pre_parse(argv=None):
    """Consume `--platform X` / `--platform=X` from argv (default sys.argv) and switch to it.
    Returns the platform in effect ('mac' | 'win')."""
    value = _take(sys.argv if argv is None else argv)
    try:
        return calib.use(value)
    except ValueError as e:
        raise SystemExit(str(e))


def add_argument(parser):
    """List --platform in a tool's --help (pre_parse already consumed it)."""
    parser.add_argument(FLAG, choices=("win", "mac"), help=HELP)


def describe():
    return calib.describe()
