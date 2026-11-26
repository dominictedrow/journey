"""Send one test push through the real alert path (exit gate check, D-016).

uv run python scripts/alert_test.py            # info push via alert()
uv run python scripts/alert_test.py --error    # log.error -> critical push via the root handler
"""

import argparse
import logging
import sys

from qt.common import alerts, config
from qt.common import logging as qtlog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--error", action="store_true", help="raise a deliberate logged error")
    args = ap.parse_args()
    settings = config.load(profile="paper")  # dev/backtest have pushes off
    qtlog.setup(settings)
    if settings.secrets.ntfy_topic is None:
        print("NTFY_TOPIC is not set in .env", file=sys.stderr)
        return 1
    if args.error:
        logging.getLogger("qt.alert_test").error("deliberate_test_error: exit-gate check, ignore")
    else:
        ok = alerts.alert(
            "info", "alert_test", "QT test alert", "If you can read this, pushes work."
        )
        print("delivered" if ok else "NOT delivered (see logs)")
    alerts.get().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
