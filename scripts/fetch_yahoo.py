"""Download daily prices from Yahoo's public chart endpoint (no account). Thin wrapper.

Usage: uv run python scripts/fetch_yahoo.py SPY IEF GLD ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trader.data.yahoo import download

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "yahoo"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()
    return 1 if download(args.tickers, args.out, args.pause) else 0


if __name__ == "__main__":
    sys.exit(main())
