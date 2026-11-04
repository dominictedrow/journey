"""Download daily OHLC + adjusted close from Yahoo's public chart endpoint (no account).

Unofficial endpoint; for personal research only. Saves one CSV per ticker to --out
(default ../data/yahoo): date, open, high, low, close, adjclose, volume.
Dates are exchange-local trading dates.

Usage: python3 -I scripts/fetch_yahoo.py SPY IEF GLD ...
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=0&period2=9999999999&interval=1d&events=div%2Csplit"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "yahoo"


def fetch(ticker: str) -> list[list]:
    req = urllib.request.Request(URL.format(t=ticker), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    tz = dt.timezone(dt.timedelta(seconds=res["meta"]["gmtoffset"]))
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    rows = []
    for i, ts in enumerate(res["timestamp"]):
        vals = [q["open"][i], q["high"][i], q["low"][i], q["close"][i], adj[i], q["volume"][i]]
        if any(v is None for v in vals[:5]):
            continue
        rows.append([dt.datetime.fromtimestamp(ts, tz).date().isoformat(), *vals])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for t in args.tickers:
        try:
            rows = fetch(t)
        except Exception as e:  # report and continue
            print(f"{t}: FAILED {e}")
            bad += 1
            continue
        with (args.out / f"{t}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "adjclose", "volume"])
            w.writerows(rows)
        print(f"{t}: {len(rows)} rows {rows[0][0]} -> {rows[-1][0]}", flush=True)
        time.sleep(args.pause)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
