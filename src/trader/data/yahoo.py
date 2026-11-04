"""Daily prices from Yahoo's public chart endpoint (no account; unofficial, personal research).

CSV layout per ticker: date, open, high, low, close, adjclose, volume (exchange-local dates).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
    "?period1=0&period2=9999999999&interval=1d&events=div%2Csplit"
)
COLUMNS = ["date", "open", "high", "low", "close", "adjclose", "volume"]


def fetch(ticker: str, timeout: float = 30) -> list[list]:
    req = urllib.request.Request(URL.format(t=ticker), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
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


def download(tickers: list[str], out: Path, pause: float = 1.0, log=print) -> list[str]:
    """Fetch and save each ticker; returns the tickers that failed."""
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for t in tickers:
        try:
            rows = fetch(t)
        except Exception as e:  # report and continue
            log(f"{t}: FAILED {e}")
            failed.append(t)
            continue
        with (out / f"{t}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)
        log(f"{t}: {len(rows)} rows {rows[0][0]} -> {rows[-1][0]}")
        time.sleep(pause)
    return failed


def load(ticker: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{ticker}.csv", parse_dates=["date"]).set_index("date")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates must be unique and increasing")
    return df
