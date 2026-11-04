"""Download Dukascopy 1-minute BID and ASK candles and build a merged bid/ask dataset.

Stdlib-only fetcher against Dukascopy's public datafeed (no account needed).
Data is free for personal research; check Dukascopy's terms before other uses.

Layout produced under --out (default: ../data/dukascopy):
  raw/<INST>/<YYYY>/<MM>/<DD>_<BID|ASK>.bi5   cached daily files (resumable)
  <INST>_1m_bidask.parquet                    merged output (build step)

Feed facts verified 2026-11-02 against the supplied bid files:
  * one LZMA file per UTC day per side; month in the URL is 0-indexed
  * record = >5if: seconds-from-midnight, open, close, low, high, volume
  * timestamps are UTC, bar-START labelled
  * prices are integers / point (EURUSD 1e5, USA500IDXUSD 1e3)

Usage:
  python3 -I scripts/fetch_dukascopy.py fetch [--inst EURUSD] [--workers 4]
  uv run python scripts/fetch_dukascopy.py build
"""

from __future__ import annotations

import argparse
import datetime as dt
import lzma
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INSTRUMENTS = {
    # name: (first day, price divisor)
    "EURUSD": (dt.date(2003, 5, 4), 1e5),
    "USA500IDXUSD": (dt.date(2011, 9, 18), 1e3),
}
SIDES = ("BID", "ASK")
URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "dukascopy"


def raw_path(out: Path, inst: str, day: dt.date, side: str) -> Path:
    return out / "raw" / inst / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}_{side}.bi5"


def fetch_one(
    out: Path, inst: str, day: dt.date, side: str, retries: int = 10, pause: float = 0.0
) -> str:
    dest = raw_path(out, inst, day, side)
    if dest.exists():
        return "cached"
    url = URL.format(inst=inst, y=day.year, m=day.month - 1, d=day.day, side=side)
    for attempt in range(retries):
        time.sleep(pause)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                body = b""
                break
            err = e
            if e.code == 429:  # rate limited: back off hard
                time.sleep(60 * (attempt + 1))
                continue
        except Exception as e:  # network hiccup
            err = e
        time.sleep(2**attempt)
    else:
        return f"FAIL {url}: {err}"
    if body:
        lzma.decompress(body)  # validate before caching
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return "ok"


def days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() != 5:  # no FX/CFD trading on Saturdays (UTC)
            yield d
        d += dt.timedelta(days=1)


def cmd_fetch(args) -> int:
    end = args.end or (dt.date.today() - dt.timedelta(days=1))
    insts = [args.inst] if args.inst else list(INSTRUMENTS)
    jobs = [(i, d, s) for i in insts for d in days(INSTRUMENTS[i][0], end) for s in SIDES]
    print(f"{len(jobs)} day-files to check", flush=True)
    fails, done = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_one, args.out, *j, pause=args.pause) for j in jobs]
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if res.startswith("FAIL"):
                fails.append(res)
            if done % 2000 == 0:
                print(f"{done}/{len(jobs)} ({len(fails)} failed)", flush=True)
    print(f"done: {done} checked, {len(fails)} failed")
    for f in fails[:20]:
        print(f)
    return 1 if fails else 0


DTYPE = [
    ("t", ">i4"),
    ("open", ">i4"),
    ("close", ">i4"),
    ("low", ">i4"),
    ("high", ">i4"),
    ("volume", ">f4"),
]


def decode(path: Path, day: dt.date):
    import numpy as np

    body = path.read_bytes()
    if not body:
        return None
    arr = np.frombuffer(lzma.decompress(body), dtype=DTYPE)
    base = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
    return arr, base


def cmd_build(args) -> int:
    import numpy as np
    import pandas as pd

    end = args.end or (dt.date.today() - dt.timedelta(days=1))
    insts = [args.inst] if args.inst else list(INSTRUMENTS)
    for inst in insts:
        start, div = INSTRUMENTS[inst]
        frames = {}
        for side in SIDES:
            parts, bases = [], []
            for d in days(start, end):
                p = raw_path(args.out, inst, d, side)
                if not p.exists():
                    print(f"missing {p}; run fetch first", file=sys.stderr)
                    return 1
                got = decode(p, d)
                if got is not None:
                    parts.append(got[0])
                    bases.append(np.full(len(got[0]), got[1], dtype=np.int64))
            a = np.concatenate(parts)
            s = side.lower()
            df = pd.DataFrame(
                {
                    f"{s}_open": a["open"].astype(np.float64) / div,
                    f"{s}_high": a["high"].astype(np.float64) / div,
                    f"{s}_low": a["low"].astype(np.float64) / div,
                    f"{s}_close": a["close"].astype(np.float64) / div,
                    f"{s}_volume": a["volume"].astype(np.float64),
                },
                index=np.concatenate(bases) + a["t"].astype(np.int64),
            )
            frames[side] = df
        m = frames["BID"].join(frames["ASK"], how="inner")
        # Dukascopy fills no-tick minutes with flat candles; drop minutes with no ticks on
        # either side.
        m = m[(m["bid_volume"] > 0) | (m["ask_volume"] > 0)]
        m.index = pd.to_datetime(m.index, unit="s", utc=True)
        m.index.name = "timestamp_utc_bar_start"
        bad = int((m["ask_close"] < m["bid_close"]).sum())
        dest = args.out / f"{inst}_1m_bidask.parquet"
        m.to_parquet(dest)
        spread = (m["ask_close"] - m["bid_close"]).median()
        print(
            f"{inst}: {len(m):,} bars {m.index[0]} -> {m.index[-1]}; "
            f"median close spread {spread:.6g}; ask<bid bars {bad}; -> {dest}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "build"])
    ap.add_argument("--inst", choices=list(INSTRUMENTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--pause", type=float, default=0.25, help="seconds each worker waits before a request"
    )
    ap.add_argument("--end", type=dt.date.fromisoformat, help="last UTC day (default: yesterday)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    return cmd_fetch(args) if args.command == "fetch" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
