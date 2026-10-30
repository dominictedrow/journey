"""Load raw 1-minute CSVs into a validated, UTC-indexed bar frame and cache them as Parquet.

Bar frame contract: DatetimeIndex named ``ts`` in UTC = bar START time, strictly increasing,
columns open/high/low/close/volume as float64.

Supplied files (Date=YYYYMMDD, Timestamp=HH:MM:SS) were verified as UTC by the weekly FX
reopen moving 22:00 -> 21:00 UTC across DST. Bar-start labelling is ASSUMED (vendor convention).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    pass


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Raise DataValidationError unless ``bars`` meets the bar frame contract."""
    if list(bars.columns) != BAR_COLUMNS:
        raise DataValidationError(f"columns must be {BAR_COLUMNS}, got {list(bars.columns)}")
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        raise DataValidationError("index must be a UTC DatetimeIndex")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise DataValidationError("index must be strictly increasing")
    if bars.isna().any().any():
        raise DataValidationError("bars contain NaN")
    o, h, low, c = (bars[k] for k in ("open", "high", "low", "close"))
    if ((h < o) | (h < c) | (low > o) | (low > c) | (h < low)).any():
        raise DataValidationError("OHLC consistency violated")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("non-positive prices")
    return bars


def read_vendor_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"Date": str, "Timestamp": str})
    ts = pd.to_datetime(raw["Date"] + " " + raw["Timestamp"], format="%Y%m%d %H:%M:%S", utc=True)
    bars = pd.DataFrame(
        {
            "open": raw["Open"].astype("float64").to_numpy(),
            "high": raw["High"].astype("float64").to_numpy(),
            "low": raw["Low"].astype("float64").to_numpy(),
            "close": raw["Close"].astype("float64").to_numpy(),
            "volume": raw["Volume"].astype("float64").to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="ts"),
    )
    return validate_bars(bars)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol}_1m.parquet"


def import_csv(csv_path: Path, symbol: str, cache_dir: Path) -> Path:
    """Parse, validate and freeze a vendor CSV as Parquet with provenance metadata."""
    bars = read_vendor_csv(csv_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_path(cache_dir, symbol)
    meta = {
        "symbol": symbol,
        "source_file": csv_path.name,
        "source_sha256": _sha256(csv_path),
        "rows": len(bars),
        "first": bars.index[0].isoformat(),
        "last": bars.index[-1].isoformat(),
        "timezone": "UTC (verified)",
        "bar_label": "start (assumed)",
    }
    table = pa.Table.from_pandas(bars, preserve_index=True)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"trader": json.dumps(meta).encode()}
    )
    pq.write_table(table, out)
    return out


def snapshot_metadata(path: Path) -> dict:
    raw = pq.read_schema(path).metadata or {}
    return json.loads(raw.get(b"trader", b"{}"))


def to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def load_bars(
    cache_dir: Path,
    symbol: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load cached 1-minute bars in [start, end) UTC."""
    bars = pd.read_parquet(cache_path(cache_dir, symbol))
    bars.index = pd.DatetimeIndex(bars.index, name="ts").tz_convert("UTC")
    if start is not None:
        bars = bars[bars.index >= to_utc(start)]
    if end is not None:
        bars = bars[bars.index < to_utc(end)]
    return validate_bars(bars[BAR_COLUMNS])
