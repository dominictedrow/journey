import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars, random_walk
from trader.data.loaders import (
    DataValidationError,
    import_csv,
    load_bars,
    snapshot_metadata,
    validate_bars,
)
from trader.data.quality import classify_gaps
from trader.data.resample import resample_bars
from trader.sessions import rth_mask

CSV = """Date,Timestamp,Open,High,Low,Close,Volume
20240102,14:30:00,1.10,1.12,1.09,1.11,10
20240102,14:31:00,1.11,1.13,1.10,1.12,5
"""


def test_import_and_load_roundtrip(tmp_path):
    src = tmp_path / "X.csv"
    src.write_text(CSV)
    out = import_csv(src, "X", tmp_path / "cache")
    meta = snapshot_metadata(out)
    assert meta["rows"] == 2 and len(meta["source_sha256"]) == 64
    bars = load_bars(tmp_path / "cache", "X")
    assert str(bars.index.tz) == "UTC"
    assert bars.index[0] == pd.Timestamp("2024-01-02 14:30", tz="UTC")
    assert len(load_bars(tmp_path / "cache", "X", start="2024-01-02 14:31")) == 1


def test_validation_rejects_bad_ohlc():
    bars = make_bars([(1.0, 0.9, 0.8, 1.0)])
    with pytest.raises(DataValidationError):
        validate_bars(bars)


def test_validation_rejects_naive_index():
    bars = make_bars([(1.0, 1.1, 0.9, 1.0)])
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(DataValidationError):
        validate_bars(bars)


def test_hourly_resample_aggregates():
    bars = random_walk(180, start="2024-01-02 00:00")
    h = resample_bars(bars, "1h")
    assert len(h) == 3
    first = bars.iloc[:60]
    assert h.iloc[0][["open", "high", "low", "close"]].tolist() == [
        first.open.iloc[0],
        first.high.max(),
        first.low.min(),
        first.close.iloc[-1],
    ]
    assert h["end"].iloc[0] == pd.Timestamp("2024-01-02 01:00", tz="UTC")


@pytest.mark.parametrize(("day", "roll_utc"), [("2024-01-10", "22:00"), ("2024-07-10", "21:00")])
def test_daily_bars_roll_at_1700_new_york(day, roll_utc):
    bars = random_walk(48 * 60, start=f"{day} 00:00")
    d = resample_bars(bars, "1D")
    roll = pd.Timestamp(f"{day} {roll_utc}", tz="UTC")
    assert roll in d.index
    assert d.loc[roll, "end"] == roll + pd.Timedelta("1D")
    before = d.index[d.index < roll][-1]
    assert d.loc[before, "end"] == roll


def test_daily_bar_spanning_dst_change_is_23_hours():
    bars = random_walk(4 * 24 * 60, start="2024-03-08 00:00")  # US DST starts 2024-03-10
    d = resample_bars(bars, "1D")
    lengths = (d["end"] - d.index).dt.total_seconds() / 3600
    assert 23.0 in set(lengths)


def test_rth_mask_handles_dst_and_early_close():
    idx = pd.DatetimeIndex(
        [
            "2024-01-10 14:29",
            "2024-01-10 14:30",
            "2024-07-10 13:30",
            "2024-07-10 20:00",
            "2024-11-29 17:59",
            "2024-11-29 18:00",
            "2024-12-25 15:00",
        ],
        tz="UTC",
    )
    assert rth_mask(idx).tolist() == [False, True, True, False, True, False, False]


def test_gap_classification():
    idx = pd.DatetimeIndex(
        [
            "2024-01-05 21:59",
            "2024-01-07 22:00",
            "2024-01-07 22:20",
            "2024-01-07 23:50",
            "2024-01-08 03:00",
        ],
        tz="UTC",
    )
    g = classify_gaps(idx)
    assert g["kind"].tolist() == ["weekend", "daily_halt", "other"]
    assert np.all(g["gap"] > pd.Timedelta("30min"))
