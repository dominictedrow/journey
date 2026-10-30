import json

import pandas as pd

from tests.conftest import random_walk
from trader.competition import CompetitionConfig, holdout_start, rank, run_competition
from trader.data.loaders import import_csv


def _write_vendor_csv(bars, path):
    out = pd.DataFrame(
        {
            "Date": bars.index.strftime("%Y%m%d"),
            "Timestamp": bars.index.strftime("%H:%M:%S"),
            "Open": bars.open,
            "High": bars.high,
            "Low": bars.low,
            "Close": bars.close,
            "Volume": bars.volume,
        }
    )
    out.to_csv(path, index=False)


def test_competition_respects_sealed_holdout(tmp_path):
    bars = random_walk(60 * 24 * 120, seed=2, start="2024-01-01 00:00", step=2.0)
    bars = bars[bars.index.dayofweek < 5]
    _write_vendor_csv(bars, tmp_path / "S.csv")
    import_csv(tmp_path / "S.csv", "USA500IDXUSD", tmp_path / "cache")
    cfg = CompetitionConfig(
        data_dir=tmp_path / "cache",
        report_dir=tmp_path / "reports",
        artifact_dir=tmp_path / "artifacts",
        instruments=[{"symbol": "USA500IDXUSD"}],
        modes=[{"name": "no_trade"}, {"name": "always_long"}, {"name": "random"}],
        cost_multipliers=[1.0, 2.0],
        holdout_months=1,
    )
    table, report = run_competition(cfg, log=lambda *_: None)
    boundary = holdout_start(cfg, "USA500IDXUSD")
    assert boundary == pd.Timestamp("2024-03-01", tz="UTC")
    ledgers = list((tmp_path / "artifacts").glob("competition_*/*.csv"))
    for f in ledgers:
        t = pd.read_csv(f, parse_dates=["exit_time"])
        assert (t.exit_time < boundary).all()
    assert len(table) == 6
    assert report.exists() and "always_long" in report.read_text()
    # Re-running never moves the seal.
    sealed = json.loads((tmp_path / "reports/holdout.json").read_text())
    run_competition(cfg, log=lambda *_: None)
    assert json.loads((tmp_path / "reports/holdout.json").read_text()) == sealed
    lines = (tmp_path / "artifacts/experiments.jsonl").read_text().splitlines()
    assert len(lines) == 12


def test_rank_prefers_better_risk_adjusted_mode():
    t = pd.DataFrame(
        {
            "expectancy_r": [0.1, -0.1],
            "sharpe": [1.0, -0.5],
            "sortino": [1.5, -0.6],
            "max_drawdown": [0.05, 0.2],
        },
        index=["good", "bad"],
    )
    assert rank(t).to_dict() == {"good": 1.0, "bad": 2.0}
