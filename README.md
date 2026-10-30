# trader

Research trading bot: one framework, switchable strategy **modes**, ranked by an after-cost
**competition**. Backtesting only — no broker connections or orders exist in this code.
Plan and rules: `START_HERE.md`, `MASTER_PLAN.md`, `AGENTS.md`.

## Setup
The project sits on an exFAT drive, which breaks virtualenvs, so the venv lives on the internal disk:

```sh
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/trader"
uv sync
uv run pytest && uv run ruff check src tests
```

## Run
```sh
uv run trader import-data ../data/EURUSD.csv --symbol EURUSD          # validate + freeze to Parquet
uv run trader import-data ../data/USA500IDXUSD.csv --symbol USA500IDXUSD
uv run trader data-report EURUSD USA500IDXUSD                         # reports/data_quality.md
uv run trader compete                                                 # configs/competition.toml
```

Outputs: `reports/competition_<stamp>.md|csv` (committed), trade ledgers and
`artifacts/experiments.jsonl` (gitignored). The last 12 months are sealed in `reports/holdout.json`
and only loaded with `--unseal-holdout` (once, for the final evaluation).

## Adding a mode
Subclass `trader.modes.base.Mode`, set `name`, `provenance`, `timeframe`, implement `on_bar` (and
optionally a causal `prepare`), register it in `trader/modes/__init__.py`, add it to the config.
Every `Enter` needs a `stop`; the risk layer sizes the trade and vetoes anything without one.
`tests/test_no_lookahead.py` automatically checks every registered mode for look-ahead.

## Layout
| Path | Role |
|---|---|
| `src/trader/data/` | CSV loader + validation, Parquet snapshots, causal resampling, quality report |
| `src/trader/sessions.py` | NYSE regular sessions, holidays, early closes (UTC) |
| `src/trader/instruments.py` | Contract specs and **assumed** costs (MES on the S&P CFD, EUR/USD) |
| `src/trader/modes/` | Mode contract, indicators, baselines |
| `src/trader/risk.py` | Mandatory stop, policy sizing, min-size veto, daily loss breaker |
| `src/trader/backtest/` | Event-driven engine (next-bar fills, 1-min stop checks) and metrics |
| `src/trader/competition.py` | Runs every mode × instrument × cost multiplier, ranks, writes report |
