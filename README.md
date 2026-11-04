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

## Rebalancer (plan-only, D-031)

Keeps an account at a target allocation (default 60% SPY / 40% IEF). It **plans trades and never sends them**: no broker connection, no account access. You place the trades yourself.

```bash
cp configs/holdings.example.toml configs/holdings.toml   # gitignored; fill in your cash and shares
uv run trader rebalance --refresh          # fresh prices, then a plan
uv run trader rebalance --no-sell          # invest new cash only, never sell
uv run trader rebalance-backtest           # compare rebalancing rules on 2005-2025
```

- **Rule:** rebalance only when a weight is more than 5 percentage points off target (`configs/rebalance.toml`). Trades under $50 are skipped. Anything you hold that isn't in the allocation is flagged for sale.
- **Safety:** the planner halts with NO TRADE on a missing price, a price more than 4 days old, or a one-day move over 20% (likely bad data). It warns when the account is 20% below the peak you've recorded. It never plans shorts, margin, or sells larger than what you hold.
- **History (60/40, 2005–2025, 5 bp per trade):** the ±5% band rebalanced 16 times (8% turnover a year) with a 31.8% max drawdown. Never rebalancing looks similar on paper, but by 2025 it had drifted to 86% stocks.
