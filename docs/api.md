# API

Base URL: `http://<host>:8000`. Interactive docs at `/docs` (Swagger UI) and `/redoc`.

Numeric money/metric fields are serialized as JSON strings (exact `Decimal` on the wire).

## Implemented (Phase 0)

### `GET /health`

Liveness/readiness probe used by the frontend's "Test connection" button.

```json
{ "status": "ok", "version": "0.1.0", "execution_mode": "paper" }
```

## Implemented (Phase 2)

### `GET /wallets`

Wallet explorer list. Query params: `tracked` (bool), `source` (exact `discovery_source`),
`search` (address substring), `sort` (`total_volume` | `total_trades` | `last_active` |
`discovered_at` | `roi_lifetime` | `realized_pnl` | `win_rate` | `sharpe`), `order`
(`asc`|`desc`), `limit` (≤200), `offset`. Returns `{ items, total, limit, offset }`; each item
embeds the latest `metrics` row (or `null`).

### `GET /wallets/{address}`

Full profile: wallet + metrics + open positions + 50 most recent indexed trades. 404 if unknown.

### `PATCH /wallets/{address}`

Body: `{ "is_tracked"?: bool, "copy_enabled"?: bool }`. Toggles tracking/copy flags.

### `POST /wallets/{address}/refresh-metrics`

Recomputes Module-3 metrics from already-indexed trades, inline; returns the wallet with
fresh metrics and refreshed rollups (`total_volume`, `total_trades`, activity window).

### `POST /wallets/{address}/backfill` (202)

Queues Celery task `analytics.backfill_and_compute` (full on-chain backfill, then metrics).
Returns `{ "task_id" }`. Requires `POLYGON_RPC_URLS`.

### `GET /discovery`

Candidate-pool status: `{ total_wallets, tracked_wallets, last_discovered_at, by_source }`.
`source: null` counts wallets auto-created by the chain indexer with no attribution.

### `POST /discovery/run` (202)

Queues Celery task `discovery.run` (leaderboard + market-holders sweep; also on the beat
schedule every 6 h). Returns `{ "task_id" }`.

## Implemented (Phase 3)

### `GET /rankings`

Latest ranking generation. Query params: `category` (`overall` | `politics` | `crypto` |
`sports` | `economics` | `hft` | `long_term` | `early_info` | `low_risk`, default `overall`),
`limit` (≤200). Returns `{ computed_at, category, categories, items }`; `categories` lists
per-category entry counts for the generation, and each item is
`{ rank, score, wallet, score_detail }` where `score_detail` carries the composite's
components (`historical` 0.40 / `recent` 0.30 / `copy_quality` 0.20 / `risk` 0.10) plus the
`explanation` JSON (each input's raw value, cohort-normalized score, weight, contribution).
`computed_at` is `null` until the first rebuild.

### `POST /rankings/rebuild`

Rescores every wallet with computed metrics against the cohort (Module 4) and writes a
fresh rankings generation per category (Module 5), inline. Also on the beat schedule
hourly as Celery task `ranking.rebuild`. Returns `{ computed_at, wallets_scored, categories }`.

## Implemented (Phase 4)

### `POST /backtests` (202)

Create and queue a Module-7 backtest. Body is the full `BacktestConfig`: `wallets`
(required, source addresses), `start`/`end` (optional window), `initial_capital`,
`sizing_mode` (`fixed` USDC per copy | `percentage` of equity | `proportional` to the
source notional) + `sizing_value`, optional per-wallet `weights`, `delay_seconds`,
`slippage_pct` (fraction, e.g. `0.005`), `fee_pct`, `max_open_positions`,
`max_position_size`. Decimals travel as strings. Returns the row with status `queued`;
Celery task `backtests.run` executes it (statuses: `queued` → `running` →
`completed`/`failed`; on failure `metrics.error` carries the reason).

Execution model: copies replay against a left-continuous step function of the source
fills' own prices — a copy delayed `delay_seconds` executes at the price observed at
that instant, slippage moves it adversely, fees charge on notional; source SELLs are
mirrored proportionally. Deterministic: same config + same fills = same result.

### `GET /backtests`

Paged list (`limit` ≤100, `offset`), newest first, without the curve/log payloads.

### `GET /backtests/{id}`

Full result: `params`, `status`, `metrics` (final_equity, total_return, cagr,
max_drawdown, sharpe, sortino, win_rate, profit_factor, turnover, fees_paid,
trades_executed/skipped), `equity_curve` (`[{t, equity}]`), and `trade_log` (per source
fill: source vs exec price, size, notional, fee, realized PnL, or the skip reason).

## Planned route groups

Routers exist as empty placeholders and gain endpoints in later phases (see `docs/plan.md`):

- `/risk`, `/copy`, `/portfolio` — Phase 5
- `/strategies` — Phase 6
- `/ws` — live updates websocket, Phase 5
