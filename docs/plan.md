# Polymarket Intelligence & Copy-Trading Platform — Implementation Plan

## Context

Dom is building the platform from the pasted spec: discover Polymarket wallets, score them
quantitatively, backtest portfolio strategies, and execute copy trades under risk controls.
Deployment target is a Beelink SER5 mini PC (Ubuntu Server 24.04, RAM upgraded to 32 GB) running
24/7, but the backend runs first on a **Windows test box** (Docker Desktop) before moving to the
Beelink — same Docker stack, host swap only. The UI ships as a **thin-client macOS `.dmg`** with a
configurable backend URL. Execution supports **paper (default) + gated live**. **No location
spoofing** — networking is legitimate per-service segmentation only.

Directory `/Users/dominictedrow/projects/polymarket_copy` is empty — greenfield.

> **Endpoint-accuracy note for the implementer:** Polymarket's public API surfaces
> (Gamma, Data-API, CLOB, WebSocket, leaderboard) and on-chain contract addresses must be
> **verified against `https://docs.polymarket.com` at implementation time** before wiring. This plan
> gives the known integration points and the exact place each is used; treat specific paths as
> "confirm then wire," and centralize them all in `backend/config.py` so a changed endpoint is a
> one-line fix. Do not hardcode endpoints in modules.

---

## Tech stack (pin at scaffold time)

- **Backend**: Python 3.12, FastAPI, Uvicorn, SQLAlchemy 2.x (async), Alembic, Pydantic v2,
  `pydantic-settings`. Async HTTP via `httpx`. Blockchain via `web3.py` 6.x. Polymarket trading via
  `py-clob-client`. WebSockets via `websockets`.
- **Workers/scheduling**: Celery 5 + Redis broker (beat for periodic jobs). (Dramatiq acceptable
  alternative; plan assumes Celery.)
- **DB**: PostgreSQL 16, Redis 7. Migrations via Alembic. Numeric money/qty columns use `NUMERIC`,
  never float.
- **Frontend**: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, TanStack Query,
  Recharts (swap to ECharts only if a chart needs it), `zustand` for light client state.
- **Desktop**: Tauri 2 (Rust shell, tiny binary, native `.dmg`). Electron is the fallback if a
  Tauri blocker appears; the frontend code is identical either way.
- **Tooling**: `ruff` + `black` + `mypy` (backend), `eslint` + `prettier` (frontend), `pytest` +
  `pytest-asyncio`, `pre-commit`, GitHub Actions CI.

---

## Repository layout

```
copytrader/
├── backend/
│   ├── pyproject.toml               # deps + tool config (ruff/black/mypy/pytest)
│   ├── alembic.ini
│   ├── app/
│   │   ├── main.py                  # FastAPI app factory, router include, CORS, lifespan
│   │   ├── config.py                # Settings (pydantic-settings): all endpoints, keys, limits
│   │   ├── deps.py                  # DB session, auth, common dependencies
│   │   ├── logging.py               # structured logging setup
│   │   ├── database/
│   │   │   ├── base.py              # async engine + sessionmaker
│   │   │   ├── models/              # SQLAlchemy models (one file per aggregate)
│   │   │   │   ├── market.py  wallet.py  trade.py  position.py
│   │   │   │   ├── wallet_metrics.py  wallet_score.py  ranking.py
│   │   │   │   ├── strategy.py  backtest.py  portfolio.py
│   │   │   │   ├── copy_order.py  risk_event.py  indexer_state.py
│   │   │   └── migrations/          # Alembic versions
│   │   ├── schemas/                 # Pydantic request/response models (mirror models/)
│   │   ├── api/
│   │   │   ├── router.py            # top-level APIRouter
│   │   │   └── routes/             # wallets.py discovery.py rankings.py backtests.py
│   │   │                            # strategies.py portfolio.py risk.py copy.py system.py ws.py
│   │   ├── blockchain/              # Module 1
│   │   │   ├── client.py            # web3 + RPC pool, retry, block cursor
│   │   │   ├── decoders.py          # decode CTF Exchange / NegRisk fill events
│   │   │   ├── indexer.py           # historical backfill (block ranges, checkpoint)
│   │   │   ├── listener.py          # live WS/poll subscription
│   │   │   └── normalize.py         # -> unified Trade/Position rows, dedup
│   │   ├── ingest/                  # REST pulls (Gamma markets, Data-API positions/activity)
│   │   │   ├── gamma.py  data_api.py  clob_api.py  leaderboard.py
│   │   ├── wallet_discovery/        # Module 2  (discovery.py + sources/)
│   │   ├── wallet_analytics/        # Module 3  (metrics.py -> WalletMetrics)
│   │   ├── adaptive_intelligence/   # Module 4  (scoring.py composite score)
│   │   ├── ranking/                 # Module 5  (rankings.py)
│   │   ├── strategy_optimizer/      # Module 6  (optimizer.py + candidates.py)
│   │   ├── backtesting/             # Module 7  (engine.py simulator + models.py + metrics.py)
│   │   ├── risk/                    # Module 8  (engine.py gate + rules/ + killswitch.py)
│   │   ├── copy_trading/            # Module 9  (engine.py + executors/ paper.py live.py base.py)
│   │   ├── portfolio/               # Module 10 (service.py + valuation.py)
│   │   ├── notifications/           # webhooks/email/telegram stubs
│   │   └── workers/                 # Celery app + beat schedule + tasks/ per module
│   └── tests/                       # mirrors app/; fixtures/ holds golden data
├── frontend/
│   ├── package.json  next.config.js  tailwind.config.ts  tsconfig.json
│   ├── src/
│   │   ├── app/                     # routes: /portfolio /wallets /discovery /risk
│   │   │                            #         /backtesting /optimization /settings
│   │   ├── components/              # charts/, tables/, layout/, ui/
│   │   ├── lib/api.ts               # typed fetch client; base URL from settings
│   │   ├── lib/ws.ts                # websocket client for live updates
│   │   └── stores/                  # zustand (settings incl. backend URL)
├── desktop/                         # Tauri 2 project wrapping the frontend build
│   ├── src-tauri/ (tauri.conf.json, Cargo.toml, icons)
│   └── scripts/build-dmg.sh
├── docker/
│   ├── backend.Dockerfile  frontend.Dockerfile  worker.Dockerfile
│   ├── docker-compose.yml           # core stack: postgres, redis, api, worker, beat, frontend
│   ├── docker-compose.dev.yml       # dev overrides (hot reload, bind mounts)
│   └── .env.example
├── scripts/                         # seed_sample_data.py, backfill.py, ops helpers
├── docs/                            # architecture.md, api.md, runbook.md, deploy.md, plan.md
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
└── README.md
```

---

## Data model (core tables)

All timestamps UTC. Money/quantity = `NUMERIC`. Addresses stored lowercase `VARCHAR(42)`.

- **markets**: `id (condition_id)`, `question`, `slug`, `category`, `outcomes (jsonb)`,
  `token_ids (jsonb)`, `active`, `closed`, `end_date`, `liquidity`, `volume`, `created_at`.
- **wallets**: `address (pk)`, `first_seen`, `last_active`, `discovery_source`, `discovered_at`,
  `total_volume`, `total_trades`, `is_tracked (bool)`, `copy_enabled (bool)`, `disabled_reason`.
- **trades**: `id`, `tx_hash`, `log_index`, `wallet_address (fk)`, `market_id (fk)`, `token_id`,
  `side (BUY/SELL)`, `price`, `size`, `usdc_amount`, `block_number`, `timestamp`,
  `trade_type (open/increase/decrease/close)`. Unique `(tx_hash, log_index)`.
- **positions**: `id`, `wallet_address`, `market_id`, `token_id`, `net_size`, `avg_entry_price`,
  `realized_pnl`, `unrealized_pnl`, `is_open`, `opened_at`, `closed_at`.
- **wallet_metrics**: `wallet_address (pk)`, `computed_at`, plus every metric in Module 3
  (roi_lifetime, realized_pnl, unrealized_pnl, win_rate, avg_trade_return, profit_factor, cagr,
  roi_30d, roi_90d, monthly_consistency, weekly_consistency, max_drawdown, losing_streak_max,
  avg_hold_hours, trade_frequency, avg_position_size, max_concurrent_positions, category_mix (jsonb),
  early_entry_score, exit_timing_score, avg_prob_improvement, volatility, concentration,
  sharpe, sortino, sample_size).
- **wallet_scores**: `wallet_address`, `computed_at`, `composite`, `historical_component`,
  `recent_component`, `copy_quality_component`, `risk_component`, `explanation (jsonb)`.
- **rankings**: `id`, `computed_at`, `category` (overall/politics/crypto/sports/economics/hft/
  long_term/early_info/low_risk), `rank`, `wallet_address`, `score`.
- **strategies**: `id`, `name`, `params (jsonb)` (selection rule, sizing model, weighting, limits),
  `is_active`, `created_at`, `adopted_at`, `rationale (jsonb)`.
- **backtests**: `id`, `strategy_id`, `params (jsonb)`, `status`, `started_at`, `finished_at`,
  `metrics (jsonb)` (cagr/sharpe/sortino/max_dd/profit_factor/win_rate/turnover),
  `equity_curve (jsonb)`, `trade_log (jsonb or child table)`.
- **portfolio_snapshots** / **portfolio_positions** / **copy_orders**: live/paper state — cash,
  open/closed positions, per-order source wallet, sizing, requested vs filled, execution quality
  (delay_ms, slippage), status, error.
- **risk_events**: `id`, `timestamp`, `type`, `severity`, `context (jsonb)`, `action_taken`.
- **indexer_state**: `id`, `stream` (historical/live), `last_block`, `updated_at` — crash recovery.

Alembic migration per phase; never edit a shipped migration.

---

## External integration points (Module 1 + ingest)

Polymarket runs on **Polygon (chain id 137)**, USDC collateral, Gnosis **Conditional Tokens
Framework** + **CTF Exchange** / **NegRisk CTF Exchange**. Integration surfaces:

- **Gamma API** (`gamma-api.polymarket.com`) — markets/events metadata → `markets` table.
- **Data API** (`data-api.polymarket.com`) — a wallet's positions/activity/trades → seed history and
  cross-check on-chain decode.
- **Leaderboard** — discovery source for top volume/PnL wallets (confirm exact host/path).
- **CLOB REST + `py-clob-client`** (`clob.polymarket.com`) — order book (paper fills) + live order
  submission.
- **CLOB WebSocket** (`wss://ws-subscriptions-clob.polymarket.com/...`) — live market/user updates.
- **On-chain** — RPC provider (Alchemy/Infura/QuickNode; configurable) subscribe to `OrderFilled`
  (or equivalent) events on the CTF Exchange contracts; decode in `decoders.py`; this is the source
  of truth for `trades`. Reconcile with Data-API for enrichment.

Two-source strategy: REST for fast backfill + enrichment, on-chain for authoritative, dedup-safe
trade capture. `indexer.py` walks block ranges with a checkpoint; `listener.py` handles the live
tail; both feed `normalize.py`.

---

## Module designs (key interfaces)

Design each module behind a small service interface so phases compose cleanly and are unit-testable.

- **Module 3 — analytics**: `compute_wallet_metrics(session, address) -> WalletMetrics`. Pure
  functions over that wallet's `trades`/`positions`; deterministic; unit-tested against fixtures.
- **Module 4 — scoring**: `score_wallet(metrics) -> WalletScore` implementing
  `0.40*historical + 0.30*recent + 0.20*copy_quality + 0.10*risk`, each sub-score normalized to
  [0,1] against the tracked cohort (percentile/z-score). `explanation` records each input's
  contribution (Transparency principle). Copy-quality uses real `copy_orders` execution stats when
  available, else a neutral prior.
- **Module 5 — ranking**: `rebuild_rankings(session)` writes a fresh `rankings` generation per
  category; API serves latest generation.
- **Module 7 — backtester**: `run_backtest(config: BacktestConfig) -> BacktestResult`. Deterministic
  event-loop replay of historical fills for selected wallets: apply execution delay, slippage, fees,
  sizing, allocation, and risk constraints; produce equity curve + metrics + trade log. This is the
  shared core reused by Module 6. Golden-file tested.
- **Module 6 — optimizer** (Celery beat, weekly/configurable):
  `optimize()` → refresh data, rescore, backtest candidate strategies (Top-5/10/20, equal/score/
  vol weighting, fixed/percentage/conviction sizing, category-diversified), compare to active
  strategy, adopt only if it beats thresholds AND passes risk constraints; write `strategies.rationale`.
- **Module 8 — risk gate**: `RiskEngine.check(intent: TradeIntent) -> RiskDecision(allow, reason,
  adjusted_size)`. Composed of ordered rule objects (portfolio limits, position limits, market
  filters, timing, wallet controls). `killswitch.py` provides a global halt (DB flag + Redis) checked
  first. Every denial/adjustment writes a `risk_event` (Transparency).
- **Module 9 — copy engine**: detects new source-wallet trades (from listener/normalize) → builds
  `TradeIntent` → `RiskEngine.check` → `Executor.execute(intent)`. `Executor` is an ABC:
  - `PaperExecutor` (default): simulate fill vs live CLOB order book, record delay/slippage.
  - `LiveExecutor` (gated): submit via `py-clob-client`; handle partial fills, exit mirroring,
    retries, failure recovery. **Disabled unless** `EXECUTION_MODE=live` AND `LIVE_TRADING_ARMED=true`
    AND kill switch off. Sizing modes: fixed / percentage / proportional / conviction / weighted.
- **Module 10 — portfolio**: maintain cash/positions from `copy_orders`; `valuation.py` marks open
  positions to current prices for unrealized PnL and writes periodic `portfolio_snapshots`.
- **Module 11 — dashboard**: one route per spec dashboard (portfolio, wallets, discovery, risk,
  backtesting, optimization) + `/settings` (backend URL, execution mode display, kill switch toggle).

---

## Configuration & secrets

Single `Settings` in `config.py` (env-driven; `.env` for dev, real env vars in prod). Keys:
`DATABASE_URL`, `REDIS_URL`, `POLYGON_RPC_URLS` (comma list, pooled), `GAMMA_API_URL`,
`DATA_API_URL`, `CLOB_API_URL`, `CLOB_WS_URL`, `LEADERBOARD_URL`, CTF/exchange contract addresses,
`EXECUTION_MODE` (paper|live, default paper), `LIVE_TRADING_ARMED` (default false),
`POLYMARKET_API_KEY/SECRET/PASSPHRASE` + wallet private key (live only; never committed; documented
in `deploy.md`), all risk limits (max exposure, daily/weekly/monthly loss, max positions, per-market/
per-wallet caps, min liquidity, max slippage, delay caps). Frontend stores only the backend base URL
(TanStack Query + zustand) so the same `.dmg` points at Windows box or Beelink.

---

## Phased delivery (each phase = shippable + tested)

**Phase 0 — Scaffold & infra.**
Files: full repo skeleton above, `pyproject.toml`, `config.py`, `database/base.py`, empty models +
first Alembic migration, `main.py` with `/health`, Celery app, all Dockerfiles, `docker-compose.yml`
+ `.env.example`, Next.js app with layout + `/settings`, `lib/api.ts`, CI, pre-commit, `docs/`.
Accept: `docker compose up` starts postgres/redis/api/worker/beat/frontend; `/health` returns 200;
frontend loads and can set backend URL; `pytest` and lint pass green (even if near-empty).

**Phase 1 — Data layer (Module 1 + ingest).**
`blockchain/` (client, decoders, indexer, listener, normalize), `ingest/` (gamma, data_api, clob_api),
`indexer_state`, dedup, checkpoint/recovery. Celery tasks for backfill + live tail.
Accept: backfilling a known wallet produces correct `trades`/`positions` matching Data-API within
tolerance; killing/restarting the indexer resumes from checkpoint with no dupes (unique constraint
test); markets sync populates `markets`.

**Phase 2 — Analytics + discovery (Modules 2–3).**
`wallet_discovery/` sources + `wallet_analytics/metrics.py`. API `/wallets`, `/wallets/{addr}`,
`/discovery`. Frontend Wallet Explorer + Discovery pages.
Accept: metrics computed deterministically for fixture wallets (golden test); discovery job expands
candidate pool with timestamps; dashboards render real profiles.

**Phase 3 — Scoring + ranking (Modules 4–5).**
`adaptive_intelligence/scoring.py`, `ranking/rankings.py`, `/rankings` API, rankings UI with score
breakdown/explanation.
Accept: composite score reproduces the 40/30/20/10 weighting on fixtures; rankings rebuild per
category; explanation JSON surfaces each contribution.

**Phase 4 — Backtesting (Module 7).**
`backtesting/` engine + metrics, `/backtests` API (create/status/result), Backtesting dashboard with
equity curves + strategy compare + export.
Accept: golden-file test — fixed fixture + fixed config → byte-stable metrics/equity curve;
delay/slippage/fee scenarios (5/15/30/60s) change results monotonically as expected.

**Phase 5 — Risk + paper copy-trading + portfolio (Modules 8–10).**
`risk/` gate + rules + killswitch, `copy_trading/` engine + `PaperExecutor`, `portfolio/` service +
valuation. APIs `/risk`, `/copy`, `/portfolio`, kill-switch endpoint. Portfolio + Risk dashboards;
live updates over websocket.
Accept: a simulated source trade flows detect→risk→paper-execute→portfolio, visible in UI;
each risk rule has a unit test proving allow/deny/adjust; kill switch halts all execution.

**Phase 6 — Strategy optimizer (Module 6).**
`strategy_optimizer/` + beat schedule, `/strategies` API, Optimization dashboard (current vs
recommended, history, rationale).
Accept: optimizer runs candidates through the Phase-4 engine, adopts only when thresholds+risk met,
and records a human-readable rationale; no adoption when improvement below threshold (tested).

**Phase 7 — Desktop `.dmg`.**
Tauri 2 project wrapping the frontend build; configurable backend URL in `/settings`; `build-dmg.sh`;
signing/notarization steps in `deploy.md`.
Accept: `.dmg` builds, installs, launches, points at the Windows-box backend, dashboards load live
data; repointing URL to the Beelink works with no rebuild.

**Phase 8 — Gated live execution (Module 9 live path).**
`LiveExecutor` via `py-clob-client`: partial fills, exit mirroring, retries, recovery. Off unless
`EXECUTION_MODE=live` + `LIVE_TRADING_ARMED=true` + kill switch clear; passes full risk gate.
Accept: on a throwaway/minimal setup, an armed live copy places and mirrors an exit correctly and
records execution quality; disarming or kill switch blocks all live orders (tested).

---

## Testing strategy

- `pytest` + `pytest-asyncio`; ephemeral Postgres (compose service or testcontainers) per run.
- Deterministic fixtures under `backend/tests/fixtures/` (sample markets/wallets/trades) drive
  analytics, scoring, and backtest golden tests.
- Every risk rule and every executor has explicit allow/deny/simulate tests.
- Frontend: component tests for chart/table wrappers; Playwright smoke of each dashboard route.
- CI runs lint + type-check + tests on push.

## Deployment

- **Dev / Windows test box**: Docker Desktop, `docker compose -f docker-compose.yml -f
  docker-compose.dev.yml up`. `.env` from `.env.example`. Desktop `.dmg` points at
  `http://<windows-box-ip>:8000`.
- **Beelink (Ubuntu 24.04)**: same compose, prod `.env`, restart policies for 24/7, Postgres volume
  backups. Repoint the desktop app's backend URL — no rebuild.
- **Networking**: single `core` Docker network for db/redis/api/worker/frontend (direct, low
  latency). Any outbound gateway is a separate, optional network domain with health-check +
  fallback-to-direct so no single path is a shared point of failure. **No VPN geo-spoofing.**
- Secrets (RPC keys, CLOB API creds, wallet key for live) via env only; documented in `deploy.md`;
  never committed. `.env.example` holds placeholders.

## Verification (end-to-end)

1. `docker compose up` → `/health` 200; `scripts/seed_sample_data.py` loads a sample cohort so
   dashboards render without a full chain backfill.
2. Backfill one real wallet → metrics/score/ranking populate and match Data-API within tolerance.
3. Run a backtest from the UI → equity curve + metrics render; delay/slippage scenarios behave.
4. Fire a simulated source trade → detect→risk→paper-fill→portfolio, visible live in UI; toggle kill
   switch and confirm halt.
5. Build the `.dmg`, install on macOS, point at the Windows-box backend, confirm live dashboards;
   repoint at the Beelink.

## Decisions (confirmed)

1. **DMG**: thin-client desktop app, configurable backend URL; test on a Windows box, then Beelink.
2. **Execution**: paper (default) + gated live built behind config + kill switch + risk layer.
3. **Start**: full Phase-0 scaffold, then modules in order.
4. **Geo/VPN**: no location spoofing; networking is legitimate segmentation only.
