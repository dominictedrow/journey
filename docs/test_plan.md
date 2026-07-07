# Test Plan

Concrete, runnable test plan derived from the tiers in [`testing.md`](testing.md). Each tier
below turns that document's classification into actual commands, pass/fail criteria, and owners
(who/what environment runs it) — not a restatement of the tiers themselves.

## Tier 1 — Fully testable (run on any Mac, no topology tricks needed)

Everything here is pure Python business logic (`Decimal`-based, DB-driven, OS/arch-independent).
Run it in tight-loop iteration on a Mac; it must also pass in CI (`ubuntu-latest`, see
`.github/workflows/ci.yml`) before merge.

| # | Area | Command | Pass criteria |
|---|------|---------|----------------|
| 1.1 | Full backend unit/integration suite | `docker build -f docker/backend.Dockerfile -t copytrader-backend . && docker run --rm copytrader-backend pytest` | All 23 files in `backend/tests/` pass; zero skips unless explicitly marked |
| 1.2 | Lint/format/type gates (same as CI) | `docker run --rm copytrader-backend ruff check .` / `black --check .` / `mypy app` | Clean exit 0 on all three — these gate every PR per `ci.yml` |
| 1.3 | Kill switch blocks live orders | `pytest backend/tests/test_live_executor.py::test_kill_switch_blocks_live_order -v` | Order is never placed when the global halt flag is set, even with `EXECUTION_MODE=live` armed |
| 1.4 | Paper mode never selects live executor | `pytest backend/tests/test_live_executor.py::test_paper_mode_never_selects_live_executor -v` | `select_executor()` returns the paper executor unless both `EXECUTION_MODE=live` **and** `LIVE_TRADING_ARMED=true` (`backend/app/copy_trading/engine.py:37`, `executors/live.py:6`) |
| 1.5 | Live-order reconciliation (crash recovery logic) | `pytest backend/tests/test_live_executor.py -k reconcile -v` (covers `reconcile_live_orders`, `backend/app/copy_trading/recovery.py:33`) | Orphaned CLOB orders are recovered exactly once; a second reconcile pass is a no-op (idempotent) |
| 1.6 | Risk gate rules (all 9 rule types) | `pytest backend/tests/test_risk_rules.py -v` | Every allow/deny/adjust/bypass branch in the 9 tests passes — wallet control, market filter, timing, loss limit, max open positions, position size, per-market/per-wallet/portfolio exposure |
| 1.7 | Backtest determinism & math | `pytest backend/tests/test_backtest_engine.py -v` | `test_deterministic` and `test_golden_run` produce bit-identical output across runs; delay/slippage/fee scenario tests are monotonically worse as configured |
| 1.8 | Wallet scoring determinism | `pytest backend/tests/test_scoring.py -v` | `test_deterministic` and `test_golden_composite_weighting` are bit-for-bit reproducible |
| 1.9 | Misconfigured live box falls back to paper | `pytest backend/tests/test_config.py -v` | Setting only `EXECUTION_MODE=live` (no `LIVE_TRADING_ARMED`) does not arm live trading |
| 1.10 | Full Docker stack against real Postgres | `docker compose -f docker/docker-compose.yml up -d`, then hit `GET /health` and run one manual copy-flow trade injection via `POST /copy/simulate` (see `docs/api.md`) | API boots, migrations apply cleanly (`alembic upgrade head` in the `api` service command), JSONB columns behave — this exercises the Postgres gap the SQLite unit suite (`conftest.py`) doesn't cover |

**Exit bar for Tier 1**: 1.1–1.9 must be green before every merge (CI already enforces 1.1/1.2).
Run 1.10 manually before any deploy to the Beelink, since CI does not stand up the real compose
stack.

## Tier 2 — Testable on Mac, only if the real topology is deliberately reproduced

These require extra setup beyond `pytest` — do not skip the setup and assume Tier 1 coverage
implies these pass too.

| # | Area | Setup | Command / procedure | Pass criteria |
|---|------|-------|----------------------|----------------|
| 2.1 | amd64 parity (crypto wheels: `web3`/`py-clob-client`, secp256k1/keccak) | None extra — this *is* what CI already does | Confirm `.github/workflows/ci.yml`'s `backend` job is green on `ubuntu-latest` (amd64) | Treat CI green as the parity gate; a local arm64 Mac Docker run passing is **not** sufficient evidence |
| 2.2 | amd64 parity, verified locally on Apple Silicon | `docker buildx build --platform linux/amd64 -f docker/backend.Dockerfile -t copytrader-backend-amd64 .` (QEMU-emulated) | `docker run --rm copytrader-backend-amd64 pytest` | Passes under emulation — slower but exact-arch; run before trusting a Beelink-bound image built on a Mac |
| 2.3 | Real LAN hop (Windows box → Beelink path) | Run `docker compose -f docker/docker-compose.yml up -d` on the Mac; find its LAN IP (`ipconfig getifaddr en0`) | On a **second device** (phone/laptop/the desktop `.dmg` itself), open the frontend and set Settings → Backend URL to `http://<mac-lan-ip>:3000`'s backend (`:8000`), not `localhost` | `useLiveState()` in `frontend/src/lib/ws.ts` connects over `ws://<lan-ip>:8000/ws/live?mode=paper`, `connected` flips true, live frames stream in |
| 2.4 | WebSocket reconnect after backend restart | Same LAN setup as 2.3, connected and streaming | `docker compose -f docker/docker-compose.yml restart api` while the second device is connected | UI shows `connected=false` momentarily, then auto-reconnects within ~2s (the `retry.current = setTimeout(connect, 2000)` backoff in `ws.ts`) with no manual page reload |
| 2.5 | Container restart recovery time | Full compose stack up | `docker kill copytrader-api-1` (or the actual container name from `docker ps`), then time until `/health` returns 200 again | Measure and log wall-clock seconds; `restart: unless-stopped` in `docker-compose.yml` should bring it back without manual intervention |
| 2.6 | Resource ceiling (32GB Beelink parity) | Add a Mac-only compose override with `deploy.resources.limits` (e.g. `memory: 4g` per worker/beat, scaled to Beelink's per-service budget) or `docker compose ... --compress` + `docker stats` | Run paper-mode copy trading for several hours (`docker stats` polling or a logged memory-sample script) | No unbounded memory growth in `worker`/`beat` processes over the observation window — flag any monotonic climb as a leak candidate |

**Exit bar for Tier 2**: run 2.3–2.5 once per significant change to `ws.ts`, the copy/live-executor
pipeline, or `docker-compose.yml`; run 2.6 as a periodic soak (e.g. weekly) rather than per-PR.

## Tier 3 — Not realistically testable on a Mac

No Mac procedure substitutes for these. Track them as pre-production checks against real hardware
or a persistent cloud Linux box, not as CI/local gates.

| # | Area | What "testing" means here | Where to run it |
|---|------|---------------------------|-------------------|
| 3.1 | 24/7 uptime / thermal / power-loss behavior of the mini PC | Extended soak (days, not hours) on the actual Beelink SER5, monitoring for thermal throttling or spontaneous reboots | Beelink SER5 only |
| 3.2 | Real production network path (router/NAT/firewall between Windows monitor box and Beelink) | Manual verification once both machines are on the production LAN — confirm the Windows `.exe`/NSIS client (see [`windows.md`](windows.md)) can actually reach the Beelink's backend URL through the real network, not just a same-subnet Mac LAN test | Production LAN, both real machines |
| 3.3 | True unattended-restart recovery (`copy.reconcile_live` beat task under a real power-cut) | Pull power on the Beelink mid-order (not `docker kill`), then verify `reconcile_live_orders` (`backend/app/copy_trading/recovery.py:33` — same function unit-tested in 1.5) actually sweeps and recovers orphaned CLOB orders after an unclean shutdown | Beelink SER5, deliberate power-cut test, **before** first `LIVE_TRADING_ARMED=true` flip |

**Exit bar for Tier 3**: 3.3 is a hard gate — do not set `LIVE_TRADING_ARMED=true` in production
until it has been run at least once against real hardware. 3.1/3.2 are ongoing operational
monitoring, not one-time pass/fail gates.

## Summary: what to run when

- **Every commit (CI)**: 1.1, 1.2 automatically; 2.1 is CI itself.
- **Every PR touching risk/copy/backtest/scoring logic**: 1.3–1.9 (already covered by 1.1, called
  out individually so a failure is traceable to the specific guarantee that broke).
- **Before any Beelink deploy**: 1.10, 2.2.
- **After any change to `ws.ts`, live executor, or compose topology**: 2.3–2.5.
- **Weekly / periodic**: 2.6 soak.
- **Before first live-armed production run, and after any recovery-logic change**: 3.3 (real
  hardware power-cut test) — non-negotiable, no Mac substitute exists.
