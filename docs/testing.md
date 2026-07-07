# Testing on the Mac vs. the Beelink

How to test this system on a Mac in totality without being unrealistic toward how it actually
runs on the Ubuntu mini PC (Beelink SER5 — see [`deploy.md`](deploy.md)). Split into three tiers
by how faithfully a Mac can stand in for the real deployment.

## Tier 1 — Fully testable, host-independent, quantifiable exactly

Risk gates, kill switch, executor selection, backtest math, portfolio accounting. Pure Python
business logic using `Decimal` throughout (39 files import it — no floats), driven by env vars and
Postgres/SQLite rows, with zero dependency on OS or CPU arch.

- **How**: `backend/tests/` (23 files) already covers this —
  `test_live_executor.py::test_kill_switch_blocks_live_order`,
  `test_paper_mode_never_selects_live_executor`, `test_risk_rules.py`, `test_backtest_engine.py`,
  `test_scoring.py`, etc. They run against an in-memory SQLite DB (`conftest.py`), so they're fast
  and fully portable to a Mac.
- **Quantifiable**: completely. `Decimal`-based math means backtest P&L, wallet scores, and risk
  calculations are bit-for-bit reproducible — assert exact values, not "close enough."
  Kill-switch/arming logic is boolean-gated (`EXECUTION_MODE`, `LIVE_TRADING_ARMED`), so pass/fail
  is deterministic — e.g. confirm a misconfigured live box always falls back to paper.
- **Caveat**: this suite runs against SQLite, not the Postgres the real stack uses
  (`docker/docker-compose.yml`'s `postgres:16`, JSONB columns). That's a prod-vs-test gap, not a
  Mac-vs-Beelink one — also run the full Docker stack locally to exercise real Postgres behavior
  (migrations, JSONB, concurrent writers), not just the SQLite-backed unit suite.

## Tier 2 — Testable on Mac, but only if you deliberately reproduce the real topology

Where "unrealistic toward the mini PC" bites if you're not careful:

- **Architecture (arm64 vs amd64)**: Docker on Mac builds images for the host's arch by default —
  `docker/docker-compose.yml` has no `platform:` pin. `web3`/`py-clob-client`
  (`backend/pyproject.toml`) pull compiled crypto wheels (secp256k1/keccak signing) — arch-specific
  binaries that a Mac-only arm64 test run wouldn't exercise. CI (`.github/workflows/ci.yml`)
  already runs `docker build`/`pytest` on `ubuntu-latest` (amd64) — the same arch as the Beelink.
  Treat CI as the amd64 parity gate; don't treat local Mac Docker runs as proof of Beelink
  behavior. For extra rigor locally: `docker buildx build --platform linux/amd64`
  (QEMU-emulated, slower but exact-arch).
- **The "two machines" monitoring split**: running the backend and opening the dashboard in the
  same terminal/browser on a Mac tests loopback, not the real LAN hop the Windows box → Beelink
  path takes. To test it honestly: run the Docker stack on the Mac, then hit it from a **second
  device** on the same LAN (phone, another laptop, or the Tauri `.dmg` itself pointed at the Mac's
  LAN IP instead of `localhost`) via the real `backendUrl` Settings field. This exercises the
  actual WebSocket reconnect logic in `frontend/src/lib/ws.ts` and the plain-HTTP LAN path that
  `Info.plist`'s ATS exception exists for.
- **Resource ceiling**: the Beelink is a fixed 32GB-RAM box running 24/7; a Mac isn't resource-capped
  the same way. Use `docker compose ... --cpus` / `--memory` limits (or a Mac-only compose
  override with `deploy.resources`) to catch memory-growth bugs in the long-running worker/beat
  processes before they hit production.
- **Quantifiable**: yes — LAN round-trip latency on the WebSocket, container restart time after
  `docker kill` (measured against `restart: unless-stopped`), memory growth over N hours of
  paper-mode operation. All numeric, all loggable.

## Tier 3 — Not realistically testable on a Mac at all

Needs the real hardware (or at minimum a cloud Linux box) before trusting a live-armed deploy:

- Actual 24/7 uptime/thermal/power-loss behavior of the mini PC itself.
- Real production network path (router, actual LAN, whatever NAT/firewall sits between the
  Windows box and the Beelink) — a Mac LAN test approximates this but isn't identical.
- True unattended-restart recovery (pulling power on the Beelink vs. `docker kill` on a Mac) — the
  reconciliation logic (`copy.reconcile_live` beat task, see `deploy.md`, sweeps the CLOB for
  orphaned orders) is designed exactly for this crash scenario, so it *is* testable in principle —
  but only by actually killing the box mid-order, which wants real or persistent infra, not a Mac
  dev session closed at night.

## Bottom line

Run the unit suite and a full local Docker stack on the Mac for tight-loop iteration (Tier 1,
fully trustworthy). Deliberately break loopback for the monitoring path to test the real
client/server split (Tier 2, trustworthy if done deliberately). Lean on CI's Linux/amd64 runs plus
an eventual real-hardware soak test before flipping `LIVE_TRADING_ARMED=true` for real (Tier 3, a
Mac can't give you this).
