# Deploy

One backend host, one packaged desktop client (macOS `.dmg`). The Docker stack runs only on the
Beelink. Windows access is a plain browser tab, not a packaged app — see "Windows access" below;
a Windows `.exe` build path exists and works (see [`docs/windows.md`](windows.md)) but was decided
against for now as unnecessary packaging effort for a personal monitoring client.

## 1. Beelink SER5 (Ubuntu Server 24.04, 32 GB RAM, 24/7) — the sole backend host

Same compose files with a production `.env`. Services already declare `restart: unless-stopped`;
Postgres data lives in the `pgdata` volume — schedule `pg_dump` backups (cron) off-device.

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Open firewall for ports 8000 (API) and 3000 (web) on the LAN — this is the only host that needs
inbound rules, since it's the only one running the backend. Desktop clients / browsers point at
`http://<beelink-ip>:8000`.

## 2. Windows access — browser tab, no install

No Windows box role, no build, no installer. Open Edge/Chrome on any Windows machine on the LAN
and go to `http://<beelink-ip>:3000` (the frontend) — same static frontend, same backend calls,
same network behavior as the packaged macOS `.dmg`. Nothing to install or build.

This has the identical Local Network Access consideration a packaged app would have (it's the same
Chromium engine either way) — see [`docs/windows.md`](windows.md) Step 3 if the connection ever
seems to hang/reconnect endlessly, but no action is needed today since LNA ships disabled by
default.

## Secrets

Env vars only; never committed. `docker/.env.example` documents every key with placeholders.

- `POLYGON_RPC_URLS` — comma-separated RPC endpoints (Alchemy/Infura/QuickNode)
- `POLYMARKET_API_KEY` / `POLYMARKET_API_SECRET` / `POLYMARKET_API_PASSPHRASE` — CLOB creds (live only)
- `WALLET_PRIVATE_KEY` — signing key for live execution only; keep it off dev machines
- `CLOB_SIGNATURE_TYPE` — `0` = EOA (the signing key holds funds, default), `1` = email/magic
  proxy, `2` = browser (Safe) proxy; non-EOA schemes also need `CLOB_FUNDER_ADDRESS`
- `EXECUTION_MODE=paper` is the default; live requires `EXECUTION_MODE=live` **and**
  `LIVE_TRADING_ARMED=true` **and** the kill switch clear

## Live execution (Phase 8)

Live trading is off by default and opt-in only. The `LiveExecutor` (`py-clob-client`) is
selected **only** when `EXECUTION_MODE=live` **and** `LIVE_TRADING_ARMED=true`; any other
configuration — including an unarmed live box — trades on paper, so a misconfiguration fails
safe. Three independent gates must all clear before a share trades:

1. **Arming** (`EXECUTION_MODE=live` + `LIVE_TRADING_ARMED=true`) — a deliberate, deploy-time
   env decision. The executor re-checks it on every order (defence in depth), so it refuses to
   place an order while disarmed even if wired in by mistake. **Disarm** = set
   `LIVE_TRADING_ARMED=false` and restart the worker.
2. **Kill switch** — `POST /risk/kill-switch` halts *all* execution (paper and live) durably
   across every process; enforced by the risk gate before the executor is ever reached.
3. **Risk gate** — the same size/exposure/slippage/delay rules paper trading passes.

Orders are marketable **Fill-And-Kill** limits priced to cross the book by
`LIVE_MARKETABLE_BUFFER_PCT` (default 1%): they fill immediately (fully or partially) and
cancel the rest rather than resting as a stale maker. Transient placement failures retry
(`LIVE_MAX_RETRIES`, `LIVE_RETRY_BACKOFF_SECONDS`). A `copy.reconcile_live` beat task
(every 120s, no-op unless armed live) sweeps the CLOB against our records: it refreshes
in-flight orders and flags any **orphan** — an order placed on the venue whose fill we never
recorded (a crash between placement and commit) — as a `risk_event` for manual review rather
than silently booking unattributable P&L. `GET /health` and `GET /risk` report `live_armed`.

## Networking

Single `core` Docker network for db/redis/api/worker/frontend (direct, low latency). Any outbound
gateway is a separate, optional network domain with health-check + fallback-to-direct. No VPN
geo-spoofing — legitimate segmentation only.

## Desktop `.dmg`

Thin-client macOS app (Tauri 2 in `desktop/`) that loads the statically-exported frontend into a
native webview and talks to a backend URL set in **Settings** — the same `.dmg` points at the
Windows box or the Beelink with no rebuild.

### Build prerequisites (macOS host)

- **Node 18+** and **Rust stable** (`curl https://sh.rustup.rs -sSf | sh`).
- **Xcode command-line tools** (`xcode-select --install`).
- The frontend static export is produced automatically (`BUILD_TARGET=desktop next build` → `out/`).

### Build

```bash
./desktop/scripts/build-dmg.sh
```

This installs deps, generates the app icons if missing (`desktop/scripts/generate_icons.py` —
swap in real artwork, or use `npm --prefix desktop run tauri -- icon <png>`), runs the frontend
static export via Tauri's `beforeBuildCommand`, and emits the disk image at:

```
desktop/src-tauri/target/release/bundle/dmg/Copytrader_<version>_aarch64.dmg
```

For iteration without packaging: `npm --prefix desktop run dev` (starts `next dev` and opens the
webview against `http://localhost:3000`).

### Pointing at a backend

On first launch, open **Settings** and set the backend URL (e.g. `http://<windows-box-ip>:8000`,
later the Beelink). It persists locally (zustand) — repointing needs no rebuild.

Plain-http LAN backends work because `desktop/src-tauri/Info.plist` ships an App Transport
Security exception (`NSAllowsArbitraryLoads` / `NSAllowsLocalNetworking`); macOS would otherwise
block non-HTTPS requests from the webview. If you later put the backend behind HTTPS, tighten this
to a scoped `NSExceptionDomains` entry.

### Signing & notarization (distribution)

An **unsigned** `.dmg` installs locally after right-click → Open (Gatekeeper warning). For
distribution to other Macs, sign + notarize. Tauri reads these from the environment during
`build-dmg.sh`:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"  # in Keychain
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"   # appleid.apple.com → App-Specific Passwords
export APPLE_TEAM_ID="TEAMID"
./desktop/scripts/build-dmg.sh
```

With `APPLE_SIGNING_IDENTITY` set, Tauri code-signs the `.app`; with the `APPLE_ID` trio set it
also submits to Apple's notary service and staples the ticket. Requires a paid Apple Developer
account and a **Developer ID Application** certificate installed in the login Keychain. Never
commit these values — they are runtime env only, like the backend secrets above.

## Desktop `.exe` (NSIS) — built but not the recommended path

A Windows NSIS installer build path exists and works (`desktop/scripts/build-windows.ps1` →
`desktop/src-tauri/target/release/bundle/nsis/Copytrader_<version>_x64-setup.exe`, unsigned, no
admin prompt on install). It was decided against in favor of the plain browser tab above — the
network behavior is identical either way, and packaging buys only a window icon/Start Menu entry,
not different backend access. Full build/signing details are kept in
[`docs/windows.md`](windows.md) if this is ever revisited.
