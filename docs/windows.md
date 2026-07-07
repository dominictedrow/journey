# Windows Desktop Client

**Decision (2026-07-07): packaging skipped — Windows access is via a plain browser tab instead.**
Steps 1-3 below were completed (NSIS target added, `.exe` built via cross-compile, WebView2
network behavior researched), but Dom decided the by-hand install/verification on the physical
Windows box (Step 3's remaining item) and the `.exe` distribution path aren't worth pursuing for a
personal monitoring client — a browser tab pointed at `http://<beelink-ip>:3000` behaves
identically over the network (see the footnote at the bottom) with zero install/build/SmartScreen
friction. This doc is kept for reference (the `.exe` build path still works if ever wanted later)
but is no longer an active plan. See [`deploy.md`](deploy.md) for the current Windows guidance.

Plan to make the desktop monitoring shell runnable on Windows. The bot itself keeps running only
on the Ubuntu mini PC (Beelink) — Windows is a monitor/control client only, same as the macOS
`.dmg` described in [`deploy.md`](deploy.md).

Verified against `desktop/src-tauri/tauri.conf.json`, `desktop/src-tauri/Cargo.toml`,
`desktop/src-tauri/Info.plist`, `desktop/src-tauri/capabilities/default.json`,
`desktop/src-tauri/src/{main.rs,lib.rs}`, `desktop/package.json`, `desktop/README.md`,
`desktop/scripts/{build-dmg.sh,generate_icons.py}`, `frontend/src/lib/{api.ts,ws.ts}`,
`frontend/src/stores/settings.ts`, `frontend/next.config.js`, and `docs/deploy.md` — plus the
live Tauri v2 config schema (`schema.tauri.app/config/2`), the Tauri v2 Windows Installer docs
(`v2.tauri.app/distribute/windows-installer/`), the Tauri v1→v2 migration notes, and Microsoft's
WebView2 Local Network Access announcement (`MicrosoftEdge/WebView2Announcements#126`), checked
2026-07-07. Today `bundle.targets` is `["dmg"]` only (`tauri.conf.json:30`); `icon.ico` is already
listed in `bundle.icon` (`tauri.conf.json:39`, wired in, just unconsumed until a Windows target
exists); `Info.plist`'s ATS exception has no Windows analog (Windows uses a different, already-safe
default — see Step 3); and the "Windows test box" section of `deploy.md` currently describes a
full Docker Compose backend host — that framing needs updating once the box becomes a
monitor-only client.

**Headline change from the previous version of this plan:** Step 3's mixed-content risk is
resolved by a Tauri v2 default (confirmed against the live schema and migration docs), not by a
runtime flag or a reverse proxy. The real open risk for the Windows spike is a *different,
currently-inactive* Chromium/WebView2 feature — Local Network Access — explained below with exact
version numbers.

## Step 1 — Add the Windows bundle target

**Status: done (2026-07-07).** Files: `desktop/src-tauri/tauri.conf.json`, `desktop/package.json`

**Config diff** — in `tauri.conf.json`, change:

```diff
-    "targets": ["dmg"],
+    "targets": ["dmg", "nsis"],
```

(`bundle.targets`, currently `tauri.conf.json:30`.) NSIS is Tauri 2's standard Windows installer;
MSI (via WiX) is the alternative, but NSIS has better upgrade/uninstall ergonomics and — unlike
WiX, which only builds on Windows — NSIS bundles can also be produced by cross-compiling from
macOS/Linux (see the alternative build path in Step 2). No other `bundle` keys need to change to
activate Windows packaging:

- `bundle.icon` (`tauri.conf.json:34-40`) already lists `"icons/icon.ico"` — Tauri picks this up
  automatically for both the app icon and the NSIS installer icon once `nsis` is in `targets`.
  Nothing to add here.
- `bundle.windows` (the object that would hold `nsis`, `webviewInstallMode`, signing keys, etc.)
  does not exist yet in `tauri.conf.json` and does not need to for a first working build — Tauri's
  schema defaults apply: `webviewInstallMode` defaults to
  `{ "type": "downloadBootstrapper", "silent": true }` (fetches the WebView2 Runtime installer at
  install time if it's missing — fine for a LAN-connected monitor box; add
  `"bundle": {"windows": {"webviewInstallMode": {"type": "offlineInstaller"}}}` later only if the
  target machine has no internet access, at the cost of ~127MB added to the installer), and an
  absent `nsis` object defaults to `installMode: "currentUser"` (installs to `%LOCALAPPDATA%`,
  **no administrator/UAC prompt** — appropriate for a single-user monitor client; `"perMachine"`
  or `"both"` are the alternatives if this ever needs a shared-machine install).

**Correction found during the actual build (not caught by review — only surfaces when `tauri
build` really runs): `app.security.comment` is not a valid key at all.** The pre-existing
`tauri.conf.json:23-26` had:

```json
"security": {
  "comment": "The frontend fetches from a user-configured backend URL, so no CSP is injected; the macOS ATS exception in Info.plist is what actually permits plain-http LAN backends.",
  "csp": null
}
```

This plan originally proposed just rewording the comment to also cover Windows. That would have
been wrong: Tauri v2's live config schema sets `additionalProperties: false` on `app.security`, so
`comment` — reworded or not — makes **every** build (`dmg` and `nsis` alike) fail immediately with
`Additional properties are not allowed ('comment' was unexpected)`. This had apparently never been
caught because `tauri build` had never actually been run in this repo before. **Fix applied:
delete the `comment` key outright** rather than reword it — the explanation of why plain-http LAN
backends work on each platform lives here in Step 3 instead, so no information is lost:

```json
"security": {
  "csp": null
}
```

**`desktop/package.json` diff** — add a sibling script next to the existing `"build"`
(`desktop/package.json:9`):

```diff
   "scripts": {
     "tauri": "tauri",
     "dev": "tauri dev",
     "build": "tauri build -- --bundles dmg",
+    "build:windows": "tauri build -- --bundles nsis",
     "icons": "python3 scripts/generate_icons.py"
   },
```

This is the minimal, complete config diff needed before any Windows-specific build can run — no
Rust code changes required (`src-tauri/src/main.rs` already has the
`#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]` attribute at line 2, which
only takes effect when compiled for Windows and is a no-op elsewhere, so it needs no change either).

## Step 2 — Produce the actual `.exe`/NSIS build

**Status: done (2026-07-07) — via the cross-compile path, not the native-Windows-box path,
because no Windows box was reachable from this session.** `desktop/scripts/build-windows.ps1` and
the `desktop/README.md` section below were added as planned; the actual first build ran on macOS
with `cargo-xwin` (see "Alternative: cross-compile from macOS" below) and produced
`Copytrader_0.1.0_x64-setup.exe` (1.4MB, unsigned), delivered directly to Dom to carry over to the
Windows box by hand.

**Second bug found only by actually running the build:** `beforeDevCommand` /
`beforeBuildCommand` in `tauri.conf.json` (`build.beforeDevCommand`/`build.beforeBuildCommand`)
used `../../frontend`, copied from the assumption that these paths resolve relative to
`tauri.conf.json`'s directory (`src-tauri/`) the same way `frontendDist` does. They don't — Tauri
runs `beforeDevCommand`/`beforeBuildCommand` with cwd equal to wherever the `tauri` CLI process
itself was invoked (here, `desktop/`, since both `build-dmg.sh` and `build-windows.ps1` run
`npm --prefix "$Desktop" run tauri -- build ...`). `../../frontend` from `desktop/` resolves one
level too high (`.../projects/frontend` instead of `.../projects/polymarket_copy/frontend`),
so the very first real build failed with `ENOENT` on the frontend's `package.json`. **Fixed:**
both changed to `../frontend`. This means `build-dmg.sh`'s `.dmg` build had the same latent bug and
had apparently never been run for real either — worth a smoke test next time that path is touched.

Files: new `desktop/scripts/build-windows.ps1`, `desktop/README.md`

### Primary path: native build on the existing Windows box

Tauri does not cross-compile the *default* way `build-dmg.sh` builds (i.e., you can't run
`npm run build:windows` unmodified on macOS and get a working `.exe` the way `tauri build` just
works for `.dmg` on macOS) — but full cross-compilation *is* documented and possible (see
alternative path below). For a single monitoring client, building natively on the Windows box
already referenced in `deploy.md` is simpler and is what a real WebView2 smoke test requires
anyway (see Step 3), so it remains the primary path.

**New prerequisites on the Windows box** (it already has git + Node from its former backend role
per `deploy.md`):

1. **Rust (MSVC toolchain)** — install via `rustup-init.exe` from https://rustup.rs; accept the
   default host triple `x86_64-pc-windows-msvc` (this is what NSIS bundling expects; do not
   install the `-gnu` toolchain).
2. **Microsoft C++ Build Tools** — install Visual Studio Build Tools and select the **"Desktop
   development with C++"** workload. This provides the MSVC linker (`link.exe`) `tauri-build`
   needs; there is no separate "just the linker" package.
3. **WebView2 Runtime** — pre-installed on Windows 11 and current Windows 10 (Evergreen,
   auto-updating, same channel as Edge). Confirm presence via Settings → Apps → search
   "WebView2 Runtime", or check inside the built app later via the webview's `edge://version`
   equivalent. If missing, Tauri's default `webviewInstallMode` (downloadBootstrapper) installs it
   automatically the first time the built app runs — nothing needs to be pre-installed just to
   *build*.
4. Node 18+ and git — already present per `deploy.md`'s existing "Windows test box" section.

**`desktop/scripts/build-windows.ps1`** should be the PowerShell equivalent of
`desktop/scripts/build-dmg.sh` (`build-dmg.sh:1-40`), i.e.:

```powershell
# Mirror build-dmg.sh: install deps, generate icons if missing, build, report output path.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Desktop = Split-Path -Parent $Here
$Root = Split-Path -Parent $Desktop

npm --prefix "$Root/frontend" install
npm --prefix "$Desktop" install

if (-not (Test-Path "$Desktop/src-tauri/icons/icon.ico")) {
    python "$Here/generate_icons.py"
}

npm --prefix "$Desktop" run tauri -- build --bundles nsis

$NsisDir = "$Desktop/src-tauri/target/release/bundle/nsis"
Write-Host "`nDone. Installer(s):"
Get-ChildItem "$NsisDir/*.exe" -ErrorAction SilentlyContinue
```

Output lands at `desktop/src-tauri/target/release/bundle/nsis/Copytrader_<version>_x64-setup.exe`
(exact filename format per Tauri's NSIS bundler; version comes from `tauri.conf.json:4`'s
top-level `"version": "0.1.0"`).

### Alternative: cross-compile from macOS (documented, not required for this plan)

Correcting the earlier assumption that Tauri "doesn't cross-compile from macOS" at all: Tauri's
own Windows Installer docs describe cross-compiling NSIS bundles from macOS/Linux via
[`cargo-xwin`](https://github.com/rust-cross/cargo-xwin), because — unlike WiX/MSI, which only
builds on Windows — the NSIS bundler itself is a portable tool. Setup:

```bash
brew install nsis llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
```

Build:

```bash
npm --prefix desktop run tauri -- build --runner cargo-xwin --target x86_64-pc-windows-msvc --bundles nsis
```

Output: `desktop/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/`. `cargo-xwin`
downloads the Windows SDK/CRT headers on first use (set `XWIN_CACHE_DIR` to share the cache across
projects/rebuilds — otherwise it re-downloads per project).

**Confirmed working (2026-07-07):** ran end-to-end on this Mac — toolchain install (~2 min),
`cargo-xwin` install (~36s), then the build itself (~90s compile) — and produced
`Copytrader_0.1.0_x64-setup.exe`. Two config bugs (the invalid `security.comment` key and the
`beforeDevCommand`/`beforeBuildCommand` cwd-relative-path bug, both described under Step 1/Step 2
above) had to be fixed first; without those fixes this path fails the same way the native path
would have.

**Why this plan still recommends the native path, not this one:** even a cross-compiled `.exe`
must be copied to a real Windows machine and run there to verify WebView2 behavior (Step 3) — you
cannot run WebView2 on macOS. Since a persistent Windows box already exists for this project,
cross-compiling buys nothing here; it's recorded as a documented alternative in case the Windows
box ever goes away, and it's the same mechanism a Linux CI runner would need if GitHub Actions
ever targets `ubuntu-latest` instead of `windows-latest` for this build.

### Longer-term: CI

A GitHub Actions `windows-latest` runner (installing the same Rust/Node toolchain via
`actions/setup-node` + `dtolnay/rust-action`, then `npm run build:windows`) is the more repeatable
long-term option and avoids depending on the physical Windows box's local state. For a single
monitoring client the local Windows-host build above is sufficient — **documented follow-up, not
built now.**

## Step 3 — Verify WebView2's network-origin behavior (mixed content is resolved; Local Network Access is the live open question)

Files: `desktop/src-tauri/tauri.conf.json` (no change expected — see below);
`desktop/src-tauri/capabilities/default.json` (confirmed **not** relevant — see below)

### What's now resolved (not a spike item)

The previous version of this plan flagged "WebView2 will likely block plain-`http://` fetches as
mixed content, because Tauri serves the app from a secure `https://tauri.localhost` origin." This
was the correct v1 behavior but is **not** Tauri v2's default, confirmed against the live v2 config
schema and the official v1→v2 migration guide:

- `WindowConfig.useHttpsScheme` (settable per-entry in `app.windows[]`) — schema description:
  *"Sets whether the custom protocols should use `https://<scheme>.localhost` instead of the
  default `http://<scheme>.localhost` on Windows and Android."* **Default: `false`.**
- The Tauri v2 migration guide states this explicitly: *"On Windows the frontend files in
  production apps are now hosted on `http://tauri.localhost` instead of `https://tauri.localhost`"*
  — this was a deliberate fix (PR `tauri-apps/tauri#7645`, "Allow http origin on Windows") for
  exactly the mixed-content-blocks-`ws://`/`fetch()` problem this plan used to worry about
  (originating from `tauri-apps/tauri#3007`).
- `desktop/src-tauri/tauri.conf.json`'s `app.windows[0]` entry (lines 13-22, the `"main"` window)
  does **not** set `useHttpsScheme`, so it stays at the default `false`. **Do not add
  `"useHttpsScheme": true`** — that would reintroduce the exact mixed-content bug this plan
  previously worried about, since it would also require the Beelink backend to be HTTPS to keep
  working (it isn't — see `deploy.md`'s plain-`http://<ip>:8000` backend URLs). It would also reset
  the app's one piece of persisted browser storage (the zustand `persist` key
  `"copytrader-settings"` in `frontend/src/stores/settings.ts:17`, which holds `backendUrl`) per
  the migration guide's storage-reset caveat — low-stakes here (one setting), but another reason
  not to touch it.
- A plain-`http://` page fetching another plain-`http://` (or `ws://`) LAN resource is **not**
  mixed content — the mixed-content algorithm only restricts an HTTPS page loading insecure
  subresources. So: **no `additionalBrowserArgs` flag and no reverse proxy should be needed by
  default**, for both call sites that matter — `apiFetch()` in `frontend/src/lib/api.ts:17-27`
  (plain `fetch()`  against `useSettings.getState().backendUrl`) and `createBackendSocket()` in
  `frontend/src/lib/ws.ts:11-14` (turns the same `backendUrl` into `ws://…` via
  `.replace(/^http/, "ws")`).
- `app.security.csp: null` (`tauri.conf.json:26`) is unrelated to this question either way — CSP
  and the browser's secure-context/mixed-content policy are independent mechanisms. `csp: null`
  never was going to fix or break the http/https question; ruling this out explicitly since it's a
  common point of confusion.
- `capabilities/default.json` (`permissions: ["core:default"]`) governs Tauri **IPC** command
  access, not the webview's own network stack. This app registers zero custom Tauri commands
  (`src-tauri/src/lib.rs`'s entire body is `tauri::Builder::default().run(...)`, per its own
  comment: *"there are no custom Tauri commands to maintain"*), so nothing here needs to change for
  Windows. This rules out the previous plan's "(possibly `capabilities/default.json`)" hedge
  entirely — there is no plausible edit to this file that would affect LAN fetch behavior.

### What's still genuinely open — Local Network Access (LNA), needs the real Windows box

There is a *different*, currently-rolling-out Chromium/WebView2 restriction that the previous plan
didn't know to look for: **Local Network Access** (Chromium's evolving Private Network Access
model), which blocks a page from reaching private-network addresses (`localhost`, `192.168.x.x`,
`10.x.x.x` — i.e. exactly the Beelink/Windows-box LAN addresses this app targets) **regardless of
scheme** — this is not a mixed-content check, so the `http://tauri.localhost` fix above does not
address it. Per Microsoft's official WebView2 announcement
(`MicrosoftEdge/WebView2Announcements#126`, checked 2026-07-07):

- **WebView2 143–144:** LNA is behind a kill switch — pre-LNA behavior, i.e. **not blocking**.
- **WebView2 145 (shipped 2026-02-12, so plausibly already on the target Windows box today):**
  LNA ships but is **disabled by default**; a force-allow test flag
  `msWebViewAllowLocalNetworkAccessChecks` exists for opting in early.
- **Future (unscheduled):** Microsoft plans to extend `CoreWebView2PermissionKind` with LNA values
  and route allow/deny through `SetPermissionStateAsync` / the `PermissionRequested` event — but
  per the announcement this app-facing permission control is **still "planned for a future
  release,"** not shipped yet.

Net effect: **a stock build should work today** (LNA off by default in the current shipping
version), but WebView2 auto-updates silently on end-user machines (same Evergreen channel as Edge)
on a timeline this project doesn't control, and Microsoft's stated direction is toward eventually
turning LNA on. A future silent WebView2 update could break LAN fetch on the monitor box with no
corresponding app change to point to.

**Status: not yet done — this is the only remaining open item in the whole plan, and it genuinely
needs the real Windows box.** The installer itself is no longer a blocker (Step 2 already produced
`Copytrader_0.1.0_x64-setup.exe` via the macOS cross-compile path and it's been handed off) — what's
left is purely the by-hand verification below, run *after* copying that `.exe` onto the Windows
box:

1. **Install and launch it.** It's unsigned, so Windows Defender SmartScreen will show "Windows
   protected your PC" — click "More info" → "Run anyway." (`nsis.installMode` defaults to
   `currentUser`, so this installs to `%LOCALAPPDATA%` with no UAC/admin prompt.)
2. Confirm the installed WebView2 Runtime version (Settings → Apps → "Microsoft Edge WebView2
   Runtime" → version number) to know which side of the 145 boundary the box is on.
3. Point it at the Beelink/Windows-box backend URL (Settings → backend URL); confirm both the
   REST calls (`apiFetch`) and the live WebSocket feed (`useLiveState` in `frontend/src/lib/ws.ts`)
   connect and stay connected (the reconnect-after-close logic at `ws.ts:46-48` retries every 2s,
   so a transient LNA prompt/denial would show as "reconnecting forever" rather than a hard crash —
   watch for that pattern specifically).
4. Proactively test against LNA turned **on**, ahead of Microsoft flipping the default, by
   temporarily setting (for this spike only, then reverting)
   `"app": {"windows": [{..., "additionalBrowserArgs": "--enable-features=LocalNetworkAccessChecks,msWebViewAllowLocalNetworkAccessChecks --disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection"}]}`
   in `tauri.conf.json` and rebuilding (from macOS via the cross-compile path, or natively on the
   Windows box — either now works per Step 2). (Note: setting `additionalBrowserArgs` at all
   replaces wry's own default `--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection`
   flag, per the Tauri schema description — the command above re-adds it explicitly so this test
   doesn't also change unrelated Edge UI behavior.) If the app still connects with LNA forced on,
   no further action is needed even after Microsoft's future default flip. If it breaks, the
   eventual fix will be a WebView2/Tauri version bump and possibly Rust-side handling of the
   `PermissionRequested` event once Microsoft ships it — not a config tweak — so record the result
   either way.
5. Known unrelated footgun to avoid while touching this config: `tauri-apps/tauri#13092` reports
   `additionalBrowserArgs` (any value, even `""`) causing *secondary* `WebviewWindow`s to render
   blank in some Tauri 2 releases. Irrelevant to this app today (it only ever creates the one
   `"main"` window declared in `tauri.conf.json:13-22`), but don't reach for
   `additionalBrowserArgs` as a first-line fix for anything else without checking whether this
   project's Tauri/wry version is affected — only use it for the LNA test in step 4 above, and
   revert it afterward unless the test shows it's actually needed in production.

If step 3 above shows LNA is a real, current problem (unlikely per the version numbers, but
possible if the box has a newer preview channel or Microsoft accelerated the rollout since this was
last checked), fall back to the previously-considered options: a `--enable-features=...` allow flag
made permanent, or a local HTTPS-with-valid-LAN-cert reverse proxy in front of the Beelink (Caddy
with an internal CA) — noting a reverse proxy doesn't obviously help LNA either, since LNA blocks
by *destination address*, not by scheme, so this would need re-verifying against whatever the
actual failure mode turns out to be.

## Step 4 — Update docs, defer code signing

**Status: done (2026-07-07).** Files: `deploy.md`, `desktop/README.md` (already covered by Step 2's
edits — no further change needed).

**`deploy.md`'s "Windows test box" section (currently lines 6-15)** describes it as a full Docker
Compose backend host:

```powershell
git clone <repo> ; cd polymarket_copy
copy docker\.env.example .env    # then edit
docker compose -f docker/docker-compose.yml up -d --build
```

Since the Beelink (`deploy.md:17-26`) is the sole always-on backend host going forward, rewrite
this section to describe the box's new role: build-and-run environment for the Windows desktop
client, not a backend host. Replace the `docker compose` instructions with the Step 2 prerequisites
(Rust MSVC toolchain, VS Build Tools "Desktop development with C++" workload, WebView2 Runtime —
Node/git already present) and a pointer to `desktop/scripts/build-windows.ps1`. Keep (or move to
the Beelink section) the firewall-port note only if the box still runs anything that listens on a
LAN port; if its only remaining job is running the desktop client (an outbound-only HTTP/WS
client), it needs **no inbound firewall exceptions** of its own — that requirement moves entirely
to whichever host runs the backend (already documented for the Beelink at `deploy.md:17-26`).

**Add a `.exe`/NSIS section** paralleling the existing "Desktop `.dmg`" section
(`deploy.md:70-127`), with the same four subsections:

- *Build prerequisites (Windows host)* — the Step 2 prerequisite list.
- *Build* — `./desktop/scripts/build-windows.ps1` (or `npm run build:windows`), output path
  `desktop/src-tauri/target/release/bundle/nsis/Copytrader_<version>_x64-setup.exe`.
- *Pointing at a backend* — identical UX to the `.dmg` (`deploy.md:99-107`): Settings → backend URL
  → `http://<beelink-ip>:8000`, persisted via the same zustand `persist` store, no rebuild needed.
  Note (replacing the ATS explanation that's macOS-specific): Windows permits this out of the box
  because Tauri serves the app from `http://tauri.localhost` by default — see `docs/windows.md`
  Step 3 for the full explanation and the one residual risk (WebView2 Local Network Access) to
  watch for on future WebView2 updates.
- *Signing (distribution)* — **out of scope/deferred**, mirroring the unsigned-`.dmg`/Gatekeeper
  posture already documented at `deploy.md:109-126`. An unsigned NSIS installer triggers Windows
  Defender SmartScreen; install via "More info → Run anyway." For future reference, the exact
  config keys (all present in Tauri v2's `bundle.windows` schema, alongside the already-referenced
  `nsis`/`webviewInstallMode`) are `certificateThumbprint`, `digestAlgorithm`, `timestampUrl`, and
  `signCommand` — an Authenticode code-signing certificate (not the free Apple-equivalent; these
  cost money from a CA and increasingly require EV/hardware-token certs to avoid SmartScreen
  entirely) would populate these, exactly analogous to the `APPLE_SIGNING_IDENTITY`/etc. env vars
  `build-dmg.sh` reads today. Not needed for a personal monitoring client.

**`desktop/README.md`** — add `scripts/build-windows.ps1` to the layout tree (currently
`README.md:20-22`, alongside `build-dmg.sh` and `generate_icons.py`), and add a "Build the `.exe`
(NSIS)" section paralleling "Build the `.dmg`" (`README.md:32-39`), with the same prerequisites
note.

## Footnote: packaging may not even be necessary

The frontend already runs unmodified in any browser pointed at the Beelink's backend URL. With
Step 3's mixed-content question resolved (both a packaged Tauri app and a plain browser tab now
face the identical http-page-fetches-http-LAN-resource situation — neither is "mixed content"),
the zero-packaging alternative is even more clearly viable than before: "just open a browser" and
a native NSIS install behave the same way over the network. What packaging still buys is a real
app icon, window chrome, and a Start Menu entry instead of a browser tab — not different network
access. If the LNA spike in Step 3 turns up a real, unfixable problem, "just open a browser" remains
a legitimate zero-packaging fallback to all four steps above (a browser tab would face the exact
same LNA restriction if/when Chromium enables it broadly, so this fallback is about avoiding
packaging effort, not about dodging LNA specifically).

### Files touched across this plan

- `desktop/src-tauri/tauri.conf.json` (`bundle.targets`; `app.security.csp` comment; optionally
  `app.windows[0].additionalBrowserArgs` transiently, for the Step 3 spike only)
- `desktop/package.json` (`build:windows` script)
- `desktop/scripts/build-windows.ps1` (new)
- `deploy.md` (rewrite "Windows test box" section; add parallel `.exe`/NSIS section)
- `desktop/README.md` (layout tree; parallel NSIS build section)
