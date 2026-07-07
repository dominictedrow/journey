# Desktop (Tauri 2)

Thin-client macOS `.dmg`. A native webview loads the statically-exported `frontend/` build and
talks to a backend URL configured in **Settings** — the same app points at the Windows test box
or the Beelink with no rebuild.

## Layout

```
desktop/
├── package.json                 # Tauri CLI + npm scripts (dev / build / icons)
├── src-tauri/
│   ├── Cargo.toml               # Rust crate (thin launcher over tauri)
│   ├── build.rs                 # tauri-build
│   ├── tauri.conf.json          # window, bundle (dmg), frontendDist = ../../frontend/out
│   ├── Info.plist               # macOS ATS exception for plain-http LAN backends
│   ├── capabilities/default.json# window permissions (core defaults only)
│   ├── src/{main.rs,lib.rs}     # `tauri::Builder::default().run(...)`
│   └── icons/                   # generated PNG + .icns + .ico set
└── scripts/
    ├── build-dmg.sh             # one-command build → target/release/bundle/dmg
    ├── build-windows.ps1        # one-command build → target/release/bundle/nsis
    └── generate_icons.py        # stdlib icon generator (swap in real artwork)
```

## Develop

```bash
npm install
npm run dev        # next dev + webview against http://localhost:3000
```

## Build the .dmg

```bash
./scripts/build-dmg.sh
```

Requires Node 18+, Rust stable, and Xcode CLT on macOS. Signing/notarization and backend-URL
details are in [`docs/deploy.md`](../docs/deploy.md).

## Build the .exe (NSIS)

```powershell
.\scripts\build-windows.ps1
```

Requires Node 18+, Rust stable (MSVC toolchain), and Visual Studio Build Tools' "Desktop
development with C++" workload on Windows. Builds natively on a Windows host; cross-compiling from
macOS/Linux via `cargo-xwin` is documented as an alternative in
[`docs/windows.md`](../docs/windows.md) Step 2. Output installer path and the WebView2
Local Network Access spike are also covered there.

## Why static export

Every dashboard page is client-rendered and fetches from the backend at runtime, so the frontend
builds to a static bundle (`BUILD_TARGET=desktop` → `output: "export"` → `frontend/out`) that
Tauri ships as-is. The Docker deployment still uses the default `standalone` server build — same
codebase, two targets.
