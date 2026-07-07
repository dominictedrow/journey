# Build the Copytrader Windows NSIS installer.
#
# Prereqs (Windows host): Node 18+, Rust stable (MSVC toolchain via https://rustup.rs — accept the
# default x86_64-pc-windows-msvc host triple, not -gnu), and Visual Studio Build Tools with the
# "Desktop development with C++" workload (provides link.exe). See docs/windows.md Step 2 for the
# full prerequisite list, including the cross-compile-from-macOS alternative.
#
# Usage (from a PowerShell prompt, any working directory):
#   .\desktop\scripts\build-windows.ps1
#
# The frontend static export runs automatically via tauri's beforeBuildCommand.

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Desktop = Split-Path -Parent $Here
$Root = Split-Path -Parent $Desktop

Write-Host "==> Installing frontend deps"
npm --prefix "$Root/frontend" install

Write-Host "==> Installing desktop (Tauri CLI) deps"
npm --prefix "$Desktop" install

# Regenerate icons only if they're missing (keeps a committed set as the default artwork).
if (-not (Test-Path "$Desktop/src-tauri/icons/icon.ico")) {
    Write-Host "==> Generating app icons"
    python "$Here/generate_icons.py"
}

Write-Host "==> Building NSIS installer (this runs the frontend static export first)"
npm --prefix "$Desktop" run tauri -- build --bundles nsis

$NsisDir = "$Desktop/src-tauri/target/release/bundle/nsis"
Write-Host "`n==> Done. Installer(s):"
Get-ChildItem "$NsisDir/*.exe" -ErrorAction SilentlyContinue
