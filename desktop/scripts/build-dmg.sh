#!/usr/bin/env bash
#
# Build the Copytrader macOS .dmg thin client.
#
# Prereqs (macOS host): Node 18+, Rust stable (https://rustup.rs), and Xcode command-line
# tools. Signing/notarization are opt-in via env vars (see docs/deploy.md) — without them you
# get a working unsigned .dmg for local install.
#
# Usage:
#   ./desktop/scripts/build-dmg.sh
#
# The frontend static export runs automatically via tauri's beforeBuildCommand.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$(dirname "$HERE")"
ROOT="$(dirname "$DESKTOP")"

echo "==> Installing frontend deps"
npm --prefix "$ROOT/frontend" install

echo "==> Installing desktop (Tauri CLI) deps"
npm --prefix "$DESKTOP" install

# Regenerate icons only if they're missing (keeps a committed set as the default artwork).
if [ ! -f "$DESKTOP/src-tauri/icons/icon.icns" ]; then
  echo "==> Generating app icons"
  python3 "$HERE/generate_icons.py"
fi

echo "==> Building .dmg (this runs the frontend static export first)"
# --bundles dmg keeps the output to just the disk image. Signing/notarization are picked up
# from the environment when set (APPLE_SIGNING_IDENTITY / APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID).
npm --prefix "$DESKTOP" run tauri -- build --bundles dmg

DMG_DIR="$DESKTOP/src-tauri/target/release/bundle/dmg"
echo
echo "==> Done. DMG(s):"
ls -1 "$DMG_DIR"/*.dmg 2>/dev/null || echo "  (no .dmg found — check the tauri build output above)"
