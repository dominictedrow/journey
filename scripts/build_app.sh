#!/bin/bash
# Build dist/Market Research.app from desktop.py via PyInstaller.
# Generates the .icns icon from icon-512.png with native macOS tools first.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

ICON_SRC="app/static/icons/icon-512.png"
ICONSET="build/AppIcon.iconset"
ICNS="build/AppIcon.icns"

if [ ! -f "$ICON_SRC" ]; then
    echo "error: $ICON_SRC not found" >&2
    exit 1
fi

echo "==> Generating app icon ($ICNS)"
rm -rf "$ICONSET" "$ICNS"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    double=$((size * 2))
    sips -z "$size" "$size" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z "$double" "$double" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$ICNS"

echo "==> Building with PyInstaller"
python3 -m PyInstaller MarketResearch.spec --noconfirm --clean

echo "==> Done: dist/Market Research.app"
echo "    Drag it to /Applications, then double-click to launch (Postgres must be running)."
