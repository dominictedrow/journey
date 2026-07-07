#!/usr/bin/env python3
"""Generate the Tauri app icon set (PNG + .icns + .ico) from a simple drawn source.

No third-party deps — writes valid PNGs with stdlib zlib/struct, then packs the macOS
`.icns` and Windows `.ico` containers around PNG-encoded members. Replace the `draw()` motif
(or drop in real artwork and re-run) to rebrand; Tauri's own `npm run tauri icon <png>` is the
alternative if you have a single high-res source. Output lands in `desktop/src-tauri/icons/`.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ICONS = Path(__file__).resolve().parent.parent / "src-tauri" / "icons"

# Brand colors (match the dashboard: slate-950 field, sky-500 accent).
BG = (2, 6, 23, 255)  # slate-950
ACCENT = (14, 165, 233, 255)  # sky-500
RING = (56, 189, 248, 255)  # sky-400


def draw(size: int) -> bytes:
    """A rounded slate tile with a sky ring + rising bar motif → raw RGBA bytes."""
    px = bytearray(size * size * 4)
    cx = cy = (size - 1) / 2
    outer = size * 0.40
    inner = size * 0.30
    radius = size * 0.18  # corner rounding of the tile

    def put(x: int, y: int, c: tuple[int, int, int, int]) -> None:
        i = (y * size + x) * 4
        px[i : i + 4] = bytes(c)

    for y in range(size):
        for x in range(size):
            # Rounded-rect background mask.
            dx = max(radius - x, x - (size - 1 - radius), 0)
            dy = max(radius - y, y - (size - 1 - radius), 0)
            if dx * dx + dy * dy > radius * radius:
                put(x, y, (0, 0, 0, 0))
                continue
            put(x, y, BG)
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if inner <= dist <= outer:
                put(x, y, RING)

    # Three rising bars (a mini equity curve) centered in the tile.
    bar_w = max(1, int(size * 0.07))
    gap = max(1, int(size * 0.05))
    heights = (0.16, 0.26, 0.36)
    total_w = 3 * bar_w + 2 * gap
    x0 = int(cx - total_w / 2)
    base = int(cy + size * 0.16)
    for k, h in enumerate(heights):
        bx = x0 + k * (bar_w + gap)
        top = int(base - size * h)
        for y in range(top, base):
            for x in range(bx, bx + bar_w):
                if 0 <= x < size and 0 <= y < size:
                    put(x, y, ACCENT)
    return bytes(px)


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def png_bytes(size: int) -> bytes:
    """Encode a `size`×`size` RGBA image (from draw) as a PNG."""
    raw = draw(size)
    stride = size * 4
    scan = bytearray()
    for y in range(size):
        scan.append(0)  # filter type 0
        scan.extend(raw[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(scan), 9))
        + _chunk(b"IEND", b"")
    )


def write_png(name: str, size: int) -> bytes:
    data = png_bytes(size)
    (ICONS / name).write_bytes(data)
    return data


def write_icns(members: dict[bytes, bytes]) -> None:
    """macOS .icns: 'icns' magic + total length + (OSType, len, PNG) entries."""
    body = b"".join(
        ostype + struct.pack(">I", len(png) + 8) + png for ostype, png in members.items()
    )
    blob = b"icns" + struct.pack(">I", len(body) + 8) + body
    (ICONS / "icon.icns").write_bytes(blob)


def write_ico(pngs: list[bytes]) -> None:
    """Windows .ico wrapping PNG entries (Vista+). width/height byte 0 == 256."""
    count = len(pngs)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = bytearray()
    data = bytearray()
    for png in pngs:
        w, h = struct.unpack(">II", png[16:24])  # IHDR width/height
        entries += struct.pack(
            "<BBBBHHII", w & 0xFF, h & 0xFF, 0, 0, 1, 32, len(png), offset
        )
        data += png
        offset += len(png)
    (ICONS / "icon.ico").write_bytes(header + bytes(entries) + bytes(data))


def main() -> None:
    ICONS.mkdir(parents=True, exist_ok=True)
    # Standard Tauri PNG set.
    p32 = write_png("32x32.png", 32)
    write_png("128x128.png", 128)
    write_png("128x128@2x.png", 256)
    p256 = write_png("icon.png", 256)  # generic app icon
    # macOS icns members (subset of Retina sizes).
    write_icns(
        {
            b"ic11": p32,  # 16@2x
            b"ic07": png_bytes(128),
            b"ic13": p256,  # 128@2x
            b"ic09": png_bytes(512),
            b"ic10": png_bytes(1024),  # 512@2x
        }
    )
    write_ico([p32, p256])
    print(f"Wrote icon set to {ICONS}")


if __name__ == "__main__":
    main()
