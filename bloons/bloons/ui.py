"""Shared UI layout from data/ui.json (portable). Normalized content coords, like the rest.

- `shop_cell(name)`: where a shop entry is ('hero' or a towers.json key) and which scroll
  position shows it.
- `panel(side)`: the upgrade/hero panel's points and boxes for 'left' or 'right'
  (state.read_screen() reports the side).
- `ability_slot(i, panel_side)`: the i-th icon in the bottom-left ability bar.
"""
import json
from pathlib import Path

UI = json.loads((Path(__file__).parent / "data" / "ui.json").read_text())


def shop_cell(name):
    """-> ('top' | 'bottom', (x, y)). Raises KeyError for an unknown name."""
    s = UI["shop"]
    i = s["order"].index(name) if name in s["order"] else None
    if i is None:
        raise KeyError(name)
    row, col = divmod(i, 2)
    x = s["columns_x"][col]
    if s["top_rows"][0] <= row <= s["top_rows"][1]:
        return "top", (x, s["top_row0_y"] + row * s["row_pitch"])
    if s["bottom_rows"][0] <= row <= s["bottom_rows"][1]:
        last = len(s["order"]) // 2 + len(s["order"]) % 2 - 1
        return "bottom", (x, s["bottom_row13_y"] - (last - row) * s["row_pitch"])
    raise KeyError(f"{name} (row {row}) is visible in neither scroll layout")


def panel(side):
    """Upgrade-panel layout for 'left' or 'right'."""
    if side not in ("left", "right"):
        raise ValueError(side)
    return UI["upgrade_panel"][side]


def ability_slot(i, panel_side=None):
    """Click point of ability bar icon i (0-based). The bar shifts right while the panel is
    open on the left."""
    a = UI["ability_bar"]
    x, y = a["slot0"]
    return x + i * a["slot_pitch"] + (a["panel_left_dx"] if panel_side == "left" else 0), y
