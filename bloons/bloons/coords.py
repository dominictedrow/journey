"""Portable coordinate math between normalized game coords and screen points / image pixels.

Normalized coords are 0-1 over the game's *content area*: the window minus its title bar
(see data/ui.json). Bounds are (x, y, w, h) in screen points, as macOS reports them.
"""


def content_rect(bounds, title_bar_pt):
    """Content area of a window in screen points: (x, y, w, h)."""
    x, y, w, h = bounds
    return x, y + title_bar_pt, w, h - title_bar_pt


def to_screen(nx, ny, bounds, title_bar_pt):
    """Normalized (nx, ny) -> screen point (x, y)."""
    cx, cy, cw, ch = content_rect(bounds, title_bar_pt)
    return cx + nx * cw, cy + ny * ch


def to_norm(x, y, bounds, title_bar_pt):
    """Screen point (x, y) -> normalized (nx, ny). The inverse of to_screen."""
    cx, cy, cw, ch = content_rect(bounds, title_bar_pt)
    return (x - cx) / cw, (y - cy) / ch


def box_to_pixels(box, img_w, img_h, title_bar_px):
    """Normalized [x0, y0, x1, y1] -> integer pixel rect (x, y, w, h) in a full-window image
    whose top `title_bar_px` rows are the title bar."""
    ch = img_h - title_bar_px
    x0, y0 = round(box[0] * img_w), round(title_bar_px + box[1] * ch)
    x1, y1 = round(box[2] * img_w), round(title_bar_px + box[3] * ch)
    return x0, y0, x1 - x0, y1 - y0
