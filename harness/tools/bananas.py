"""Banana pickup for farm games: click whatever appeared in the farm strip since the reference.

Bananas only get collected by a click (hover does nothing, 2026-11-15). The farm strip (left of
the map's left path) is static grass, so a frame diff against a clean reference finds them;
blobs bigger than a banana (a new farm) are folded into the reference instead of clicked.

    import bananas; bananas.collect(h)        # call often; first call takes the reference
"""
import cv2
import numpy as np

STRIP = (0.015, 0.03, 0.205, 0.97)   # x0, y0, x1, y1 normalized; the autumn map's farm strip
PATH_BAND = (0.76, 0.88)             # the bottom path crosses the strip
MIN_AREA, MAX_AREA = 300, 9000       # px at 1920x1080. ponytail: per-map strip; add others when we farm them
_ref = {}


def _crop(h):
    im = h.game.grab().content_bgr()
    H, W = im.shape[:2]
    x0, y0, x1, y1 = STRIP
    return im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)], W, H


def collect(h, max_clicks=12):
    crop, W, H = _crop(h)
    if "img" not in _ref:
        _ref["img"] = crop
        return 0
    diff = cv2.absdiff(crop, _ref["img"]).max(axis=2)
    ch = crop.shape[0]
    diff[int((PATH_BAND[0] - STRIP[1]) / (STRIP[3] - STRIP[1]) * ch):
         int((PATH_BAND[1] - STRIP[1]) / (STRIP[3] - STRIP[1]) * ch)] = 0   # bloons pass here
    for t in h.registry.values():    # the farm sprite itself animates (bananas on the tree)
        if t.kind == "banana_farm":
            px = int((t.xy[0] - STRIP[0]) * W); py = int((t.xy[1] - STRIP[1]) * H)
            diff[max(0, py - 110):py + 110, max(0, px - 90):px + 90] = 0
    mask = cv2.morphologyEx((diff > 40).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    clicks, boxes = 0, []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        x, y, w, hh = stats[i, :4]
        if a > MAX_AREA:                      # a farm/tower placed here: make it background
            _ref["img"][y:y + hh, x:x + w] = crop[y:y + hh, x:x + w]
        elif a >= MIN_AREA and clicks < max_clicks:
            cx, cy = cents[i]
            h.game.click(STRIP[0] + cx / W, STRIP[1] + cy / H)
            clicks += 1
            boxes.append((x, y, w, hh))
    if boxes:   # whatever is there now is background (a collected banana, or a hole in the ref)
        h.sleep(0.3)
        now, _, _ = _crop(h)
        for x, y, w, hh in boxes:
            _ref["img"][y:y + hh, x:x + w] = now[y:y + hh, x:x + w]
    return clicks
