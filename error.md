# Error Report — Charts Extremely Elongated (All Pages Except Home Shifts)

## Root Cause

**Chart.js `responsive: true` + `maintainAspectRatio: false` reads the canvas's *parent element* height — not the canvas's own inline `height` style. Every affected canvas lives inside a `.card` that is a grid item in `.grid-2`, `.grid-3`, or `.grid-4`. CSS Grid's default `align-items: stretch` expands those `.card` divs to match the tallest item in their row. Chart.js then sizes the canvas to fill that stretched parent height, producing an extremely elongated chart.**

---

## Why the Home Shifts Chart Is Immune

`home/index.html` line 112 wraps the Stories + Shifts section in:
```html
<div class="grid-2 mb-16" style="align-items:start;">
```
`align-items: start` disables stretch — each grid item shrinks to its own content height. The `.card` containing `shiftsChart` is only as tall as the canvas + padding, so Chart.js reads the correct height.

Every other page uses bare `.grid-2 / .grid-3 / .grid-4` with no `align-items` override, leaving the default `stretch` active.

---

## Failure Chain

1. A `.card` is a grid item inside `.grid-2`, `.grid-3`, or `.grid-4` (no `align-items` set).
2. CSS Grid `align-items: stretch` (the default) stretches the `.card` to the full row height — which may be hundreds or thousands of pixels if another card in the same row contains more text.
3. Chart.js is initialised with `responsive: true, maintainAspectRatio: false`.
4. Chart.js ResizeObserver reads `canvas.parentElement.clientHeight` (the stretched `.card` height).
5. Chart.js sets the canvas `height` to that value, overriding the canvas's own inline `height` style completely.
6. Result: sparklines intended to be 38–50 px tall become 300–1200 px tall; the P&L chart on the portfolio page was measured at 1184 px.

---

## Affected Templates and Canvas Sizes

| Template | Canvas selector | Intended height |
|---|---|---|
| `assets/us.html` | `canvas.spark-*` (equity + macro grids) | 38 px |
| `assets/international.html` | `canvas.spark-*` (country + fx grids) | 38 px |
| `assets/crypto.html` | `canvas.spark-*` (price + price2 grids) | 34–50 px |
| `assets/crypto.html` | `#chart-btc-dom`, `#chart-fear-greed`, `#chart-etf-flows`, `#chart-stablecoin` | 180 px |
| `routines/daily.html` | `canvas.spark-*` | 38 px |
| `routines/weekly.html` | `canvas.spark-*` | 38 px |
| `routines/monthly.html` | `canvas.spark-*` (tier-1 + tier-2 grids) | 34–38 px |
| `portfolio/stocks.html` | `#chart-pnl` | 160 px |
| `framework/tier.html` | `canvas.sparkline` | 44 px |

---

## Fix

Chart.js documentation states: *"The parent element's height must be defined by something other than the chart itself."* The canonical fix is to wrap every affected `<canvas>` in a **positioned container div with an explicit height**. Chart.js reads the container height; the canvas fills it exactly.

### Pattern to apply everywhere

**Before (broken):**
```html
<canvas id="spark-XYZ" style="height:38px;width:100%;"></canvas>
```

**After (fixed):**
```html
<div style="position:relative;height:38px;">
  <canvas id="spark-XYZ" style="width:100%;height:100%;"></canvas>
</div>
```

The `position:relative` is required — Chart.js's ResizeObserver uses it as the resize anchor.

### Heights to use per location

| Location | Wrapper height |
|---|---|
| All sparkline canvases (`spark-*`) in asset / routine pages | `38px` (or match existing inline value per template) |
| `assets/crypto.html` price2 grid sparklines | `34px` |
| `assets/crypto.html` large charts (btc-dom, fear-greed, etf-flows, stablecoin) | `180px` |
| `framework/tier.html` `.sparkline` canvases | `44px` |
| `portfolio/stocks.html` `#chart-pnl` | `160px` |

The doughnut chart (`#chart-donut`) uses `responsive: false` and is not affected.

### Scope of change

Only Jinja2 template files require edits — one wrapping `<div>` per `<canvas>`. No Python, no CSS, no JS chart options, and no data pipeline changes are needed.
