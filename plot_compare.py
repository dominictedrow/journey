import json

import matplotlib.pyplot as plt

with open("checkpoints/history_run4_filters16_32_64.json") as f:
    base = json.load(f)
with open("checkpoints/history.json") as f:
    wide = json.load(f)

n = 10  # compare first 10 epochs of both runs
epochs = range(1, n + 1)

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
BASE_COLOR = "#2a78d6"   # categorical slot 1 (blue)
WIDE_COLOR = "#eb6834"   # categorical slot 8 (orange)

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(epochs, base["val_loss"][:n], color=BASE_COLOR, linewidth=2,
        label="16/32/64 filters — val loss")
ax.plot(epochs, wide["val_loss"][:n], color=WIDE_COLOR, linewidth=2,
        label="32/64/128 filters — val loss")

ax.set_xlabel("Epoch", color=INK_SECONDARY)
ax.set_ylabel("Val loss (BCE)", color=INK_SECONDARY)
ax.set_title("Filter width comparison — first 10 epochs", color=INK_PRIMARY, fontsize=13, loc="left")

ax.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(AXIS)

ax.tick_params(colors=INK_SECONDARY)
ax.set_xticks(list(epochs))

legend = ax.legend(frameon=False, loc="upper right")
for text in legend.get_texts():
    text.set_color(INK_PRIMARY)

fig.tight_layout()
fig.savefig("checkpoints/loss_curve_comparison.png", dpi=150, facecolor=SURFACE)
print("saved checkpoints/loss_curve_comparison.png")
