import json

import matplotlib.pyplot as plt

with open("checkpoints/history.json") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
TRAIN_COLOR = "#2a78d6"  # categorical slot 1 (blue)
VAL_COLOR = "#e34948"    # categorical slot 6 (red)

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(epochs, history["train_loss"], color=TRAIN_COLOR, linewidth=2, label="Train loss")
ax.plot(epochs, history["val_loss"], color=VAL_COLOR, linewidth=2, label="Val loss")

ax.set_xlabel("Epoch", color=INK_SECONDARY)
ax.set_ylabel("Loss (BCE)", color=INK_SECONDARY)
ax.set_title("CatDogCNN training loss — 20 epochs", color=INK_PRIMARY, fontsize=13, loc="left")

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
fig.savefig("checkpoints/loss_curve.png", dpi=150, facecolor=SURFACE)
print("saved checkpoints/loss_curve.png")
