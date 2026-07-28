import json

import matplotlib.pyplot as plt

RUNS = [
    ("checkpoints/history_run1.json", "Run 1 — 3ep baseline"),
    ("checkpoints/history_run2.json", "Run 2 — 5ep"),
    ("checkpoints/history_run3.json", "Run 3 — 5ep"),
    ("checkpoints/history_run4_filters16_32_64.json", "Run 4 — 16/32/64 filters"),
    ("checkpoints/history_run5_filters32_64_128_10ep.json", "Run 5 — 32/64/128, 10ep"),
    ("checkpoints/history_run6_filters32_64_128_20ep.json", "Run 6 — 32/64/128, 20ep, dropout 0.5"),
    ("checkpoints/history_run7_filters32_64_128_20ep_dropout03_180x202.json", "Run 7 — 32/64/128, 20ep, dropout 0.3, 180x202"),
    ("checkpoints/history_run8_pool4x4_288x320_dropout03.json", "Run 8 — 4x4 pool, 288x320, dropout 0.3"),
]

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"

# categorical slots 1-8, fixed order
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def style_axis(ax, ylabel, max_epochs):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY)
    ax.set_xticks(range(1, max_epochs + 1, 2))
    ax.set_xlim(0.5, max_epochs + 3.5)  # headroom for end-of-line value labels


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=SURFACE)

histories = []
max_epochs = 0
for (path, label), color in zip(RUNS, SERIES_COLORS):
    with open(path) as f:
        history = json.load(f)
    histories.append((history, label, color))
    max_epochs = max(max_epochs, len(history["val_acc"]))

for history, label, color in histories:
    val_loss = history["val_loss"]
    epochs = range(1, len(val_loss) + 1)
    ax_loss.plot(epochs, val_loss, color=color, linewidth=2, label=label)
    ax_loss.scatter([len(val_loss)], [val_loss[-1]], color=color, s=28, zorder=3)
    ax_loss.annotate(f"{val_loss[-1]:.3f}", (len(val_loss), val_loss[-1]),
                      textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK_SECONDARY, va="center")

for history, label, color in histories:
    val_acc = history["val_acc"]
    epochs = range(1, len(val_acc) + 1)
    ax_acc.plot(epochs, val_acc, color=color, linewidth=2, label=label)
    ax_acc.scatter([len(val_acc)], [val_acc[-1]], color=color, s=28, zorder=3)
    ax_acc.annotate(f"{val_acc[-1]:.3f}", (len(val_acc), val_acc[-1]),
                     textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK_SECONDARY, va="center")

style_axis(ax_loss, "Validation loss (BCE)", max_epochs)
style_axis(ax_acc, "Validation accuracy", max_epochs)
ax_loss.set_title("Validation loss", color=INK_PRIMARY, fontsize=13, loc="left")
ax_acc.set_title("Validation accuracy", color=INK_PRIMARY, fontsize=13, loc="left")

fig.suptitle("CatDogCNN — all runs compared", color=INK_PRIMARY, fontsize=15, x=0.02, ha="left")

legend = ax_acc.legend(frameon=False, loc="lower right", fontsize=8.5)
for text in legend.get_texts():
    text.set_color(INK_PRIMARY)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("checkpoints/run_history_comparison.png", dpi=150, facecolor=SURFACE)
print("saved checkpoints/run_history_comparison.png")
