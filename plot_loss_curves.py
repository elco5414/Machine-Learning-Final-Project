import os
import torch
import matplotlib.pyplot as plt

MODELS = {
    "5-day":  "models/model_5.pth",
    "20-day": "models/model_20.pth",
    "60-day": "models/model_60.pth",
}

OUTPUT_PATH = "figures/loss_curves.png"
os.makedirs("figures", exist_ok=True)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
colors = {"train": "#3b82f6", "val": "#ef4444"}

for ax, (name, path) in zip(axes, MODELS.items()):
    checkpoint = torch.load(path, map_location="cpu")

    # check losses were saved
    if "train_losses" not in checkpoint:
        ax.text(0.5, 0.5, f"No loss history\nsaved for {name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title(f"{name} Model")
        continue

    train_losses = checkpoint["train_losses"]
    val_losses   = checkpoint["val_losses"]
    epochs       = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train", color=colors["train"], linewidth=2)
    ax.plot(epochs, val_losses,   label="Val",   color=colors["val"],   linewidth=2)

    # mark the best epoch with a vertical line
    best_epoch = val_losses.index(min(val_losses)) + 1
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_title(f"{name} Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss" if ax is axes[0] else "")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

fig.suptitle("Training vs Validation Loss Across Horizons",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"✓ Saved to {OUTPUT_PATH}")
plt.show()
