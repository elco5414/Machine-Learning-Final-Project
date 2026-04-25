import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


MODELS = {
    "5-day":  {"path": "models/model_5.pth", "data": "data/training_data_5.parquet"},
    "20-day": {"path": "models/model_20.pth", "data": "data/training_data_20.parquet"},
    "60-day": {"path": "models/model_60.pth", "data": "data/training_data_60.parquet"},
}

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "volume_change", "volume_ratio", "action"
]

TEST_SPLIT  = 0.1
OUTPUT_PATH = "figures/predicted_vs_actual.png"
os.makedirs("figures", exist_ok=True)


class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output_head(self.dropout(hidden[-1])).squeeze(-1)


class StockDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        return self.features[idx:idx + self.seq_len], self.labels[idx + self.seq_len]


def get_test_predictions(model_path, data_path):
    """Load model + test data and return (actual, predicted) arrays."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    config     = checkpoint["config"]

    model = StockPredictor(
        config["input_size"], config["hidden_size"],
        config["num_layers"], config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load and prep data (matching training exactly)
    df = pd.read_parquet(data_path)
    features = df[FEATURE_COLS].values.astype(np.float32)
    labels   = df["label"].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels   = np.nan_to_num(labels,   nan=0.0, posinf=0.0, neginf=0.0)

    # Same train/val/test split as training
    test_end = len(features)
    val_end  = int(len(features) * 0.9)
    train_end = int(len(features) * 0.8)

    scaler = StandardScaler()
    scaler.fit(features[:train_end])
    test_features = scaler.transform(features[val_end:])
    test_labels   = labels[val_end:]

    test_dataset = StockDataset(test_features, test_labels, config["seq_len"])
    test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)

    actuals, predictions = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            predictions.extend(pred)
            actuals.extend(y.numpy())

    return np.array(actuals), np.array(predictions)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, info) in zip(axes, MODELS.items()):
    try:
        actual, predicted = get_test_predictions(info["path"], info["data"])
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, f"Data not found:\n{e.filename}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(f"{name} Model")
        continue

    # downsample to 5000 points for readable scatter
    if len(actual) > 5000:
        idx = np.random.choice(len(actual), 5000, replace=False)
        actual    = actual[idx]
        predicted = predicted[idx]

    ax.scatter(actual, predicted, alpha=0.2, s=8, color="#3b82f6")

    # perfect prediction line
    lim = max(abs(actual).max(), abs(predicted).max())
    ax.plot([-lim, lim], [-lim, lim], "--", color="#ef4444",
            linewidth=1.5, label="Perfect prediction")

    # correlation
    corr = np.corrcoef(actual, predicted)[0, 1]
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    ax.set_title(f"{name} Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return" if ax is axes[0] else "")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    ax.text(0.98, 0.02,
            f"Correlation: {corr:.3f}\nRMSE: {rmse:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

fig.suptitle("Predicted vs Actual Returns (Test Set)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"✓ Saved to {OUTPUT_PATH}")
plt.show()
