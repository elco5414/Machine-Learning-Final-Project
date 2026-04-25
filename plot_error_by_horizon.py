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

OUTPUT_PATH = "figures/error_by_horizon.png"
os.makedirs("figures", exist_ok=True)


class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
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



def compute_test_metrics(model_path, data_path):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    config     = checkpoint["config"]

    model = StockPredictor(
        config["input_size"], config["hidden_size"],
        config["num_layers"], config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_parquet(data_path)
    features = df[FEATURE_COLS].values.astype(np.float32)
    labels   = df["label"].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels   = np.nan_to_num(labels,   nan=0.0, posinf=0.0, neginf=0.0)

    train_end = int(len(features) * 0.8)
    val_end   = int(len(features) * 0.9)

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

    actuals     = np.array(actuals)
    predictions = np.array(predictions)

    return {
        "rmse": np.sqrt(np.mean((actuals - predictions) ** 2)),
        "mae":  np.mean(np.abs(actuals - predictions)),
        "corr": np.corrcoef(actuals, predictions)[0, 1]
    }


results = {}
for name, info in MODELS.items():
    try:
        results[name] = compute_test_metrics(info["path"], info["data"])
        print(f"✓ {name}: RMSE={results[name]['rmse']:.4f}, "
              f"MAE={results[name]['mae']:.4f}, Corr={results[name]['corr']:.3f}")
    except FileNotFoundError as e:
        print(f"✗ {name}: file not found — {e.filename}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
names  = list(results.keys())
colors = ["#3b82f6", "#8b5cf6", "#ec4899"]

metrics = [
    ("rmse", "Root Mean Squared Error (RMSE)"),
    ("mae",  "Mean Absolute Error (MAE)"),
    ("corr", "Correlation (Predicted vs Actual)")
]

for ax, (key, title) in zip(axes, metrics):
    values = [results[name][key] for name in names]
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=2)

    # value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}" if key != "corr" else f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)

fig.suptitle("Model Performance Across Prediction Horizons",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved to {OUTPUT_PATH}")
plt.show()