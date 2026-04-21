"""
Trains an LSTM model on the processed S&P 500 data to predict
the % return of a stock over the next 5 trading days.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


DATA_PATH = "data/training_data.parquet"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npy")

SEQUENCE_LEN = 30  # How many past days the model looks at
BATCH_SIZE = 512  # How many samples per training step
EPOCHS = 30  # How many times to train over the full dataset
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128  # Size of LSTM hidden layer
NUM_LAYERS = 2  # Number of stacked LSTM layers
DROPOUT = 0.2  # Dropout rate to prevent overfitting
VAL_SPLIT = 0.1  # 10% of data used for validation
TEST_SPLIT = 0.1  # 10% of data used for final testing

FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_10d",
    "volatility_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_position",
    "volume_change",
    "volume_ratio",
    "action",
]

os.makedirs(MODEL_DIR, exist_ok=True)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock sequences.

    For each sample, instead of feeding just one row of features,
    we feed a SEQUENCE of the last SEQUENCE_LEN days. This lets
    the LSTM learn from the pattern of recent history, not just
    a single snapshot.

    Example:
        SEQUENCE_LEN = 30
        Input:  30 days of features  →  shape (30, 17)
        Output: % return on day 31   →  shape (1,)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]  # (seq_len, num_features)
        y = self.labels[idx + self.seq_len]  # scalar
        return x, y


class StockPredictor(nn.Module):
    """
    LSTM-based stock return predictor.

    Architecture:
        Input  →  LSTM layers  →  Dropout  →  Linear  →  Predicted % return

    The LSTM processes the sequence of past days and the final
    hidden state is passed through a linear layer to produce
    the predicted return.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super(StockPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # input shape: (batch, seq_len, features)
        )

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # single output: predicted % return
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hidden, _) = self.lstm(x)

        # Take the last hidden state (most recent timestep)
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        last_hidden = self.dropout(last_hidden)

        out = self.output_head(last_hidden)  # (batch_size, 1)
        return out.squeeze(-1)  # (batch_size,)


def load_data():
    print("Loading training data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows, {df['Ticker'].nunique()} tickers\n")

    # Sort by ticker then date so sequences are chronologically correct
    df = (
        df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        if "Date" in df.columns
        else df.sort_index()
    )

    features = df[FEATURE_COLS].values.astype(np.float32)
    labels = df["label"].values.astype(np.float32)

    return features, labels


def prepare_data(features: np.ndarray, labels: np.ndarray):
    """Scale features and split into train/val/test sets."""

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    # Split BEFORE scaling to prevent data leakage
    # (scaler should only be fit on training data)
    train_end = int(len(features) * (1 - VAL_SPLIT - TEST_SPLIT))
    val_end = int(len(features) * (1 - TEST_SPLIT))

    train_features = features[:train_end]
    val_features = features[train_end:val_end]
    test_features = features[val_end:]

    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]

    # Fit scaler on training data only, then apply to all splits
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    # Save scaler — needed later for inference in FastAPI
    np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_})
    print(f"  Scaler saved to {SCALER_PATH}")

    print(f"  Train: {len(train_features):,} rows")
    print(f"  Val:   {len(val_features):,} rows")
    print(f"  Test:  {len(test_features):,} rows\n")

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        scaler,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training data."""
    model.train()
    total_loss = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def train():
    print("=" * 50)
    print("  Stock Market Predictor — Model Training")
    print("=" * 50, "\n")

    # Device — use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}\n")

    # Load and prepare data
    features, labels = load_data()
    (train_feat, train_labels, val_feat, val_labels, test_feat, test_labels, scaler) = (
        prepare_data(features, labels)
    )

    # Create datasets and loaders
    train_dataset = StockDataset(train_feat, train_labels, SEQUENCE_LEN)
    val_dataset = StockDataset(val_feat, val_labels, SEQUENCE_LEN)
    test_dataset = StockDataset(test_feat, test_labels, SEQUENCE_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = StockPredictor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}\n")

    # Loss, optimizer, and learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 5  # Stop if val loss doesn't improve for 5 epochs
    patience_count = 0

    print("  Epoch | Train Loss | Val Loss  | Status")
    print("  " + "-" * 45)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "input_size": len(FEATURE_COLS),
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "seq_len": SEQUENCE_LEN,
                        "features": FEATURE_COLS,
                    },
                },
                MODEL_PATH,
            )
            status = "✓ saved"
        else:
            patience_count += 1
            status = f"no improvement ({patience_count}/{patience})"

        print(f"  {epoch:5d} | {train_loss:.6f}   | {val_loss:.6f}  | {status}")

        # Early stopping
        if patience_count >= patience:
            print(
                f"\n  Early stopping at epoch {epoch} — no improvement for {patience} epochs."
            )
            break

    # Final evaluation on test set
    print("\n  Loading best model for final evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = validate(model, test_loader, criterion, device)

    print("\n" + "=" * 50)
    print("  Training Complete!")
    print("=" * 50)
    print(f"  Best val loss:  {best_val_loss:.6f}")
    print(f"  Test loss:      {test_loss:.6f}")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    train()
