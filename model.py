"""
Trains an LSTM model on the processed S&P 500 data to predict
the % return of a stock over the next prediction_horizon trading days.
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

PREDICTION_HORIZON = 60

DATA_PATH = "data/training_data.parquet"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_" + str(PREDICTION_HORIZON) + ".pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_" + str(PREDICTION_HORIZON) + ".npy")

SEQUENCE_LEN = 120  # How many past days the model looks at
BATCH_SIZE = 512  # How many samples per training step
EPOCHS = 40  
LEARNING_RATE = 0.001 
HIDDEN_SIZE = 256  # Size of LSTM hidden layer
NUM_LAYERS = 2  # Number of stacked LSTM layers
DROPOUT = 0.3  
VAL_SPLIT = 0.1 
TEST_SPLIT = 0.1 

# recommended features we found online
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
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = seq_len

    # how many valid samples in dataset
    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len] 
        y = self.labels[idx + self.seq_len] 
        return x, y


class StockPredictor(nn.Module):
    """
    LSTM-based stock return predictor.

    The LSTM processes the sequence of past days and it produces the predicted return.
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
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1), 
        )

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)

        # Take the last hidden state bc it knows the previous days
        last_hidden = hidden[-1] 
        last_hidden = self.dropout(last_hidden)

        out = self.output_head(last_hidden)  # (batch_size, 1)
        return out.squeeze(-1)  # (batch_size,)


def load_data():

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
    """
    Scale features beause all the features are on different scales, so model would take
    the larger scaled ones to be more important even if they arent
    Split into train/val/test sets.
    """

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    train_end = int(len(features) * (1 - VAL_SPLIT - TEST_SPLIT))
    val_end = int(len(features) * (1 - TEST_SPLIT))

    train_features = features[:train_end]
    val_features = features[train_end:val_end]
    test_features = features[val_end:]

    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features) # find the mean and std
    val_features = scaler.transform(val_features) # apply found mean and std
    test_features = scaler.transform(test_features) # apply found mean and std

    np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_})

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
    """Run one epoch the training data, makes the code look cleaner"""

    model.train()
    total_loss = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping which prevents exploding gradients in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Run model on validation set"""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

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

    # Initialize model
    model = StockPredictor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

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

    print("Epoch | Train Loss | Val Loss")
    print("  " + "-" * 45)

    train_losses = []
    val_losses   = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_loss":     val_loss,
                "train_losses": train_losses, 
                "val_losses":   val_losses, 
                "config": {
                    "input_size":  len(FEATURE_COLS),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers":  NUM_LAYERS,
                    "dropout":     DROPOUT,
                    "seq_len":     SEQUENCE_LEN,
                    "features":    FEATURE_COLS,
                    "horizon":     PREDICTION_HORIZON 
                }
            }, MODEL_PATH)
        else:
            patience_count += 1

        print(f"  {epoch:5d} | {train_loss:.6f}   | {val_loss:.6f}")

        # Early stopping
        if patience_count >= patience:
            print("stopped early due to no more improve for patience amount of epochs")
            break

    # Final evaluation on test set
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = validate(model, test_loader, criterion, device)

    print("Done!")
    print("Best val loss: ", best_val_loss)
    print("Test loss: ", test_loss)


if __name__ == "__main__":
    train()
