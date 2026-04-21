"""
price_model.py

Trains a single XGBoost regressor on all S&P 500 tickers to predict N-day-ahead
log returns, where N (days_ahead) is itself a feature. At prediction time we
convert the predicted log return back into a price:

    predicted_price = current_price * exp(predicted_log_return)

Why log returns instead of raw prices:
- Raw prices are non-stationary (they trend upward over time), so a model
  trained on them mostly learns "predict yesterday's price."
- Log returns are roughly stationary and symmetric around zero, which is what
  tree-based regressors handle well.

Usage:
    python price_model.py

Outputs:
    models/price_model.json   -- trained XGBoost model
    models/feature_cols.json  -- ordered list of feature columns (for predict.py)
    models/metadata.json      -- training config + metrics
"""

import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

# -------- config --------
START_DATE = "2015-01-01"
END_DATE = None  # None = today
MAX_HORIZON = 90  # days ahead
HORIZONS = [1, 5, 10, 21, 42, 63, 90]  # training horizons sampled per row
TEST_SPLIT_DATE = "2024-01-01"  # chronological split
MODEL_DIR = "models/"
RANDOM_SEED = 42

FEATURE_COLS = [
    # lagged log returns
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    # rolling stats of daily log returns
    "mean_ret_5",
    "mean_ret_21",
    "mean_ret_63",
    "std_ret_5",
    "std_ret_21",
    "std_ret_63",
    # volume signal
    "vol_change_1",
    "vol_change_5",
    # seasonality
    "day_of_week",
    "month",
    # horizon (tells the model how far ahead to predict)
    "days_ahead",
]

TARGET_COL = "target_log_return"


# -------- data loading --------
def get_sp500_tickers():
    """Scrape current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    tickers = tables[0]["Symbol"].tolist()
    # yfinance uses '-' instead of '.' for share classes (e.g., BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def download_prices(tickers, start, end):
    """Download OHLCV for all tickers. Returns a dict of {ticker: DataFrame}."""
    print(f"Downloading {len(tickers)} tickers from yfinance...")
    # batched download is much faster than one-at-a-time
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    out = {}
    for t in tickers:
        try:
            df = data[t].dropna()
            if len(df) > 200:  # need enough history for features
                out[t] = df
        except KeyError:
            continue
    print(f"Got usable data for {len(out)} / {len(tickers)} tickers.")
    return out


# -------- feature engineering --------
def build_features_for_ticker(df, ticker):
    """Build features + multi-horizon targets for one ticker.

    Produces one row per (trade_day, horizon) pair, so the same model can
    predict at any horizon.
    """
    df = df.copy()
    # drop any rows where Close or Volume is missing or non-positive --
    # yfinance occasionally returns 0 or NaN for halted/delisted periods,
    # and log(0) = -inf would poison all downstream features.
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)].copy()
    if len(df) < 100:
        raise ValueError(f"not enough clean rows ({len(df)})")

    df["log_close"] = np.log(df["Close"])
    df["ret_1d"] = df["log_close"].diff()

    # lagged returns (features are always backward-looking to avoid leakage)
    df["ret_1"] = df["ret_1d"]
    df["ret_5"] = df["log_close"].diff(5)
    df["ret_10"] = df["log_close"].diff(10)
    df["ret_21"] = df["log_close"].diff(21)

    # rolling stats on daily returns
    for w in (5, 21, 63):
        df[f"mean_ret_{w}"] = df["ret_1d"].rolling(w).mean()
        df[f"std_ret_{w}"] = df["ret_1d"].rolling(w).std()

    # volume -- use log1p so zero-volume days don't produce -inf
    log_vol = np.log1p(df["Volume"])
    df["vol_change_1"] = log_vol.diff()
    df["vol_change_5"] = log_vol.diff(5)

    # seasonality
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # ---- build per-horizon rows ----
    rows = []
    for h in HORIZONS:
        sub = df.copy()
        sub["days_ahead"] = h
        # target: log return h days in the future
        sub[TARGET_COL] = sub["log_close"].shift(-h) - sub["log_close"]
        sub["ticker"] = ticker
        rows.append(sub)

    out = pd.concat(rows)
    # drop any row with NaN features or target
    keep_cols = FEATURE_COLS + [TARGET_COL, "ticker", "Close"]
    out = out[keep_cols]
    # replace any lingering inf with NaN then drop -- belt and suspenders
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def build_full_dataset(price_dict):
    frames = []
    for ticker, df in tqdm(price_dict.items(), desc="Building features"):
        try:
            frames.append(build_features_for_ticker(df, ticker))
        except Exception as e:
            print(f"  skipping {ticker}: {e}")
    full = pd.concat(frames).sort_index()
    # final safety net: belt + suspenders + safety pins. strip any row
    # that has a non-finite value in any feature or target column.
    n_before = len(full)
    numeric_cols = FEATURE_COLS + [TARGET_COL]
    mask = np.isfinite(full[numeric_cols].to_numpy()).all(axis=1)
    full = full[mask]
    n_dropped = n_before - len(full)
    if n_dropped:
        print(f"  dropped {n_dropped:,} rows with inf/nan values")
    return full


# -------- training --------
def train(df):
    # chronological split — DO NOT shuffle financial data
    split = pd.Timestamp(TEST_SPLIT_DATE)
    train_df = df[df.index < split]
    test_df = df[df.index >= split]
    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        tree_method="hist",
        early_stopping_rounds=20,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE (log return): {mae:.4f}")
    print(f"Test R^2:              {r2:.4f}")

    # also report per-horizon MAE — long horizons will be much worse
    print("\nPer-horizon test MAE:")
    for h in HORIZONS:
        mask = test_df["days_ahead"] == h
        if mask.any():
            h_mae = mean_absolute_error(y_test[mask], preds[mask])
            print(f"  {h:>3}d ahead: MAE = {h_mae:.4f}")

    metrics = {"test_mae": float(mae), "test_r2": float(r2)}
    return model, metrics


# -------- saving --------
def save_artifacts(model, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "price_model.json"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE or "today",
        "test_split_date": TEST_SPLIT_DATE,
        "max_horizon": MAX_HORIZON,
        "horizons_trained": HORIZONS,
        "metrics": metrics,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved model + metadata to ./{MODEL_DIR}/")


# -------- main --------
def main():
    print("=" * 60)
    print("  Training price model (XGBoost) on S&P 500")
    print("=" * 60)
    tickers = get_sp500_tickers()
    price_dict = download_prices(tickers, START_DATE, END_DATE)
    dataset = build_full_dataset(price_dict)
    print(f"Final dataset: {len(dataset):,} rows")
    model, metrics = train(dataset)
    save_artifacts(model, metrics)


if __name__ == "__main__":
    main()
