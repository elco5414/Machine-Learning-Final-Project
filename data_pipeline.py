"""
Fetches S&P 500 OHLCV data, gets features (RSI, MACD, volume, rolling returns), and labels
each data point with the actual % return after N days.
"""

import io
import os
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta  # helps with features calculation
import yfinance as yf  # a lot of finance data
from tqdm import tqdm


PREDICTION_HORIZON = 5    # how many days into future it predicts, will need multiple models to have multiple timeframe predictions 
START_DATE         = "2015-01-01"
END_DATE           = "2024-12-31"
OUTPUT_DIR         = "data"
PROCESSED_DIR      = os.path.join(OUTPUT_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# fetch S&P 500 companies
def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # Parse only the first table (the S&P 500 constituent table)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    
    # Make sure we got the right table
    if "Symbol" not in df.columns:
        raise ValueError(f"Unexpected table columns: {df.columns.tolist()}")
    
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    print(f"  Found {len(tickers)} tickers.\n")
    return tickers


# download OHLCV data
def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download OHLCV data for a single company."""
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,   # adjusts for splits & dividends
            progress=False
        )
        if df.empty or len(df) < 100:
            return None         # skip companies with insufficient data
        
        if isinstance(df.columns, pd.MultiIndex): # yfinance was returning ('Close', 'MMM') and we just want 'Close'
            df.columns = [col[0] for col in df.columns]

        df["Ticker"] = ticker
        return df
    except Exception as e:
        print(f"  Warning: failed to download {ticker} — {e}")
        return None


def download_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers."""
    print("Downloading OHLCV data for all tickers...")
    data = {}
    for ticker in tqdm(tickers):
        df = download_ticker(ticker)
        if df is not None:
            data[ticker] = df
    print(f"  Successfully downloaded {len(data)} tickers.\n")
    return data


# get features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as model input features.

    Features added:
        - Returns: 1d, 5d, 10d, 20d rolling returns
        - Volatility: 10d and 20d rolling std of returns
        - RSI: 14-period Relative Strength Index
        - MACD: MACD line, signal line, histogram
        - Bollinger Bands: upper, mid, lower
        - Volume change: % change in volume vs 5d avg
    """
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    # --- Returns ---
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # --- Volatility ---
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # --- RSI ---
    df["rsi_14"] = ta.rsi(close, length=14)

    # --- MACD ---
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]        = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"]   = macd["MACDh_12_26_9"]

    # --- Bollinger Bands ---
    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        # Dynamically find column names regardless of pandas_ta version
        upper_col = [c for c in bbands.columns if c.startswith("BBU")][0]
        mid_col   = [c for c in bbands.columns if c.startswith("BBM")][0]
        lower_col = [c for c in bbands.columns if c.startswith("BBL")][0]

        df["bb_upper"] = bbands[upper_col]
        df["bb_mid"]   = bbands[mid_col]
        df["bb_lower"] = bbands[lower_col]
        df["bb_position"] = (close - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

    # --- Volume ---
    df["volume_change"] = volume.pct_change(1)
    df["volume_ratio"]  = volume / volume.rolling(5).mean()

    return df


# label data
def label_data(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Label each row with the actual % return N days into the future.
    """
    df = df.copy()
    # close is the closing price, pct_change is the % change, the - makes it move backward so that this is the future of past 5 days ago
    df["label"] = df["Close"].pct_change(horizon).shift(-horizon)

    df["action"] = 0  # 0=hold, 1=buy, 2=sell

    return df


# clean up data
FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "volume_change", "volume_ratio",
    "action"
]

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows and clip extreme outliers."""
    df = df.copy()

    # Drop rows where label or any feature is NaN
    cols_needed = FEATURE_COLS + ["label"]
    df = df.dropna(subset=cols_needed)

    # Clip extreme outliers (beyond 3 std) to reduce noise
    for col in FEATURE_COLS:
        if col == "action":
            continue
        mean = df[col].mean()
        std  = df[col].std()
        df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

    return df


# save processed data to local machine, using parquet bc it is smaller and faster
def process_and_save(ticker: str, raw_df: pd.DataFrame) -> bool:
    """Run the full pipeline for one ticker and save to Parquet."""
    try:
        df = engineer_features(raw_df)
        df = label_data(df)
        df = clean_and_normalize(df)

        if len(df) < 50:
            return False  # Not enough data after cleaning

        out_path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
        df[FEATURE_COLS + ["label", "Ticker"]].to_parquet(out_path)
        return True
    except Exception as e:
        print(f"  Warning: failed to process {ticker} — {e}")
        return False


# combine all data into one for training
def combine_all(processed_dir: str) -> pd.DataFrame:
    """Combine all per-ticker Parquet files into one master DataFrame."""
    print("\nCombining all processed files...")
    frames = []
    for fname in os.listdir(processed_dir):
        if fname.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(processed_dir, fname)))

    if not frames:
        raise ValueError("No processed files found.")

    combined = pd.concat(frames, ignore_index=True)
    out_path  = os.path.join(OUTPUT_DIR, "training_data.parquet")
    combined.to_parquet(out_path)
    print(f"  Combined dataset: {len(combined):,} rows")
    print(f"  Saved to: {out_path}\n")
    return combined


# main program
def run_pipeline():
    print("=" * 50)
    print("  Stock Market Prediction — Data Pipeline")
    print("=" * 50, "\n")

    tickers = get_sp500_tickers()

    raw_data = download_all_tickers(tickers)

    print("Processing tickers (features + labels + cleaning)...")
    success, failed = 0, 0
    for ticker, df in tqdm(raw_data.items()):
        if process_and_save(ticker, df):
            success += 1
        else:
            failed += 1
    print(f"  Processed: {success} tickers | Skipped: {failed} tickers\n")

    training_data = combine_all(PROCESSED_DIR)

    print("Pipeline complete! Summary:")
    print(f"  Total rows:     {len(training_data):,}")
    print(f"  Total tickers:  {training_data['Ticker'].nunique()}")
    print(f"  Features:       {len(FEATURE_COLS)}")
    print(f"  Label:          % return over {PREDICTION_HORIZON} days")
    print(f"  Output:         {OUTPUT_DIR}/training_data.parquet")


if __name__ == "__main__":
    run_pipeline()