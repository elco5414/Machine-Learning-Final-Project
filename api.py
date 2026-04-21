"""
Stock Market Prediction - FastAPI Backend
==========================================
Serves predictions from two models:
  1. LSTM model (5-day return + BUY/SELL/HOLD recommendations)
  2. XGBoost model (absolute price on a specific future date, up to 90 days)

Also serves the static frontend at the root URL.

Install dependencies:
    pip install fastapi uvicorn yfinance numpy torch pandas xgboost

Run:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000 in your browser.
"""

# ─────────────────────────────────────────────
# OPENMP WORKAROUND -- MUST BE FIRST
# ─────────────────────────────────────────────
# torch and xgboost each ship their own libomp.dylib on Mac.
# Loading both in one process triggers an OpenMP duplicate-lib
# abort that crashes the Python interpreter (segfault at
# __kmp_suspend_initialize_thread). This env var tells OpenMP
# to tolerate multiple copies. Must be set BEFORE torch/xgboost
# are imported, hence the placement above all other imports.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io

# our XGBoost price model
# our XGBoost price model is invoked as a subprocess to avoid the torch +
# xgboost OpenMP collision that crashes Python on Mac. MAX_HORIZON is still
# imported from predict.py so the API can validate date ranges without
# actually loading xgboost into this process.
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import torch
import torch.nn as nn
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from predict import MAX_HORIZON
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# Directory containing this file -- used to locate the frontend and the
# subprocess runner script.
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "models/model.pth"
SCALER_PATH = "models/scaler.npy"

SEQUENCE_LEN = 30

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

WATCHLIST = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "MA",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "BAC",
    "PFE",
    "AVGO",
    "COST",
    "DIS",
    "CSCO",
    "TMO",
    "ACN",
    "MCD",
    "NEE",
    "NKE",
    "INTC",
    "QCOM",
    "UNH",
    "LIN",
    "DHR",
    "TXN",
    "PM",
    "AMGN",
    "RTX",
]


# ─────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
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
        _, (hidden, _) = self.lstm(x)
        last_hidden = self.dropout(hidden[-1])
        out = self.output_head(last_hidden)
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# LOAD LSTM MODEL AT STARTUP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
config = checkpoint["config"]

lstm_model = StockPredictor(
    input_size=config["input_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
).to(device)
lstm_model.load_state_dict(checkpoint["model_state"])
lstm_model.eval()

scaler_data = np.load(SCALER_PATH, allow_pickle=True).item()
scaler_mean = scaler_data["mean"]
scaler_scale = scaler_data["scale"]

print(f"✓ LSTM model loaded from {MODEL_PATH}")
print(f"✓ Scaler loaded from {SCALER_PATH}")
print(f"✓ Running on {device}")


# ─────────────────────────────────────────────
# TICKER LIST (for dropdown)
# ─────────────────────────────────────────────
_cached_tickers: Optional[list[str]] = None


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia. Cached after first call."""
    global _cached_tickers
    if _cached_tickers is not None:
        return _cached_tickers
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(io.StringIO(response.text))
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        _cached_tickers = sorted(tickers)
    except Exception as e:
        print(f"Warning: could not fetch S&P 500 tickers ({e}); using watchlist")
        _cached_tickers = sorted(WATCHLIST)
    return _cached_tickers


# ─────────────────────────────────────────────
# LSTM HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fetch_recent_data(ticker: str, days: int = 200) -> Optional[pd.DataFrame]:
    try:
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < SEQUENCE_LEN:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception:
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        upper_col = [c for c in bbands.columns if c.startswith("BBU")][0]
        mid_col = [c for c in bbands.columns if c.startswith("BBM")][0]
        lower_col = [c for c in bbands.columns if c.startswith("BBL")][0]
        df["bb_upper"] = bbands[upper_col]
        df["bb_mid"] = bbands[mid_col]
        df["bb_lower"] = bbands[lower_col]
        band_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / band_range.replace(
            0, float("nan")
        )

    df["volume_change"] = volume.pct_change(1)
    df["volume_ratio"] = volume / volume.rolling(5).mean()
    df["action"] = 0
    return df


def scale_features(features: np.ndarray) -> np.ndarray:
    return (features - scaler_mean) / scaler_scale


def predict_return(ticker: str, action: int = 0) -> Optional[dict]:
    """Predict 5-day return + fetch latest close for portfolio impact math."""
    df = fetch_recent_data(ticker)
    if df is None:
        return None

    current_price = float(df["Close"].iloc[-1])

    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < SEQUENCE_LEN:
        return None

    features = df[FEATURE_COLS].values[-SEQUENCE_LEN:].astype(np.float32)
    features[:, -1] = action
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = scale_features(features)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = lstm_model(x).item()

    return {
        "current_price": current_price,
        "predicted_return_pct": round(pred * 100, 2),
    }


def get_recommendation(predicted_return: float) -> str:
    if predicted_return > 3:
        return "Strong Buy"
    elif predicted_return > 1:
        return "Buy"
    elif predicted_return > -1:
        return "Hold"
    elif predicted_return > -3:
        return "Sell"
    else:
        return "Strong Sell"


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Stock Market Predictor API",
    description="LSTM + XGBoost predictions on S&P 500 stocks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-origin deployment, safe to open up
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────
class PortfolioRequest(BaseModel):
    portfolio: dict[str, int]  # { "AAPL": 10, "TSLA": 5 }


class PricePredictionRequest(BaseModel):
    ticker: str
    target_date: str  # "YYYY-MM-DD"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "lstm_model": MODEL_PATH,
        "device": str(device),
        "max_price_horizon_days": MAX_HORIZON,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tickers")
def tickers():
    """Return the full S&P 500 ticker list for the frontend dropdown."""
    return {"tickers": get_sp500_tickers()}


@app.post("/portfolio")
def analyze_portfolio(request: PortfolioRequest):
    """Analyze each stock in the portfolio + compute total value and impact."""
    if not request.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio cannot be empty")

    results = {}
    total_value = 0.0
    weighted_return_sum = 0.0
    successful_value = 0.0

    for ticker, shares in request.portfolio.items():
        ticker = ticker.upper()
        pred = predict_return(ticker, action=1)

        if pred is None:
            results[ticker] = {
                "shares": shares,
                "current_price": None,
                "position_value": None,
                "predicted_return_pct": None,
                "predicted_dollar_change": None,
                "recommendation": "Unknown",
                "error": "Could not fetch data for this ticker",
            }
            continue

        position_value = pred["current_price"] * shares
        predicted_dollar_change = position_value * (pred["predicted_return_pct"] / 100)

        results[ticker] = {
            "shares": shares,
            "current_price": round(pred["current_price"], 2),
            "position_value": round(position_value, 2),
            "predicted_return_pct": pred["predicted_return_pct"],
            "predicted_dollar_change": round(predicted_dollar_change, 2),
            "recommendation": get_recommendation(pred["predicted_return_pct"]),
        }
        total_value += position_value
        successful_value += position_value
        weighted_return_sum += pred["predicted_return_pct"] * position_value

    # weighted average return across successfully-priced holdings
    portfolio_return = (
        round(weighted_return_sum / successful_value, 2)
        if successful_value > 0
        else None
    )
    projected_change = (
        round(successful_value * (portfolio_return / 100), 2)
        if portfolio_return is not None
        else None
    )

    return {
        "portfolio": results,
        "summary": {
            "total_value": round(total_value, 2),
            "weighted_return_pct": portfolio_return,
            "projected_dollar_change": projected_change,
            "horizon_days": 5,
        },
        "analyzed": datetime.now().isoformat(),
    }


@app.post("/predict-price")
def predict_price_endpoint(request: PricePredictionRequest):
    """Predict exact closing price on a specific future date (up to 90 trading days).

    Runs the XGBoost model in a fresh subprocess. This isolates xgboost's
    OpenMP runtime from torch's, which otherwise collide and segfault the
    Python interpreter on Mac.
    """
    runner_path = os.path.join(STATIC_DIR, "predict_runner.py")
    try:
        completed = subprocess.run(
            [sys.executable, runner_path, request.ticker, request.target_date],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    # predict_runner.py always prints JSON to stdout, even on error
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        # something went badly wrong -- surface stderr so we can debug
        raise HTTPException(
            status_code=500,
            detail=f"Prediction subprocess failed: {completed.stderr or 'no output'}",
        )

    if "error" in payload:
        kind = payload.get("kind", "other")
        status = 400 if kind == "value" else 500
        raise HTTPException(status_code=status, detail=payload["error"])

    return payload


# ─────────────────────────────────────────────
# SERVE STATIC FRONTEND
# ─────────────────────────────────────────────
# Assumes index.html sits next to this file. Served at the root URL.
# (STATIC_DIR is defined near the top of the file.)


@app.get("/")
def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)
