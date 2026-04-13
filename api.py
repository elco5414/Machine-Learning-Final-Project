"""
Stock Market Prediction - FastAPI Backend
==========================================
Serves predictions from the trained LSTM model.

Install dependencies:
    pip install fastapi uvicorn yfinance numpy torch pandas

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /portfolio       - Analyze a portfolio of stocks
    POST /predict-move    - Predict outcome of a specific move
    GET  /suggestions     - Get new company suggestions
    GET  /health          - Health check
"""

import os
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "models/model.pth"
SCALER_PATH = "models/scaler.npy"

SEQUENCE_LEN = 30   # Must match what the model was trained with

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "volume_change", "volume_ratio",
    "action"
]

# Tickers to scan for suggestions (subset of S&P 500)
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "PEP", "KO", "BAC", "PFE", "AVGO", "COST",
    "DIS", "CSCO", "TMO", "ACN", "MCD", "NEE", "NKE", "INTC",
    "QCOM", "UNH", "LIN", "DHR", "TXN", "PM", "AMGN", "RTX"
]


# ─────────────────────────────────────────────
# MODEL DEFINITION (must match train_model.py)
# ─────────────────────────────────────────────
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True
        )
        self.dropout    = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden     = hidden[-1]
        last_hidden     = self.dropout(last_hidden)
        out             = self.output_head(last_hidden)
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# LOAD MODEL & SCALER AT STARTUP
# ─────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
config     = checkpoint["config"]

model = StockPredictor(
    input_size  = config["input_size"],
    hidden_size = config["hidden_size"],
    num_layers  = config["num_layers"],
    dropout     = config["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

model.load_state_dict(checkpoint["model_state"])
model.eval()


# Load scaler parameters
scaler_data  = np.load(SCALER_PATH, allow_pickle=True).item()
scaler_mean  = scaler_data["mean"]
scaler_scale = scaler_data["scale"]

print(f"✓ Model loaded from {MODEL_PATH}")
print(f"✓ Scaler loaded from {SCALER_PATH}")
print(f"✓ Running on {device}")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fetch_recent_data(ticker: str, days: int = 200) -> pd.DataFrame | None:
    """Fetch recent OHLCV data for a ticker."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=days)
        df    = yf.download(ticker, start=start, end=end,
                            auto_adjust=True, progress=False)
        if df.empty or len(df) < SEQUENCE_LEN:
            return None

        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        return df
    except Exception:
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (same as data pipeline)."""
    df    = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    df["return_1d"]      = close.pct_change(1)
    df["return_5d"]      = close.pct_change(5)
    df["return_10d"]     = close.pct_change(10)
    df["return_20d"]     = close.pct_change(20)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["rsi_14"]         = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"]        = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"]   = macd["MACDh_12_26_9"]

    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        upper_col        = [c for c in bbands.columns if c.startswith("BBU")][0]
        mid_col          = [c for c in bbands.columns if c.startswith("BBM")][0]
        lower_col        = [c for c in bbands.columns if c.startswith("BBL")][0]
        df["bb_upper"]   = bbands[upper_col]
        df["bb_mid"]     = bbands[mid_col]
        df["bb_lower"]   = bbands[lower_col]
        band_range       = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / band_range.replace(0, float("nan"))

    df["volume_change"] = volume.pct_change(1)
    df["volume_ratio"]  = volume / volume.rolling(5).mean()
    df["action"]        = 0

    return df


def scale_features(features: np.ndarray) -> np.ndarray:
    """Apply the same scaling used during training."""
    return (features - scaler_mean) / scaler_scale


def predict_return(ticker: str, action: int = 0) -> float | None:
    df = fetch_recent_data(ticker)
    if df is None:
        print(f"  ✗ {ticker}: fetch_recent_data returned None")
        return None
    print(f"  ✓ {ticker}: fetched {len(df)} rows")

    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"  ✓ {ticker}: {len(df)} rows after feature engineering + dropna")

    if len(df) < SEQUENCE_LEN:
        print(f"  ✗ {ticker}: not enough rows ({len(df)} < {SEQUENCE_LEN})")
        return None
    

    # Take the last SEQUENCE_LEN rows as input
    features            = df[FEATURE_COLS].values[-SEQUENCE_LEN:].astype(np.float32)
    features[:, -1]     = action
    features            = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features            = scale_features(features)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).item()

    return round(pred * 100, 2)


def get_recommendation(predicted_return: float) -> str:
    """Convert predicted return into a simple recommendation."""
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
    title       = "Stock Market Predictor API",
    description = "Predicts stock returns using an LSTM model trained on S&P 500 data",
    version     = "1.0.0"
)

@app.on_event("startup")
async def startup_event():
    print("=" * 40)
    print("  API Ready!")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Device: {device}")
    print("=" * 40)

# Allow requests from your frontend (React dev server runs on 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class PortfolioRequest(BaseModel):
    portfolio: dict[str, int]   # { "AAPL": 10, "TSLA": 5 }

class MoveRequest(BaseModel):
    action: str                 # "BUY", "SELL", or "HOLD"
    ticker: str                 # e.g. "NVDA"
    shares: Optional[int] = 1
    portfolio: Optional[dict[str, int]] = None  # optionally include portfolio for impact calc


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    """Check the API is running."""
    return {
        "status":    "ok",
        "model":     MODEL_PATH,
        "device":    str(device),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/portfolio")
def analyze_portfolio(request: PortfolioRequest):
    """
    Analyze each stock in the portfolio.
    Returns predicted return + recommendation for each holding.
    """
    if not request.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio cannot be empty")

    results = {}
    for ticker, shares in request.portfolio.items():
        ticker = ticker.upper()
        pred   = predict_return(ticker, action=1)  # action=1 (buy/hold)

        if pred is None:
            results[ticker] = {
                "shares":             shares,
                "predicted_return":   None,
                "recommendation":     "Unknown",
                "error":              "Could not fetch data for this ticker"
            }
        else:
            results[ticker] = {
                "shares":           shares,
                "predicted_return": f"{pred:+.2f}%",
                "recommendation":   get_recommendation(pred)
            }

    return {
        "portfolio": results,
        "analyzed":  datetime.now().isoformat()
    }


@app.post("/predict-move")
def predict_move(request: MoveRequest):
    """
    Predict the outcome of a specific move.
    Optionally calculates portfolio impact if portfolio is provided.
    """
    ticker = request.ticker.upper()
    action_map = {"BUY": 1, "HOLD": 0, "SELL": 2}

    if request.action.upper() not in action_map:
        raise HTTPException(status_code=400, detail="Action must be BUY, SELL, or HOLD")

    action = action_map[request.action.upper()]
    pred   = predict_return(ticker, action=action)

    if pred is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}")

    response = {
        "ticker":             ticker,
        "action":             request.action.upper(),
        "shares":             request.shares,
        "predicted_return":   f"{pred:+.2f}%",
        "recommendation":     get_recommendation(pred),
    }

    # If portfolio provided, estimate overall portfolio impact
    if request.portfolio:
        total_shares  = sum(request.portfolio.values()) + request.shares
        impact        = (pred * request.shares) / total_shares
        response["portfolio_impact"] = f"{impact:+.2f}%"

    return response


@app.get("/suggestions")
def get_suggestions(limit: int = 5):
    """
    Scan the watchlist and return the top performing predicted stocks
    that could be good buys.
    """
    scored = []

    for ticker in WATCHLIST:
        pred = predict_return(ticker, action=1)
        if pred is not None:
            scored.append({
                "ticker":           ticker,
                "predicted_return": f"{pred:+.2f}%",
                "recommendation":   get_recommendation(pred),
                "pred_value":       pred  # used for sorting, not returned
            })

    # Sort by predicted return descending
    scored.sort(key=lambda x: x["pred_value"], reverse=True)

    # Remove internal sorting field before returning
    suggestions = []
    for s in scored[:limit]:
        s.pop("pred_value")
        suggestions.append(s)

    return {
        "suggestions": suggestions,
        "generated":   datetime.now().isoformat()
    }
