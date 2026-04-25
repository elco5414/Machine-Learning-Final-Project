"""
Stock Market Prediction api
Serves predictions from LSTM models with multiple prediction horizons:
  - 5-day model (1 week ahead)
  - 20-day model (1 month ahead)  
  - 60-day model (1 quarter ahead)

Install dependencies:
    pip install fastapi uvicorn yfinance numpy torch pandas xgboost

Run:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000 in your browser.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
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

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS = {
    "5": {
        "path": "models/model_5.pth",
        "scaler": "models/scaler_5.npy",
        "seq_len": 30,
        "name": "5-day",
        "days": 5
    },
    "20": {
        "path": "models/model_20.pth",
        "scaler": "models/scaler_20.npy", 
        "seq_len": 60,
        "name": "20-day",
        "days": 20
    },
    "60": {
        "path": "models/model_60.pth",
        "scaler": "models/scaler_60.npy",  
        "seq_len": 120,
        "name": "60-day",
        "days": 60
    },
}

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "volume_change", "volume_ratio", "action",
]

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "PEP", "KO", "BAC", "PFE", "AVGO", "COST",
    "DIS", "CSCO", "TMO", "ACN", "MCD", "NEE", "NKE", "INTC",
    "QCOM", "UNH", "LIN", "DHR", "TXN", "PM", "AMGN", "RTX",
]



class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden = self.dropout(hidden[-1])
        return self.output_head(last_hidden).squeeze(-1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}
loaded_scalers = {}

for horizon, config in MODELS.items():
    try:
        checkpoint = torch.load(config["path"], map_location=device)
        model_config = checkpoint["config"]

        model = StockPredictor(
            model_config["input_size"], model_config["hidden_size"],
            model_config["num_layers"], model_config["dropout"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        loaded_models[horizon] = model

        scaler_data = np.load(config["scaler"], allow_pickle=True).item()
        loaded_scalers[horizon] = {
            "mean": scaler_data["mean"],
            "scale": scaler_data["scale"]
        }
    except FileNotFoundError:
        print(f"✗ {config['name']} model not found at {config['path']}")


_cached_tickers: Optional[list[str]] = None


def get_sp500_tickers() -> list[str]:
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



def fetch_recent_data(ticker: str, days: int = 200) -> Optional[pd.DataFrame]:
    try:
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
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
        df["bb_position"] = (close - df["bb_lower"]) / band_range.replace(0, float("nan"))

    df["volume_change"] = volume.pct_change(1)
    df["volume_ratio"] = volume / volume.rolling(5).mean()
    df["action"] = 0
    return df


def scale_features(features: np.ndarray, horizon: str) -> np.ndarray:
    scaler = loaded_scalers[horizon]
    return (features - scaler["mean"]) / scaler["scale"]


def predict_return(ticker: str, horizon: str = "5", action: int = 0) -> Optional[dict]:
    """Predict return using specified model horizon."""
    if horizon not in loaded_models:
        return None

    config = MODELS[horizon]
    model = loaded_models[horizon]
    seq_len = config["seq_len"]

    df = fetch_recent_data(ticker, days=300)  # fetch more for longer sequences
    if df is None or len(df) < seq_len:
        return None

    current_price = float(df["Close"].iloc[-1])

    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < seq_len:
        return None

    features = df[FEATURE_COLS].values[-seq_len:].astype(np.float32)
    features[:, -1] = action
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = scale_features(features, horizon)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).item()

    return {
        "current_price": current_price,
        "predicted_return_pct": round(pred * 100, 2),
        "horizon_days": config["days"]
    }


def get_recommendation(predicted_return: float) -> str:
    if predicted_return > 3:   return "Strong Buy"
    elif predicted_return > 1: return "Buy"
    elif predicted_return > -1: return "Hold"
    elif predicted_return > -3: return "Sell"
    else:                       return "Strong Sell"


app = FastAPI(
    title="Stock Market Predictor API",
    description="LSTM predictions on S&P 500 stocks with 5/20/60-day horizons",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class PortfolioRequest(BaseModel):
    portfolio: dict[str, int]
    horizon: Optional[str] = "5"  # 5, 20, or 60


class PricePredictionRequest(BaseModel):
    ticker: str
    target_date: str



@app.get("/tickers")
def tickers():
    return {"tickers": get_sp500_tickers()}


@app.post("/portfolio")
def analyze_portfolio(request: PortfolioRequest):
    if not request.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio cannot be empty")

    horizon = request.horizon or "5"
    if horizon not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model for {horizon}-day horizon not available")

    results = {}
    total_value = 0.0
    weighted_return_sum = 0.0
    successful_value = 0.0

    for ticker, shares in request.portfolio.items():
        ticker = ticker.upper()
        pred = predict_return(ticker, horizon=horizon, action=1)

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
            "horizon_days": MODELS[horizon]["days"],
        },
        "analyzed": datetime.now().isoformat(),
    }


@app.post("/predict-price")
def predict_price_endpoint(request: PricePredictionRequest):
    runner_path = os.path.join(STATIC_DIR, "predict_runner.py")
    try:
        completed = subprocess.run(
            [sys.executable, runner_path, request.ticker, request.target_date],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise HTTPException(
            status_code=500,
            detail=f"Prediction subprocess failed: {completed.stderr or 'no output'}",
        )

    if "error" in payload:
        kind = payload.get("kind", "other")
        status = 400 if kind == "value" else 500
        raise HTTPException(status_code=status, detail=payload["error"])

    return payload


@app.get("/suggestions")
def get_suggestions(limit: int = 6, horizon: str = "5"):
    """Scan the watchlist and return top performers for selected horizon."""
    if horizon not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model for {horizon}-day horizon not available")

    scored = []
    for ticker in WATCHLIST:
        pred = predict_return(ticker, horizon=horizon, action=1)
        if pred is not None:
            pct = pred["predicted_return_pct"]
            scored.append({
                "ticker": ticker,
                "predicted_return": f"{pct:+.2f}%",
                "recommendation": get_recommendation(pct),
                "pred_value": pct
            })

    scored.sort(key=lambda x: x["pred_value"], reverse=True)
    for s in scored: s.pop("pred_value")

    return {
        "suggestions": scored[:limit],
        "horizon_days": MODELS[horizon]["days"],
        "generated": datetime.now().isoformat()
    }


@app.get("/")
def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)