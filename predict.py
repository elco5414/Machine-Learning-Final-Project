"""
predict.py

Loads the trained price model and predicts a stock's closing price on a
given future date.

Example:
    from predict import predict_price
    result = predict_price("AAPL", "2026-06-01")
    print(result)
    # {
    #   "ticker": "AAPL",
    #   "as_of": "2026-04-21",
    #   "target_date": "2026-06-01",
    #   "days_ahead": 29,
    #   "current_price": 172.43,
    #   "predicted_price": 178.12,
    #   "predicted_return_pct": 3.30,
    # }
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

MODEL_DIR = "models"
MAX_HORIZON = 90


# -------- model loading (cached at module level) --------
_model = None
_feature_cols = None


def _load_model():
    global _model, _feature_cols
    if _model is not None:
        return _model, _feature_cols

    model_path = os.path.join(MODEL_DIR, "price_model.json")
    features_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run `python price_model.py` first."
        )

    _model = xgb.XGBRegressor()
    _model.load_model(model_path)
    with open(features_path) as f:
        _feature_cols = json.load(f)
    return _model, _feature_cols


# -------- feature construction for a single prediction --------
def _build_features_for_prediction(ticker, days_ahead):
    """Fetch recent price history for `ticker` and build the latest feature row.

    Mirrors the feature logic from price_model.py exactly. Any divergence
    between training features and prediction features silently corrupts
    results, so changes must be made in both places.
    """
    # need ~1 year of history to compute the longest rolling window (63d)
    # and some buffer for weekends/holidays
    df = yf.download(
        ticker,
        period="1y",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No price data returned for ticker {ticker!r}.")

    # yfinance sometimes returns a multiindex column frame; flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    # same cleanup as training: drop any non-positive Close/Volume rows
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
    df["log_close"] = np.log(df["Close"])
    df["ret_1d"] = df["log_close"].diff()

    df["ret_1"] = df["ret_1d"]
    df["ret_5"] = df["log_close"].diff(5)
    df["ret_10"] = df["log_close"].diff(10)
    df["ret_21"] = df["log_close"].diff(21)

    for w in (5, 21, 63):
        df[f"mean_ret_{w}"] = df["ret_1d"].rolling(w).mean()
        df[f"std_ret_{w}"] = df["ret_1d"].rolling(w).std()

    log_vol = np.log1p(df["Volume"])
    df["vol_change_1"] = log_vol.diff()
    df["vol_change_5"] = log_vol.diff(5)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["days_ahead"] = days_ahead

    latest = df.dropna().iloc[-1]
    return latest


def _business_days_between(start, end):
    """Count trading days (Mon-Fri, ignoring holidays for simplicity)
    between two dates. Good enough for horizon calculation."""
    return int(np.busday_count(start.date(), end.date()))


# -------- public API --------
def predict_price(ticker, future_date):
    """Predict the closing price of `ticker` on `future_date`.

    Args:
        ticker: Stock symbol, e.g. "AAPL". Use "-" not "." (e.g. "BRK-B").
        future_date: str ("YYYY-MM-DD") or datetime/date.

    Returns:
        dict with keys: ticker, as_of, target_date, days_ahead,
        current_price, predicted_price, predicted_return_pct.

    Raises:
        ValueError: if the target date is in the past or beyond MAX_HORIZON
            trading days, or if yfinance returns no data.
    """
    model, feature_cols = _load_model()

    # normalize future_date
    if isinstance(future_date, str):
        target = pd.Timestamp(future_date)
    else:
        target = pd.Timestamp(future_date)

    today = pd.Timestamp(datetime.now().date())
    if target <= today:
        raise ValueError(
            f"future_date must be after today ({today.date()}); got {target.date()}."
        )

    days_ahead = _business_days_between(today, target)
    if days_ahead < 1:
        raise ValueError("future_date must be at least one trading day ahead.")
    if days_ahead > MAX_HORIZON:
        raise ValueError(
            f"Max supported horizon is {MAX_HORIZON} trading days "
            f"(~{MAX_HORIZON * 7 // 5} calendar days). "
            f"Requested horizon: {days_ahead} trading days."
        )

    feats = _build_features_for_prediction(ticker, days_ahead)
    X = feats[feature_cols].values.reshape(1, -1)
    predicted_log_return = float(model.predict(X)[0])

    current_price = float(feats["Close"])
    predicted_price = current_price * float(np.exp(predicted_log_return))
    predicted_return_pct = (np.exp(predicted_log_return) - 1) * 100

    return {
        "ticker": ticker.upper(),
        "as_of": str(today.date()),
        "target_date": str(target.date()),
        "days_ahead": days_ahead,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_return_pct": round(float(predicted_return_pct), 2),
    }


if __name__ == "__main__":
    # quick demo
    import pprint

    demo_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    pprint.pp(predict_price("AAPL", demo_date))
