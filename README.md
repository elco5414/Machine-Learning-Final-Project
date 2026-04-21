# Machine-Learning-Final-Project
**Group Members:**
- Mariana Vadas-Arendt
- Elizabeth Coleman

**Premise**
This model helps lower the financial barrier for users to trading and investment. Many people do not have the capital to risk whilst learning the ebs and flows of trading which this will help resolve. This project allows users to experiement without assuming the same kinds of risk. 

## How to Use
general should work:  
- `python3.12 -m venv venv`
- `source venv/bin/activate`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`
- `python data_pipeline.py`

very specific commands i had to use unfortunately
- `/opt/homebrew/bin/python3.12 -m venv venv`
- `source venv/bin/activate`
- `./venv/bin/python -m pip install --upgrade pip`
- `./venv/bin/python -m pip install -r requirements.txt`
- `./venv/bin/python data_pipeline.py`

---
title: Market Terminal
emoji: 📈
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Stock Market Prediction on the S&P 500

A final project exploring two complementary approaches to forecasting U.S. equities, served through a single web interface.

## Premise

This project lowers the financial barrier between new investors and the mechanics of equity markets. A newcomer without capital to risk can still develop intuition for how positions behave, what a recommendation system is actually doing under the hood, and how forecast accuracy degrades with time horizon. The interface surfaces two questions a first-time investor typically has:

1. *Given what I hold, should I buy, sell, or hold each position?*
2. *If I bought shares of X today, what might they be worth in N days?*

These are answered by two separately-trained models, each deliberately chosen to illustrate a different class of approach to financial time series prediction.

## Methodology

### Data

Daily OHLCV (open, high, low, close, volume) data for the current S&P 500 constituents, pulled via `yfinance`. Tickers are scraped from Wikipedia's list of S&P 500 companies so the membership stays current. Training window: 2015-01-01 through the most recent close. Prices are split- and dividend-adjusted (`auto_adjust=True`) so that the label — realized forward return — is not contaminated by corporate actions.

Quality controls applied per ticker before feature engineering:

- Tickers with fewer than 200 trading days of history are dropped (insufficient for the longest rolling window).
- Rows with non-positive `Close` or `Volume` are dropped before any log transform (yfinance occasionally reports `0` for halted or delisted periods, which would produce `-inf` in downstream features).
- After feature construction, any row containing a non-finite value in any feature or target column is dropped.

### Model I — LSTM (short-horizon directional signal)

A stacked LSTM regresses the next-5-trading-day percentage return from a 30-day window of engineered features. Trained with PyTorch.

**Features per day (17 total):** lagged returns (1, 5, 10, 20 days), realized volatility (10, 20 days), RSI(14), MACD (line / signal / histogram), Bollinger bands (upper / middle / lower / position), normalized volume change and volume ratio, and a categorical action token reserved for inference-time what-if queries.

**Architecture:** two-layer LSTM, 128-unit hidden state, 0.2 dropout, followed by a two-layer MLP head (128 → 64 → 1). MSE loss, Adam optimizer at 1e-3 with ReduceLROnPlateau, gradient clipping at 1.0 to stabilize the recurrent gradients. Early stopping on validation loss with patience 5.

**Output:** a scalar 5-day percentage return, converted at the API layer to a discrete recommendation (Strong Buy / Buy / Hold / Sell / Strong Sell) using fixed thresholds of ±1% and ±3%.

### Model II — XGBoost (arbitrary-horizon price estimation)

A single gradient-boosted regressor predicts log returns at any horizon from 1 to 90 trading days. The horizon `days_ahead` is itself a feature, so one model serves the full 1–90 day range rather than training 90 separate models.

**Why log returns rather than raw prices.** Raw stock prices are non-stationary: a model trained on them will, to first order, learn "predict yesterday's close." Log returns are approximately stationary and roughly symmetric around zero, which is the regime tree-based learners handle best. At inference the predicted log return is converted back to a price with `predicted_price = current_price × exp(predicted_log_return)`.

**Features (15 total):** lagged log returns at horizons 1, 5, 10, 21 days; rolling mean and standard deviation of daily log returns at 5, 21, 63 day windows; log-volume change at 1 and 5 days (using `log1p` to avoid the zero-volume trap); day-of-week and month for light seasonality; and `days_ahead` itself.

**Training setup.** XGBoost regressor, 500 trees, depth 6, learning rate 0.05, 80% subsampling on both rows and columns, histogram tree method. Chronological 80/20 split on 2024-01-01: the entire test set is strictly after the training data. Random shuffling is never used — it would leak future information into training and produce fake-good metrics.

**Pooling across tickers.** One model is trained on all S&P 500 stocks rather than per-ticker. The rationale is that the dominant patterns — mean reversion, volatility clustering, momentum — are shared across large-cap equities, so the model benefits from seeing ~500× more data. The tradeoff is that it cannot learn stock-specific quirks.

## Results

Metrics are reported on the out-of-sample test set (post-2024-01-01 for the XGBoost model).

**XGBoost, by horizon:**

| Horizon (trading days) | Test MAE (log return) | Approx. % price error |
|---:|---:|---:|
| 1 | 0.0142 | ≈ 1.4% |
| 5 | 0.0322 | ≈ 3.2% |
| 10 | 0.0461 | ≈ 4.7% |
| 21 | 0.0676 | ≈ 7.0% |
| 42 | 0.0961 | ≈ 10.1% |
| 63 | 0.1176 | ≈ 12.5% |
| 90 | 0.1392 | ≈ 14.9% |

Overall test R² is 0.026. This is a meaningful positive signal — in financial ML, R² values in the 0.01–0.05 range on out-of-sample return prediction are considered legitimate, and anything substantially higher generally indicates data leakage. Errors scale roughly with the square root of horizon, consistent with the random-walk-plus-small-predictable-signal view of markets.

**LSTM.** MSE loss on the held-out 10% test split, reported in the training log. Evaluation is primarily qualitative via the recommendation interface, since the model is used as a classifier surrogate rather than a point estimator.

**Honest limitations.** Neither model is suitable for actual trading. Transaction costs, slippage, bid-ask spreads, and tax effects are not modeled. Accuracy at the 60- and 90-day horizons is comparable to guessing "no change," since unpredictable noise dominates the signal at long horizons. The pooled-ticker training approach means the model cannot distinguish sector-specific dynamics.

## System architecture

```
┌────────────────┐        ┌─────────────────────────┐
│  index.html    │ HTTP   │       api.py            │
│  (frontend)    │◄──────►│  FastAPI, loads LSTM    │
└────────────────┘        │  + scaler at startup    │
                          │                         │
                          │   /predict-price  ──────┼──► predict_runner.py
                          │   /portfolio            │    (fresh subprocess,
                          │   /tickers              │     loads XGBoost only)
                          │   /health               │
                          └─────────────────────────┘
```

The XGBoost model is invoked via subprocess rather than in-process. Both PyTorch and XGBoost ship their own copies of `libomp.dylib` on macOS, and loading both into one Python interpreter causes an OpenMP thread-pool collision that segfaults the process. Running XGBoost in its own short-lived subprocess is the cleanest workaround — the two libraries never share an address space.

## Repository layout

```
.
├── data_pipeline.py      # downloads OHLCV, builds features, writes Parquet
├── train_model.py        # trains the LSTM; writes models/model.pth
├── price_model.py        # trains the XGBoost model; writes models/price_model.json
├── predict.py            # XGBoost inference helper (imports xgboost only)
├── predict_runner.py     # subprocess wrapper around predict.py
├── api.py                # FastAPI app; loads LSTM + serves frontend
├── index.html            # single-page frontend
├── requirements.txt
├── data/                 # processed Parquet files (generated)
└── models/               # trained model artifacts (generated)
```

## Setup and usage

### Prerequisites

- **Python 3.10, 3.11, 3.12, or 3.13.** Python 3.14 is not yet supported: `pandas-ta` pins `numba==0.61.2`, which does not support 3.14 as of this writing.
- On macOS, OpenMP must be installed for XGBoost: `brew install libomp`.

### Installation

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*If your shell has an alias or path quirk that causes `python` or `pip` to resolve outside the venv (as on the machine this was developed on), use the venv executables explicitly:*

```bash
/opt/homebrew/bin/python3.12 -m venv venv      # use an absolute path to avoid alias resolution
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

### Reproducing the pipeline

Run in order. Each step takes 1–5 minutes depending on network and hardware.

```bash
# 1. download and process data -> writes data/training_data.parquet
python data_pipeline.py

# 2. train the LSTM -> writes models/model.pth and models/scaler.npy
python train_model.py

# 3. train the XGBoost model -> writes models/price_model.json
python price_model.py
```

### Running the web interface

```bash
python -m uvicorn api:app --port 8000
```

Then open `http://localhost:8000` in a browser. The left panel uses the LSTM model to analyze a user-defined portfolio over a 5-day horizon; the right panel uses the XGBoost model to project a single stock's closing price on a chosen future date.

## Academic notes

- **Chronological splitting.** The most important methodological choice in financial ML: splits must be time-ordered, never random. Random splitting lets the model see future patterns at training time and inflates metrics dramatically.
- **Feature-target alignment.** All features are backward-looking at the prediction date. The target is the realized return from day *t* to day *t+h*, computed as a forward-shift on the price series. Any leakage between these would invalidate the results.
- **Parity between training and inference features.** Feature construction is duplicated in `price_model.py` (training) and `predict.py` (inference). Any divergence between these — even a subtle one like a different window size on a rolling statistic — silently corrupts predictions. Both paths use the same `log1p` on volume, the same zero-value filtering on `Close`, and the same feature-column ordering pulled from a JSON manifest.
- **Not investment advice.** All code and outputs are for academic coursework only.
