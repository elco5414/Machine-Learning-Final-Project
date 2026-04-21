# Machine-Learning-Final-Project
### Portfolio Predictor

**Group Members:**
- Mariana Vadas-Arendt
- Elizabeth Coleman

**Hugging Face Hosting for Application**

[Hosted Application Link](https://huggingface.co/spaces/elco5414/portfolio-prediction)

- the blurb below is needed for hugging face set-up just ignore. 
---
title: Market Terminal
emoji: 📊
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

## How to Use
**Dependencies**
- Python 3.10, 3.11, 3.12, or 3.13. **Python 3.14 is not yet supported**: `pandas-ta` pins `numba==0.61.2`, which does not support 3.14 as of this writing.
- On macOS, OpenMP must be installed for XGBoost: `brew install libomp`.
- for all specific package dependencies they are in `requirements.txt` and you can install them via `pip install -r requirements.txt`

**general installation**s
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*If your shell has an alias or path quirk that causes `python` or `pip` to resolve outside the venv (like me), use the venv executables explicitly, you have to use abs path*

```bash
/opt/homebrew/bin/python3.12 -m venv venv     
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```
If you had to use absolute path for installation - you have to continue to use that path through out also. 

Also if you are not familiar with virtual environments, you use `deactivate` to exit from it. 

**training**
- of course, if you are using the exact repo, you do not need to do these again, bc they have already been trained, so if you want to go nuts but its not a requirenment

```bash
# 1. download and process data -> writes data/training_data.parquet
python data_pipeline.py

# 2. train the LSTM -> writes models/model.pth and models/scaler.npy
python train_model.py

# 3. train the XGBoost model -> writes models/price_model.json
python price_model.py
```

**running web-app locally**
```bash
python -m uvicorn api:app --port 8000
```
Then open `http://localhost:8000` in a browser

If all these things are not calling to you - it is hosted on hugging face and you can see/use it there too (see link above).
****

#### What is it?
**Premise**
A final project exploring two complementary approaches to forecasting U.S. equities, served through a single web interface.

This project lowers the financial barrier between new investors and the mechanics of equity markets. A newcomer without capital to risk can still develop intuition for how positions behave, what a recommendation system is actually doing under the hood, and how forecast accuracy degrades with time horizon. The interface surfaces two questions a first-time investor typically has:

1. *Given what I hold, should I buy, sell, or hold each position? (trained to decide based on next 5 days)*
2. *If I bought shares of X today, what might they be worth in N days? (limited to 90 days)*

These are answered by two separately-trained models, each deliberately chosen to illustrate a different class of approach to financial time series prediction.

**Data**

Daily OHLCV (open, high, low, close, volume) data for the current S&P 500 constituents, pulled via `yfinance`. Tickers are scraped from Wikipedia's list of S&P 500 companies so the membership stays current. Training window: 2015-01-01 through the most recent close. Prices are split- and dividend-adjusted (`auto_adjust=True`) so that the label — realized forward return — is not contaminated by corporate actions.

Quality controls applied per ticker before feature engineering:

- Tickers with fewer than 200 trading days of history are dropped (insufficient for the longest rolling window).
- Rows with non-positive `Close` or `Volume` are dropped before any log transform (yfinance occasionally reports `0` for halted or delisted periods, which would produce `-inf` in downstream features).
- After feature construction, any row containing a non-finite value in any feature or target column is dropped.

##### Model I — Long Short Term Memory (LSTM) - short-horizon directional signal

A stacked LSTM regresses the next-5-trading-day percentage return from a 30-day window of engineered features. Trained with PyTorch.

**Features per day (17 total):** 
- lagged returns (1, 5, 10, 20 days)
- realized volatility (10, 20 days)
- RSI(14)
- MACD (line / signal / histogram)
- Bollinger bands (upper / middle / lower / position)
- normalized volume change and volume ratio
- categorical action token reserved for inference-time what-if queries.

**Architecture:** 
- two-layer Long Short Term Memory (LSTM)
- 128-unit hidden state, 0.2 dropout
- followed by a two-layer MLP head (128 → 64 → 1)
- MSE loss
- Adam optimizer at 1e-3 with ReduceLROnPlateau
- gradient clipping at 1.0 to stabilize the recurrent gradients
- Early stopping on validation loss with patience 5

**Output:** 
- a scalar 5-day percentage return converted at the API layer to a discrete recommendation: (Strong Buy / Buy / Hold / Sell / Strong Sell) using fixed thresholds of ±1% and ±3%

###### Model II — XGBoost (arbitrary-horizon price estimation)

A single gradient-boosted regressor predicts log returns at any horizon from 1 to 90 trading days. The horizon `days_ahead` is itself a feature, so one model serves the full 1–90 day range rather than training 90 separate models.

**Why log returns rather than raw prices.**
Raw stock prices are non-stationary: a model trained on them will, to first order, learn "predict yesterday's close." Log returns are approximately stationary and roughly symmetric around zero, which is the regime tree-based learners handle best. At inference the predicted log return is converted back to a price with `predicted_price = current_price × exp(predicted_log_return)`.

**Features (15 total):** 
- lagged log returns at horizons 1, 5, 10, 21 days
- rolling mean and standard deviation of daily log returns at 5, 21, 63 day windows
- log-volume change at 1 and 5 days (using `log1p` to avoid the zero-volume trap)
- day-of-week and month for light seasonality
- and `days_ahead` itself.

**Training setup.** 
XGBoost regressor, 500 trees, depth 6, learning rate 0.05, 80% subsampling on both rows and columns, histogram tree method. Chronological 80/20 split on 2024-01-01: the entire test set is strictly after the training data. Random shuffling is never used — it would leak future information into training and produce fake-good metrics.

**Pooling across tickers.** 
One model is trained on all S&P 500 stocks rather than per-ticker. The rationale is that the dominant patterns — mean reversion, volatility clustering, momentum — are shared across large-cap equities, so the model benefits from seeing ~500× more data. The tradeoff is that it cannot learn stock-specific quirks.

###### Results

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

**LSTM.** 
MSE loss on the held-out 10% test split, reported in the training log. Evaluation is primarily qualitative via the recommendation interface, since the model is used as a classifier surrogate rather than a point estimator.

**Honest limitations.** 
Neither model is suitable for actual trading. Transaction costs, slippage, bid-ask spreads, and tax effects are not modeled. Accuracy at the 60- and 90-day horizons is comparable to guessing "no change," since unpredictable noise dominates the signal at long horizons. The pooled-ticker training approach means the model cannot distinguish sector-specific dynamics.

**note**
- Do not use this for your actual investment advice, this is for academics
- AI was used for the ideation of these models and the generation of model code

****
