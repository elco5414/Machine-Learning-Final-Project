"""
evaluate_model.py

Loads the trained price model and generates diagnostic figures on the
held-out test set (everything on/after TEST_SPLIT_DATE).

Reuses data-loading and feature-building logic from price_model.py so the
test set is reconstructed identically to training.

Outputs (saved to ./figures/):
    1_per_horizon_mae.png        -- MAE by prediction horizon
    2_predicted_vs_actual.png    -- scatter of predicted vs actual log return
    3_residual_histogram.png     -- distribution of prediction errors
    4_residuals_over_time.png    -- errors across the test window (drift check)
    5_feature_importance.png     -- which features the model relies on

Usage:
    python evaluate_model.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# Reuse the exact same pipeline as training so the test set is identical
from price_model import (
    END_DATE,
    FEATURE_COLS,
    HORIZONS,
    START_DATE,
    TARGET_COL,
    TEST_SPLIT_DATE,
    build_full_dataset,
    download_prices,
    get_sp500_tickers,
)
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_DIR = "models"
FIG_DIR = "figures"


def load_model():
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "price_model.json"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        cols = json.load(f)
    # attach feature names so importance plot shows real names, not f0/f1/...
    model.get_booster().feature_names = cols
    return model, cols


def rebuild_test_set():
    """Rebuild the full dataset and slice out the test portion."""
    print("Rebuilding test set (this downloads S&P 500 prices)...")
    tickers = get_sp500_tickers()
    price_dict = download_prices(tickers, START_DATE, END_DATE)
    dataset = build_full_dataset(price_dict)
    split = pd.Timestamp(TEST_SPLIT_DATE)
    test_df = dataset[dataset.index >= split].copy()
    print(f"Test rows: {len(test_df):,}")
    return test_df


# -------- figure 1: per-horizon MAE --------
def plot_per_horizon_mae(test_df, preds, y_test):
    maes = []
    for h in HORIZONS:
        mask = test_df["days_ahead"] == h
        if mask.any():
            maes.append(mean_absolute_error(y_test[mask], preds[mask]))
        else:
            maes.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [str(h) for h in HORIZONS], maes, color="steelblue", edgecolor="black"
    )
    ax.set_xlabel("Prediction horizon (trading days ahead)")
    ax.set_ylabel("Mean Absolute Error (log return)")
    ax.set_title(
        "Test MAE by prediction horizon\n(longer horizons are harder to predict)"
    )
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mae:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "1_per_horizon_mae.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# -------- figure 2: predicted vs actual --------
def plot_predicted_vs_actual(test_df, preds, y_test):
    fig, ax = plt.subplots(figsize=(9, 9))

    # color by horizon so we can see how the spread grows with horizon
    horizons = test_df["days_ahead"].values
    scatter = ax.scatter(
        y_test,
        preds,
        c=horizons,
        cmap="viridis",
        alpha=0.2,
        s=6,
    )

    # y = x reference line (perfect predictions)
    lim = max(abs(y_test.min()), abs(y_test.max()), abs(preds.min()), abs(preds.max()))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="perfect (y = x)")

    ax.set_xlabel("Actual log return")
    ax.set_ylabel("Predicted log return")
    ax.set_title(
        "Predicted vs actual log returns on test set\n"
        "A good model's points hug the red line; a flat cloud means the model "
        "is mostly predicting the mean."
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("days_ahead")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "2_predicted_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# -------- figure 3: residual histogram --------
def plot_residual_histogram(preds, y_test):
    residuals = y_test.values - preds
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=100, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero error")
    ax.axvline(
        residuals.mean(),
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"mean = {residuals.mean():.4f}",
    )
    ax.set_xlabel("Residual (actual − predicted, log return)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Distribution of prediction errors on test set\n"
        "Ideally centered at zero and roughly symmetric. A shifted mean "
        "indicates systematic bias."
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "3_residual_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# -------- figure 4: residuals over time --------
def plot_residuals_over_time(test_df, preds, y_test):
    residuals = y_test.values - preds
    # aggregate to daily mean absolute residual so the line is readable
    tmp = pd.DataFrame(
        {"abs_resid": np.abs(residuals)},
        index=test_df.index,
    )
    daily_mae = tmp.groupby(tmp.index).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        daily_mae.index,
        daily_mae["abs_resid"],
        color="steelblue",
        linewidth=0.8,
        alpha=0.7,
    )
    # rolling mean for a cleaner trend line
    rolling = daily_mae["abs_resid"].rolling(21).mean()
    ax.plot(
        rolling.index,
        rolling,
        color="darkred",
        linewidth=2,
        label="21-day rolling mean",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily mean absolute residual")
    ax.set_title(
        "Prediction error across the test window\n"
        "A flat trend means the model generalizes consistently; spikes "
        "correspond to volatile market periods."
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "4_residuals_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# -------- figure 5: feature importance --------
def plot_feature_importance(model):
    fig, ax = plt.subplots(figsize=(10, 7))
    # "gain" = average improvement in loss when this feature is used for a split.
    # More meaningful than "weight" (raw split count) for assessing usefulness.
    xgb.plot_importance(model, ax=ax, importance_type="gain", show_values=False)
    ax.set_title(
        "Feature importance (gain)\n"
        "Shows which features most reduce prediction error across all 500 trees."
    )
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "5_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# -------- main --------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading model...")
    model, feature_cols = load_model()

    test_df = rebuild_test_set()
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    print("Running predictions on test set...")
    preds = model.predict(X_test)

    # overall metrics (sanity check — should match training output)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  Test MAE: {mae:.4f}")
    print(f"  Test R^2: {r2:.4f}")

    print("\nGenerating figures...")
    plot_per_horizon_mae(test_df, preds, y_test)
    plot_predicted_vs_actual(test_df, preds, y_test)
    plot_residual_histogram(preds, y_test)
    plot_residuals_over_time(test_df, preds, y_test)
    plot_feature_importance(model)

    print(f"\nDone. All figures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
